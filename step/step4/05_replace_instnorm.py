#!/usr/bin/env python3
"""
Step 4-5: InstanceNormalization → GlobalAveragePool 替换
目的: 消除 hb_mapper 将 ReduceMean 分配到 CPU 导致的子图碎片化问题

问题根因:
  BPU 的 HzSQuantizedReduceMean 不支持 InstanceNorm 展开后的某些张量形状，
  导致 ReduceMean 落 CPU，形成 13 个 BPU 子图，FPS 从理论 44 降到实际 2.4。

替换策略 (InstanceNorm 3D → 4D GlobalAveragePool):
  ONNX 中的 InstanceNorm 输入 shape: (N, G, L)  [GroupNorm 展开后]
  GlobalAveragePool 要求 4D 输入，因此:
    x4   = Reshape(x, [N, G, 1, L])       → 插入空间维度
    mean = GlobalAveragePool(x4)          → (N,G,1,1)  BPU 原生支持!
    xc   = x4 - mean                      → 中心化
    var  = GlobalAveragePool(xc * xc)     → (N,G,1,1)  BPU 原生支持!
    std  = Sqrt(var + eps)                → HzLut
    norm = xc / std                       → 归一化
    (scale/bias 非 1/0 时应用 affine 变换)
    out  = Reshape(norm, [N, G, L])       → 恢复原 shape

输入: step4/fire_detect.onnx  (含 NHWC 6-output Detect_LSDECD 头)
输出: step4/fire_detect_gap.onnx

运行 (Windows):
  C:\\d\\Anaconda3\\envs\\yolo\\python.exe step/step4/05_replace_instnorm.py

Ubuntu 后续:
  scp step4/fire_detect_gap.onnx sunrise@<ip>:/data/  (或 scp 到 Docker)
  修改 04_quantization_config.yaml: onnx_model: '/data/fire_detect_gap.onnx'
  重新 hb_mapper makertbin
"""

import sys
import copy
import numpy as np
import onnx
import onnx.numpy_helper as nph
from onnx import helper
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

STEP4_DIR = Path(__file__).parent
SRC_ONNX  = STEP4_DIR / "fire_detect.onnx"
DST_ONNX  = STEP4_DIR / "fire_detect_gap.onnx"


def build_const_dict(graph: onnx.GraphProto) -> dict:
    d = {}
    for n in graph.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value":
                    d[n.output[0]] = nph.to_array(attr.t)
    for init in graph.initializer:
        d[init.name] = nph.to_array(init)
    return d


def replace_instance_norm(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """
    替换所有 InstanceNormalization 节点为 GlobalAveragePool 等价子图。
    返回 (新模型, 替换数量)
    """
    model = copy.deepcopy(model)
    graph = model.graph
    const_dict = build_const_dict(graph)

    inst_norm_nodes = [n for n in graph.node if n.op_type == "InstanceNormalization"]
    print(f"[替换] 找到 {len(inst_norm_nodes)} 个 InstanceNormalization 节点")

    new_inits = []
    replacements = {}

    for node in inst_norm_nodes:
        x_in     = node.input[0]   # (N, G, L)
        scale_in = node.input[1]   # (G,)  GroupNorm 学习参数 γ
        bias_in  = node.input[2]   # (G,)  GroupNorm 学习参数 β
        y_out    = node.output[0]  # (N, G, L)

        eps = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = float(attr.f)

        nm = node.name or f"IN_{y_out}"
        scale_arr = const_dict.get(scale_in)
        bias_arr  = const_dict.get(bias_in)

        if scale_arr is None or bias_arr is None:
            print(f"  [SKIP] {nm}: 无法提取 scale/bias")
            continue

        G = scale_arr.shape[0]
        trivial_scale = np.allclose(scale_arr, 1.0)
        trivial_bias  = np.allclose(bias_arr,  0.0)
        print(f"  → {nm}  G={G}  eps={eps:.2e}  "
              f"affine={'trivial' if trivial_scale and trivial_bias else 'learned'}")

        # ── 常量 ────────────────────────────────────────────────────────────────
        eps_name    = nm + "_eps_cst"
        shape4d_name = nm + "_shape4d"
        shape3d_name = nm + "_shape3d"
        new_inits += [
            nph.from_array(np.array([[[[eps]]]], dtype=np.float32), name=eps_name),
            nph.from_array(np.array([0, G, 1, -1], dtype=np.int64),  name=shape4d_name),
            nph.from_array(np.array([0, G, -1],    dtype=np.int64),  name=shape3d_name),
        ]

        # ── 中间张量名 ──────────────────────────────────────────────────────────
        t_x4     = nm + "_x4"
        t_mean   = nm + "_mean"
        t_center = nm + "_center"
        t_sq     = nm + "_sq"
        t_var    = nm + "_var"
        t_vareps = nm + "_vareps"
        t_std    = nm + "_std"
        t_norm   = nm + "_norm"

        sub_nodes = [
            # 1. (N,G,L) → (N,G,1,L)
            helper.make_node("Reshape", [x_in, shape4d_name], [t_x4],    name=nm+"_rs4d"),
            # 2. GlobalAveragePool → (N,G,1,1)  ← BPU 池化算子，全量支持
            helper.make_node("GlobalAveragePool", [t_x4], [t_mean],      name=nm+"_gap1"),
            # 3. x - mean
            helper.make_node("Sub",  [t_x4, t_mean],     [t_center],    name=nm+"_sub"),
            # 4. (x-mean)²
            helper.make_node("Mul",  [t_center, t_center], [t_sq],      name=nm+"_sq"),
            # 5. var = GAP((x-mean)²)  ← BPU 池化算子
            helper.make_node("GlobalAveragePool", [t_sq], [t_var],       name=nm+"_gap2"),
            # 6. var + eps
            helper.make_node("Add",  [t_var, eps_name],  [t_vareps],    name=nm+"_addeps"),
            # 7. sqrt(var+eps)  → HzLut
            helper.make_node("Sqrt", [t_vareps],         [t_std],       name=nm+"_sqrt"),
            # 8. (x-mean)/std
            helper.make_node("Div",  [t_center, t_std],  [t_norm],      name=nm+"_div"),
        ]

        if trivial_scale and trivial_bias:
            sub_nodes.append(
                helper.make_node("Reshape", [t_norm, shape3d_name], [y_out], name=nm+"_rs3d")
            )
        else:
            # 非平凡 affine 参数：broadcast 形式 (1,G,1,1) → (N,G,1,L)
            s4d_name = nm + "_scale4d"
            b4d_name = nm + "_bias4d"
            new_inits += [
                nph.from_array(scale_arr.reshape(1, G, 1, 1).astype(np.float32), name=s4d_name),
                nph.from_array(bias_arr.reshape(1, G, 1, 1).astype(np.float32),  name=b4d_name),
            ]
            t_scaled  = nm + "_scaled"
            t_biased  = nm + "_biased"
            sub_nodes += [
                helper.make_node("Mul", [t_norm, s4d_name],   [t_scaled],  name=nm+"_scale"),
                helper.make_node("Add", [t_scaled, b4d_name], [t_biased],  name=nm+"_bias"),
                helper.make_node("Reshape", [t_biased, shape3d_name], [y_out], name=nm+"_rs3d"),
            ]

        replacements[id(node)] = sub_nodes

    if not replacements:
        print("[WARN] 没有节点被替换!")
        return model, 0

    graph.initializer.extend(new_inits)

    new_nodes = []
    for n in graph.node:
        if id(n) in replacements:
            new_nodes.extend(replacements[id(n)])
        else:
            new_nodes.append(n)
    del graph.node[:]
    graph.node.extend(new_nodes)

    return model, len(replacements)


def verify_numerical(src: Path, dst: Path) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[验证] onnxruntime 未安装，跳过数值验证")
        return True

    np.random.seed(42)
    dummy = (np.random.randn(1, 3, 640, 640) * 64 + 128).astype(np.float32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]

    s_orig = ort.InferenceSession(str(src), sess_options=so, providers=providers)
    s_new  = ort.InferenceSession(str(dst), sess_options=so, providers=providers)
    iname  = s_orig.get_inputs()[0].name

    out_orig = s_orig.run(None, {iname: dummy})
    out_new  = s_new.run(None,  {iname: dummy})

    print(f"\n[验证] 数值对比 ({len(out_orig)} 个输出):")
    print(f"{'输出':>6}  {'max_diff':>12}  {'cosine':>10}  {'状态':>6}")
    print("-" * 48)
    all_ok = True
    for i, (a, b) in enumerate(zip(out_orig, out_new)):
        md = float(np.abs(a - b).max())
        cs = float(np.dot(a.ravel(), b.ravel()) /
                   (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        ok = md < 1e-3 and cs > 0.999
        if not ok:
            all_ok = False
        print(f"  out[{i}]  {md:>12.2e}  {cs:>10.6f}  {'✓' if ok else '⚠️  WARN'}")

    print()
    if all_ok:
        print("[验证] 通过 ✓ — GAP 替换与原 InstanceNorm 数值等价")
    else:
        print("[验证] ⚠️  存在差异，请检查替换逻辑")
    return all_ok


def main():
    print("=" * 60)
    print("STEP 4-5: InstanceNorm → GlobalAveragePool 替换")
    print(f"  输入: {SRC_ONNX}")
    print(f"  输出: {DST_ONNX}")
    print("=" * 60)

    if not SRC_ONNX.exists():
        print(f"[ERROR] 找不到输入 ONNX: {SRC_ONNX}")
        sys.exit(1)

    model = onnx.load(str(SRC_ONNX))
    orig_node_count = len(model.graph.node)
    print(f"[加载] 原始节点数: {orig_node_count}")

    model_new, n_replaced = replace_instance_norm(model)
    print(f"\n[完成] 替换 {n_replaced} 个 InstanceNormalization → GAP 子图")
    print(f"[完成] 新节点数: {len(model_new.graph.node)}  "
          f"(+{len(model_new.graph.node) - orig_node_count} 展开节点)")

    print("\n[检查] onnx.checker ...")
    try:
        onnx.checker.check_model(model_new)
        print("[检查] check_model 通过 ✓")
    except onnx.checker.ValidationError as e:
        print(f"[检查] 警告: {e}")

    onnx.save(model_new, str(DST_ONNX))
    size_mb = DST_ONNX.stat().st_size / 1024 / 1024
    print(f"[保存] {DST_ONNX}  ({size_mb:.2f} MB)")

    print("\n[验证] 数值等价验证 ...")
    verify_numerical(SRC_ONNX, DST_ONNX)

    print("=" * 60)
    print("完成! 下一步:")
    print("  1. 运行 06_validate_gap.py 验证检测效果")
    print("  2. 将 fire_detect_gap.onnx 传到 Ubuntu Docker:")
    print("     scp step4/fire_detect_gap.onnx user@ubuntu:/data/")
    print("  3. 在 Docker 内修改 04_quantization_config.yaml:")
    print("     onnx_model: '/data/fire_detect_gap.onnx'")
    print("  4. 重新运行 hb_mapper makertbin")
    print("=" * 60)


if __name__ == "__main__":
    main()
