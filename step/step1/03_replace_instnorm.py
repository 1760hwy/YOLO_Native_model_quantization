#!/usr/bin/env python3
"""
Step 1-3: 替换 InstanceNormalization → GlobalAveragePool 等价子图
目的: 消除 hb_mapper 将 ReduceMean 分配到 CPU 的问题

ONNX 中 GroupNorm 的实际形态:
  Reshape:       (1, C, H, W) → (1, G, L)    [L = (C//G)*H*W]
  InstanceNorm:  scale=ones(G), bias=zeros(G)  [ReduceMean over axis=2]
  Reshape back:  (1, G, L) → (1, C, H, W)
  Mul + Add:     实际 γ/β (学习参数) 在此处应用

替换策略 (3D InstanceNorm → 4D GlobalAveragePool):
  InstanceNorm(x: N,G,L) 等价于:
    x4   = Reshape(x, N,G,1,L)       [→ 4D]
    mean = GlobalAveragePool(x4)      [→ N,G,1,1]  BPU 支持!
    xc   = x4 - mean
    var  = GlobalAveragePool(xc*xc)   [→ N,G,1,1]
    std  = Sqrt(var + eps)
    norm = xc / std
    out  = Reshape(norm, N,G,L)       [→ 3D, 恢复原 shape]
    (scale=ones bias=zeros 时可省略 scale/bias 应用)

产物: step/step1/best_gap.onnx

运行:
  C:\\d\\Anaconda3\\envs\\yolo\\python.exe step/step1/03_replace_instnorm.py
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

STEP1_DIR = Path(__file__).parent
SRC_ONNX  = STEP1_DIR / "best.onnx"
DST_ONNX  = STEP1_DIR / "best_gap.onnx"


def build_const_dict(graph) -> dict:
    """提取图中所有 Constant 节点的值"""
    d = {}
    for n in graph.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value":
                    d[n.output[0]] = nph.to_array(attr.t)
        # initializer 也加进去
    for init in graph.initializer:
        d[init.name] = nph.to_array(init)
    return d


def replace_instance_norm(model: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(model)
    graph = model.graph

    const_dict = build_const_dict(graph)

    inst_norm_nodes = [n for n in graph.node if n.op_type == "InstanceNormalization"]
    print(f"[BP-1] 找到 {len(inst_norm_nodes)} 个 InstanceNormalization 节点")

    new_inits = []
    replacements = {}     # id(node) → [替换节点列表]

    for node in inst_norm_nodes:
        x_in     = node.input[0]   # shape: (N, G, L)  3D
        scale_in = node.input[1]   # shape: (G,)  通常全1
        bias_in  = node.input[2]   # shape: (G,)  通常全0
        y_out    = node.output[0]  # shape: (N, G, L)

        eps = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = float(attr.f)

        nm = node.name or f"IN_{y_out}"
        print(f"  → {nm}")

        scale_arr = const_dict.get(scale_in)
        bias_arr  = const_dict.get(bias_in)

        if scale_arr is None or bias_arr is None:
            print(f"    [SKIP] 无法提取 scale/bias，跳过")
            continue

        G = scale_arr.shape[0]
        trivial_scale = np.allclose(scale_arr, 1.0)
        trivial_bias  = np.allclose(bias_arr,  0.0)
        print(f"    G={G}, eps={eps:.2e}, trivial_scale={trivial_scale}, trivial_bias={trivial_bias}")

        # ── 常量: eps ──────────────────────────────────────────────────────────
        eps_cst_name = nm + "_eps_cst"
        new_inits.append(nph.from_array(np.array([[[[eps]]]], dtype=np.float32),
                                        name=eps_cst_name))

        # ── 4D reshape shape: (N, G, 1, L) ────────────────────────────────────
        # 用 [-1, G, 1, -1] 可能有多个 -1，改用 [0, G, 1, -1]
        # 但 ONNX Reshape 只允许一个 -1；另一个用 0（复制输入维度）
        # 实际: dim0=N(copy), dim1=G(fixed), dim2=1(fixed), dim3=L(infer)
        shape_4d_name = nm + "_shape4d"
        new_inits.append(nph.from_array(
            np.array([0, G, 1, -1], dtype=np.int64), name=shape_4d_name
        ))
        # 恢复原 3D shape 的常量
        shape_3d_name = nm + "_shape3d"
        new_inits.append(nph.from_array(
            np.array([0, G, -1], dtype=np.int64), name=shape_3d_name
        ))

        # ── 中间张量名 ─────────────────────────────────────────────────────────
        t_x4      = nm + "_x4"      # (N, G, 1, L)
        t_mean    = nm + "_mean"    # (N, G, 1, 1)
        t_center  = nm + "_center"  # (N, G, 1, L)
        t_sq      = nm + "_sq"      # (N, G, 1, L)
        t_var     = nm + "_var"     # (N, G, 1, 1)
        t_vareps  = nm + "_vareps"  # (N, G, 1, 1)
        t_std     = nm + "_std"     # (N, G, 1, 1)
        t_norm    = nm + "_norm"    # (N, G, 1, L)
        t_out_4d  = nm + "_out4d"   # before scale/bias or before 3D reshape

        # ── 构建替换节点 ───────────────────────────────────────────────────────
        sub_nodes = [
            # 1. (N,G,L) → (N,G,1,L)   为 GAP 提供 4D 输入
            helper.make_node("Reshape", [x_in, shape_4d_name], [t_x4],    name=nm+"_rs4d"),
            # 2. GlobalAveragePool: mean over last two dims → (N,G,1,1)   BPU 池化!
            helper.make_node("GlobalAveragePool", [t_x4],        [t_mean], name=nm+"_gap1"),
            # 3. x - mean  (broadcast (N,G,1,L) - (N,G,1,1))
            helper.make_node("Sub",  [t_x4, t_mean],             [t_center], name=nm+"_sub"),
            # 4. (x-mean)^2
            helper.make_node("Mul",  [t_center, t_center],       [t_sq],   name=nm+"_sq"),
            # 5. var = GAP((x-mean)^2)   → (N,G,1,1)
            helper.make_node("GlobalAveragePool", [t_sq],         [t_var],  name=nm+"_gap2"),
            # 6. var + eps
            helper.make_node("Add",  [t_var, eps_cst_name],      [t_vareps], name=nm+"_addeps"),
            # 7. sqrt(var+eps)  → BPU HzLut
            helper.make_node("Sqrt", [t_vareps],                 [t_std],  name=nm+"_sqrt"),
            # 8. (x-mean)/std
            helper.make_node("Div",  [t_center, t_std],          [t_norm], name=nm+"_div"),
        ]

        if trivial_scale and trivial_bias:
            # scale=1 bias=0 → 直接 reshape 回 3D
            sub_nodes.append(
                helper.make_node("Reshape", [t_norm, shape_3d_name], [y_out], name=nm+"_rs3d")
            )
        else:
            # 应用非平凡 scale/bias
            s4d_name = nm + "_scale4d"
            b4d_name = nm + "_bias4d"
            new_inits += [
                nph.from_array(scale_arr.reshape(1, G, 1, 1).astype(np.float32), name=s4d_name),
                nph.from_array(bias_arr.reshape(1, G, 1, 1).astype(np.float32),  name=b4d_name),
            ]
            t_scaled = nm + "_scaled"
            sub_nodes += [
                helper.make_node("Mul", [t_norm, s4d_name],    [t_scaled], name=nm+"_scale"),
                helper.make_node("Add", [t_scaled, b4d_name],  [t_out_4d], name=nm+"_bias"),
                helper.make_node("Reshape", [t_out_4d, shape_3d_name], [y_out], name=nm+"_rs3d"),
            ]

        replacements[id(node)] = sub_nodes

    if not replacements:
        print("[WARN] 没有节点被替换!")
        return model

    # ── 将 eps 常量加到 initializer ─────────────────────────────────────────
    graph.initializer.extend(new_inits)

    # ── 重建节点列表 ─────────────────────────────────────────────────────────
    new_nodes = []
    for n in graph.node:
        if id(n) in replacements:
            new_nodes.extend(replacements[id(n)])
        else:
            new_nodes.append(n)

    del graph.node[:]
    graph.node.extend(new_nodes)

    print(f"[BP-1] 替换完成，共替换 {len(replacements)} 个节点")
    return model


def verify_numerical(src: Path, dst: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[BP-3] onnxruntime 未安装，跳过验证")
        return

    np.random.seed(0)
    # 用接近真实的输入范围
    dummy = (np.random.randn(1, 3, 640, 640) * 64 + 128).astype(np.float32)

    providers = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.log_severity_level = 3

    s_orig = ort.InferenceSession(str(src), sess_options=so, providers=providers)
    s_new  = ort.InferenceSession(str(dst), sess_options=so, providers=providers)
    iname  = s_orig.get_inputs()[0].name

    out_orig = s_orig.run(None, {iname: dummy})
    out_new  = s_new.run(None,  {iname: dummy})

    print(f"[BP-3] 数值对比 ({len(out_orig)} 个输出):")
    all_ok = True
    for i, (a, b) in enumerate(zip(out_orig, out_new)):
        md = float(np.abs(a - b).max())
        cs = float(np.dot(a.ravel(), b.ravel()) /
                   (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        ok = md < 1e-3
        if not ok:
            all_ok = False
        print(f"  out[{i}] max_diff={md:.2e}  cosine={cs:.6f}  [{'OK' if ok else 'WARN'}]")

    print("[BP-3] 验证" + ("通过 ✓" if all_ok else "存在差异，请检查!"))


def main():
    print("=" * 60)
    print("STEP 1-3: InstanceNorm → GlobalAveragePool (4D) 替换")
    print(f"  src: {SRC_ONNX}")
    print(f"  dst: {DST_ONNX}")
    print("=" * 60)

    if not SRC_ONNX.exists():
        print(f"[ERROR] 找不到: {SRC_ONNX}")
        sys.exit(1)

    model = onnx.load(str(SRC_ONNX))
    print(f"[BP-0] 原始节点数: {len(model.graph.node)}")

    model_new = replace_instance_norm(model)
    print(f"[BP-1] 新节点数:   {len(model_new.graph.node)}")

    print("[BP-2] ONNX checker ...")
    try:
        onnx.checker.check_model(model_new)
        print("[BP-2] check_model OK")
    except onnx.checker.ValidationError as e:
        print(f"[BP-2] 警告: {e}")

    onnx.save(model_new, str(DST_ONNX))
    size_mb = DST_ONNX.stat().st_size / 1024 / 1024
    print(f"[BP-2] 保存: {DST_ONNX}  ({size_mb:.2f} MB)")

    print("[BP-3] 数值等价验证 ...")
    verify_numerical(SRC_ONNX, DST_ONNX)

    print("=" * 60)
    print("完成! 下一步:")
    print("  1. scp step/step1/best_gap.onnx user@ubuntu:/data/fire_detect.onnx")
    print("  2. 重新 hb_mapper 量化 — GlobalAveragePool 应全落 BPU")
    print("=" * 60)


if __name__ == "__main__":
    main()
