#!/usr/bin/env python3
"""
Step 1-1b: GroupNorm → BatchNorm 替换后导出 ONNX

原因:
  Detect_LSDECD 使用 Conv_GN / DEConv_GN，内部含 nn.GroupNorm(16, dim)
  GroupNorm 在 ONNX 中分解为 ReduceMean+Sub+Mul+Add，BPU 不支持 ReduceMean
  导致整个检测头 (model.24) 全部在 CPU 运行，帧率从 90+ 暴跌到个位数

方法 (无需重新训练):
  1. 加载 best.pt，注册 forward hook 收集每个 GroupNorm 层的逐通道输入均值/方差
  2. 用 step3/calibration_f32/ 中 30 张 .f32 图片运行模型
  3. 将所有 GroupNorm 替换为 BatchNorm2d (使用收集到的统计量作为 running_mean/var)
  4. 用 torch.onnx.export 直接导出 ONNX (opset=11)

BatchNormalization 是 BPU 原生支持算子，可完全在 BPU 上运行。

替换目标 (5 个 GroupNorm):
  model.24.conv[0/1/2].0.gn  ← Conv_GN.gn  (stride 8/16/32 输入投影)
  model.24.share_conv[0/1].bn ← DEConv_GN.bn (共享细节增强卷积)

断点说明:
  BP-A: 加载 best.pt
  BP-B: 注册 hook + 运行校准数据收集 GN 统计量
  BP-C: 替换 GroupNorm → BatchNorm2d
  BP-D: 导出并验证 ONNX

运行方式:
  C:\\d\\Anaconda3\\envs\\yolo\\python.exe step/step1/01b_gn_to_bn.py
"""

import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import onnx
from pathlib import Path
from typing import Dict, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
STEP1_DIR = Path(__file__).parent
STEP3_DIR = STEP1_DIR.parent / "step3"

MODEL_PT  = ROOT / "yolo11-HGNetV2-C3k2-DWR-LSDECD-SlideLoss" / "weights" / "best.pt"
CAL_DIR   = STEP3_DIR / "calibration_f32"
ONNX_OUT  = STEP1_DIR / "best.onnx"

IMGSZ       = 640
OPSET       = 11
N_CAL       = 30    # 校准样本数，越多统计越准，30 已足够

# ── 日志 ──────────────────────────────────────────────────────────────────────
LOG_FILE = STEP1_DIR / "step1.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step1_01b")


# ─────────────────────────────────────────────────────────────────────────────
# 1. 统计收集
# ─────────────────────────────────────────────────────────────────────────────

def collect_gn_stats(model_nn: nn.Module, cal_dir: Path, n_samples: int) -> Dict:
    """
    对每个 GroupNorm 层注册 forward hook，
    运行 n_samples 张校准图，收集逐通道输入均值和方差。

    返回 dict: {module_fullname -> {'running_mean': Tensor, 'running_var': Tensor}}
    """
    stats: Dict = {}
    hooks: List = []

    # 注册 hook
    for name, module in model_nn.named_modules():
        if not isinstance(module, nn.GroupNorm):
            continue

        def _make_hook(n: str):
            def _hook(mod: nn.Module, inp: Tuple, out: torch.Tensor) -> None:
                x = inp[0].detach().float()      # (1, C, H, W)
                c_mean = x.mean(dim=[0, 2, 3])   # (C,)  per-channel mean over spatial
                c_var  = x.var( dim=[0, 2, 3], unbiased=False)  # (C,)
                if n not in stats:
                    stats[n] = {
                        "mean_acc": torch.zeros_like(c_mean),
                        "var_acc":  torch.zeros_like(c_var),
                        "count":    0,
                        "C":        x.shape[1],
                    }
                stats[n]["mean_acc"] += c_mean
                stats[n]["var_acc"]  += c_var
                stats[n]["count"]    += 1
            return _hook

        hooks.append(module.register_forward_hook(_make_hook(name)))
        log.info(f"  [hook] {name}  (GroupNorm {module.num_groups}g, {module.num_channels}ch)")

    if not hooks:
        log.warning("  未找到任何 GroupNorm 层，跳过统计")
        return {}

    # 运行校准图
    cal_files = sorted(cal_dir.glob("*.f32"))[:n_samples]
    if not cal_files:
        log.error(f"  [BP-B] 校准目录无 .f32 文件: {cal_dir}")
        sys.exit(1)
    log.info(f"  [BP-B] 运行 {len(cal_files)} 张校准图 ...")

    model_nn.eval()
    with torch.no_grad():
        for i, f32_path in enumerate(cal_files, 1):
            # .f32 文件: float32 NCHW 0-255 (由 step3/01_prepare_calibration.py 生成)
            arr = np.fromfile(str(f32_path), dtype=np.float32)
            if arr.size != IMGSZ * IMGSZ * 3:
                log.warning(f"  跳过异常文件 {f32_path.name} (size={arr.size})")
                continue
            img_t = torch.from_numpy(arr).reshape(1, 3, IMGSZ, IMGSZ) / 255.0
            model_nn(img_t)
            if i % 10 == 0:
                log.info(f"  [{i}/{len(cal_files)}] 已处理")

    # 移除 hook
    for h in hooks:
        h.remove()

    # 计算平均统计量
    result: Dict = {}
    for name, s in stats.items():
        n = s["count"]
        result[name] = {
            "running_mean": s["mean_acc"] / n,
            "running_var":  s["var_acc"]  / n,
        }
        log.info(
            f"  [stat] {name}: count={n}, "
            f"mean=[{result[name]['running_mean'].min():.3f}, {result[name]['running_mean'].max():.3f}], "
            f"var=[{result[name]['running_var'].min():.3f}, {result[name]['running_var'].max():.3f}]"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. GroupNorm → BatchNorm2d 替换
# ─────────────────────────────────────────────────────────────────────────────

def replace_gn_with_bn(model_nn: nn.Module, stats: Dict) -> int:
    """
    遍历所有 GroupNorm 模块，用 BatchNorm2d 替换。
    BatchNorm 的 running_mean/var 来自校准统计，weight/bias 从 GroupNorm 复制。
    """
    to_replace: List[Tuple[str, nn.GroupNorm]] = [
        (name, module)
        for name, module in model_nn.named_modules()
        if isinstance(module, nn.GroupNorm)
    ]

    replaced = 0
    for name, gn_module in to_replace:
        if name not in stats:
            log.warning(f"  [skip] {name} 无统计量，跳过（未被校准数据激活）")
            continue

        C = gn_module.num_channels
        bn = nn.BatchNorm2d(C, eps=gn_module.eps, momentum=0.1, affine=True, track_running_stats=True)

        # 复制仿射参数 (gamma / beta)
        bn.weight = nn.Parameter(gn_module.weight.data.clone())
        bn.bias   = nn.Parameter(gn_module.bias.data.clone())

        # 设置固定 running stats（来自校准数据）
        bn.running_mean.copy_(stats[name]["running_mean"])
        bn.running_var.copy_(stats[name]["running_var"])
        bn.num_batches_tracked.fill_(1)  # 标记已有统计

        # 锁定为 eval 模式（不更新 running stats）
        bn.eval()

        # 找到父模块并替换属性
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = model_nn.get_submodule(parent_name)
        else:
            parent, attr = model_nn, parts[0]

        setattr(parent, attr, bn)
        log.info(f"  [replace] {name}: GroupNorm({gn_module.num_groups}g, {C}ch) → BatchNorm2d({C}ch)")
        replaced += 1

    return replaced


# ─────────────────────────────────────────────────────────────────────────────
# 3. 导出 ONNX
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(model_nn: nn.Module, out_path: Path) -> None:
    model_nn.eval()
    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ)

    log.info(f"  [BP-D] torch.onnx.export → {out_path}")
    torch.onnx.export(
        model_nn,
        dummy,
        str(out_path),
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["images"],
        verbose=False,
    )

    # 验证
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    size_mb     = out_path.stat().st_size / 1024 / 1024
    n_inputs    = len(onnx_model.graph.input)
    n_outputs   = len(onnx_model.graph.output)
    opset_ver   = onnx_model.opset_import[0].version

    log.info(f"  [BP-D] 文件大小:   {size_mb:.2f} MB")
    log.info(f"  [BP-D] opset:      {opset_ver}")
    log.info(f"  [BP-D] 输入数量:   {n_inputs}")
    log.info(f"  [BP-D] 输出数量:   {n_outputs}")

    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        log.info(f"  [BP-D]   输入: {inp.name} → {shape}")
    for i, out in enumerate(onnx_model.graph.output):
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        log.info(f"  [BP-D]   输出{i}: {out.name} → {shape}")

    if n_outputs != 6:
        log.error(f"  [BP-D] 输出数量应为 6，实际 {n_outputs}，请检查 head.py 是否已执行 01_modify_head.py")
        sys.exit(1)

    # 验证不再含有 GroupNorm 产生的 ReduceMean
    gn_nodes = [n for n in onnx_model.graph.node if n.op_type == "ReduceMean"]
    if gn_nodes:
        log.warning(f"  [BP-D] 仍有 {len(gn_nodes)} 个 ReduceMean 节点（非 GroupNorm 来源，可忽略）")
    else:
        log.info("  [BP-D] ReduceMean 节点: 0 ✓ (GroupNorm 已完全消除)")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("STEP 1-1b: GroupNorm → BatchNorm + ONNX 导出")
    log.info(f"  模型:     {MODEL_PT}")
    log.info(f"  校准数据: {CAL_DIR}")
    log.info(f"  输出:     {ONNX_OUT}")
    log.info("=" * 60)

    # ── BP-A: 加载模型 ────────────────────────────────────────────────────────
    log.info("[BP-A] 加载 best.pt ...")
    if not MODEL_PT.exists():
        log.error(f"[BP-A] 模型文件不存在: {MODEL_PT}")
        sys.exit(1)
    if not CAL_DIR.exists():
        log.error(f"[BP-A] 校准数据目录不存在: {CAL_DIR}，请先运行 step3/01_prepare_calibration.py")
        sys.exit(1)

    sys.path.insert(0, str(ROOT))
    from ultralytics import YOLO

    wrapper   = YOLO(str(MODEL_PT))
    model_nn  = wrapper.model.float()
    model_nn.eval()
    log.info("[BP-A] 模型加载成功 ✓")

    # 统计模型中的 GroupNorm 数量
    gn_list = [(n, m) for n, m in model_nn.named_modules() if isinstance(m, nn.GroupNorm)]
    log.info(f"[BP-A] 发现 GroupNorm 层: {len(gn_list)} 个")
    for n, m in gn_list:
        log.info(f"         {n}  (num_groups={m.num_groups}, num_channels={m.num_channels})")

    # ── BP-B: 收集 GroupNorm 统计量 ───────────────────────────────────────────
    log.info(f"\n[BP-B] 收集 GroupNorm 逐通道统计量 (N={N_CAL} 张校准图) ...")
    gn_stats = collect_gn_stats(model_nn, CAL_DIR, N_CAL)
    log.info(f"[BP-B] 收集完成，共 {len(gn_stats)} 层有统计量 ✓")

    # ── BP-C: 替换 GroupNorm → BatchNorm2d ────────────────────────────────────
    log.info("\n[BP-C] 替换 GroupNorm → BatchNorm2d ...")
    n_replaced = replace_gn_with_bn(model_nn, gn_stats)
    log.info(f"[BP-C] 成功替换 {n_replaced} 个 GroupNorm ✓")

    if n_replaced == 0:
        log.error("[BP-C] 未替换任何层，请检查校准数据和模型")
        sys.exit(1)

    # 验证：确认不再有 GroupNorm
    remaining = [(n, m) for n, m in model_nn.named_modules() if isinstance(m, nn.GroupNorm)]
    if remaining:
        log.warning(f"[BP-C] 仍有 {len(remaining)} 个 GroupNorm 未替换（无统计量）:")
        for n, _ in remaining:
            log.warning(f"         {n}")
    else:
        log.info("[BP-C] 所有 GroupNorm 已替换 ✓")

    # 保持替换后的 BN 使用 running stats（不更新）
    model_nn.eval()

    # ── BP-D: 导出 ONNX ───────────────────────────────────────────────────────
    log.info(f"\n[BP-D] 导出 ONNX (opset={OPSET}) ...")
    export_onnx(model_nn, ONNX_OUT)

    log.info("=" * 60)
    log.info("STEP 1-1b 完成 ✅")
    log.info(f"  产物: {ONNX_OUT}")
    log.info("  下一步: python step/step2/01_sample_images.py (若未运行)")
    log.info("          python step/step2/02_validate_onnx.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
