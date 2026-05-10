#!/usr/bin/env python3
"""
EI_I1 — 诊断可视化（论文图表生成）
====================================
读取四组模型的 cls logits .npy 文件，生成：
  Fig 1: 四组 cls logits 分布对比直方图（论文核心图）
  Fig 2: 四组关键指标对比柱状图

Run on PC after copying results/ from board:
  python3 plot_diagnosis.py --results_dir ./results --out_dir ./figures
"""

import json
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib 未安装，跳过图表生成，仅输出文本报告")


GROUPS = [
    ("fire_detect_g1_wrong_norm", "G1: Double Norm",      "#d62728", "Input Semantic Mismatch"),
    ("fire_detect_g2_no_fire",    "G2: No Fire Calib",    "#ff7f0e", "Sample Dist. Mismatch"),
    ("fire_detect_g3_int8_cls",   "G3: INT8 cls",         "#9467bd", "Quant Precision Mismatch"),
    ("fire_detect_g4_full_fix",   "G4: Full Fix (Ours)",  "#2ca02c", "Full Fix Baseline"),
]


def load_group(results_dir: Path, stem: str):
    logit_path  = results_dir / stem / "cls_logits_all.npy"
    report_path = results_dir / stem / "logit_report.json"
    logits  = np.load(str(logit_path)).ravel() if logit_path.exists() else None
    report  = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    return logits, report


def print_text_report(data):
    print()
    print("=" * 72)
    print("  EI_I1 诊断报告: 三层校准失配对照")
    print("=" * 72)
    header = f"{'组别':<20} {'logit均值':>10} {'logit最大':>10} {'正值占比%':>10} {'检测框数':>8}"
    print(header)
    print("-" * 72)
    for stem, label, color, tag in GROUPS:
        logits, report = data[stem]
        if logits is None:
            print(f"  {label:<18} {'(无数据)'}")
            continue
        mean = report.get("logit_mean", float(logits.mean()))
        maxv = report.get("logit_max",  float(logits.max()))
        posp = report.get("logit_pct_positive", float((logits > 0).mean()) * 100)
        ndet = report.get("total_det", "?")
        print(f"  {label:<18} {mean:>10.4f} {maxv:>10.4f} {posp:>10.2f} {ndet:>8}")
    print("=" * 72)


def plot_logit_histograms(data, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.ravel()
    fig.suptitle("EI-I1: cls Branch Logit Distribution -- Three-Level Calibration Mismatch",
                 fontsize=13, fontweight="bold", y=0.98)

    x_min, x_max = -45, 10

    for ax, (stem, label, color, tag) in zip(axes, GROUPS):
        logits, report = data[stem]
        if logits is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{label}\n({tag})", fontsize=10)
            continue

        ax.hist(logits, bins=120, range=(x_min, x_max),
                color=color, alpha=0.75, edgecolor="none", density=True)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="logit = 0")

        mean = float(logits.mean())
        maxv = float(logits.max())
        posp = float((logits > 0).mean()) * 100
        ndet = report.get("total_det", "?")

        ax.axvline(mean, color=color, linewidth=1.8, linestyle="-",
                   alpha=0.9, label=f"mean={mean:.2f}")

        info = (f"mean={mean:.2f}\nmax={maxv:.2f}\n"
                f"P(>0)={posp:.1f}%\ndet={ndet}")
        ax.text(0.97, 0.97, info, transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(f"{label}\n({tag})", fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("cls logit value", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = out_dir / "fig1_logit_distribution.png"
    plt.savefig(str(fig_path), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] Fig 1 → {fig_path}")


def plot_bar_comparison(data, out_dir: Path):
    labels, means, maxvs, posps, ndets = [], [], [], [], []
    for stem, label, color, tag in GROUPS:
        logits, report = data[stem]
        if logits is None:
            continue
        labels.append(label.replace(": ", "\n"))
        means.append(report.get("logit_mean", float(logits.mean())))
        maxvs.append(report.get("logit_max",  float(logits.max())))
        posps.append(report.get("logit_pct_positive", float((logits > 0).mean()) * 100))
        ndets.append(report.get("total_det", 0))

    colors = [c for stem, label, c, tag in GROUPS if data[stem][0] is not None]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("EI-I1: Key Metrics Comparison Across Groups (Three-Level Calibration Mismatch)", fontsize=12, fontweight="bold")

    x = np.arange(len(labels))
    w = 0.5

    # logit mean
    axes[0].bar(x, means, width=w, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("cls Logit Mean", fontweight="bold")
    axes[0].set_ylabel("mean logit value")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(means):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

    # positive ratio
    axes[1].bar(x, posps, width=w, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Positive Logit Ratio (%)", fontweight="bold")
    axes[1].set_ylabel("% pixels with logit > 0")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(posps):
        axes[1].text(i, v + 0.1, f"{v:.2f}%", ha="center", fontsize=8)

    # detection count
    axes[2].bar(x, ndets, width=w, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    axes[2].set_title("Total Detections (conf>0.1)", fontweight="bold")
    axes[2].set_ylabel("number of detections")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(ndets):
        axes[2].text(i, v + 0.3, str(v), ha="center", fontsize=8)

    plt.tight_layout()
    fig_path = out_dir / "fig2_metrics_comparison.png"
    plt.savefig(str(fig_path), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] Fig 2 → {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="EI_I1 logits 诊断可视化")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--out_dir",     default="./figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {stem: load_group(results_dir, stem) for stem, *_ in GROUPS}

    missing = [stem for stem, *_ in GROUPS if data[stem][0] is None]
    if missing:
        print(f"[WARN] 缺少数据: {missing}")

    print_text_report(data)

    if HAS_MPL:
        plot_logit_histograms(data, out_dir)
        plot_bar_comparison(data, out_dir)
    else:
        print("\n安装 matplotlib 后可生成图表: pip install matplotlib")


if __name__ == "__main__":
    main()
