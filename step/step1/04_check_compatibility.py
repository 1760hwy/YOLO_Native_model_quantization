#!/usr/bin/env python3
"""
Step 1-4: RDK X5 (bayes-e) 算子兼容性分析
目的: 对照地平线官方 OE v1.2.8 算子支持列表，检查每个算子的兼容状态
      并给出不兼容算子的替换建议

参考文档:
  地平线 OpenExplorer v1.2.8 算子支持列表:
  https://developer.horizon.ai/forumDetail/112555549341653662
  地平线 RDK X5 (bayes-e) ONNX 算子支持:
  horizon_xj3_open_explorer/docs/cn/ddk_doc/module/supported_op_list_and_restrictions/

断点说明:
  BP-A: 加载 operators.json
  BP-B: 逐一比对兼容性数据库
  BP-C: 输出分类结果 (BPU/CPU/需替换)
  BP-D: 保存 operator_compatibility.json

运行方式:
  C:\d\Anaconda3\envs\yolo\python.exe step/step1/04_check_compatibility.py
"""

import sys
import json
import logging
from pathlib import Path

# ── 日志 ─────────────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "step1.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("step1_04")

# ── 路径 ─────────────────────────────────────────────────────────────────────
STEP1_DIR           = Path(__file__).parent
OPERATORS_JSON      = STEP1_DIR / "operators.json"
COMPATIBILITY_JSON  = STEP1_DIR / "operator_compatibility.json"

# ── RDK X5 (bayes-e) 算子兼容性数据库 ─────────────────────────────────────────
# 基于地平线 OpenExplorer v1.2.8 官方文档整理
# status:
#   "BPU"          → 完全在 BPU 运行，INT8 量化支持
#   "BPU_int16"    → 需配置 node_info 使用 INT16 才能在 BPU 运行
#   "CPU"          → 强制在 CPU 运行（不影响功能，影响性能）
#   "BPU_partial"  → BPU 支持但有限制条件
#   "unsupported"  → 当前工具链版本不支持，需替换
#
# workaround: 替换方法或配置说明
RDKX5_OP_DB: dict[str, dict] = {
    # ── 卷积 & 线性 ──────────────────────────────────────────────────────────
    "Conv": {
        "status": "BPU",
        "note": "完全支持，标准二维卷积。kernel size/stride/dilation/group 均支持",
        "workaround": None,
    },
    "ConvTranspose": {
        "status": "BPU",
        "note": "转置卷积，支持常规配置",
        "workaround": None,
    },
    "Gemm": {
        "status": "BPU",
        "note": "全连接/矩阵乘法，支持",
        "workaround": None,
    },
    "MatMul": {
        "status": "BPU",
        "note": "矩阵乘法，二维场景完全支持；高维场景建议验证",
        "workaround": None,
    },

    # ── 激活函数 ──────────────────────────────────────────────────────────────
    "Relu": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "LeakyRelu": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "PRelu": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Sigmoid": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "HardSigmoid": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "HardSwish": {
        "status": "BPU",
        "note": "完全支持 (= x * HardSigmoid(x))",
        "workaround": None,
    },
    "Mish": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Silu": {
        "status": "BPU",
        "note": "SiLU = x * Sigmoid(x)，以 Mul+Sigmoid 形式在 BPU 上运行",
        "workaround": None,
    },
    "Tanh": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Elu": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Selu": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Clip": {
        "status": "BPU",
        "note": "完全支持（常用于 ReLU6）",
        "workaround": None,
    },
    "Gelu": {
        "status": "BPU",
        "note": "bayes-e 支持 Gelu 算子（OE v1.2.8+）",
        "workaround": None,
    },
    "Softmax": {
        "status": "BPU_int16",
        "note": "⚠️ 默认 INT8 量化时 Softmax 落 CPU，导致子图分裂 FPS 暴跌。"
                "需在 hb_mapper YAML 的 node_info 中强制为 BPU + INT16",
        "workaround": (
            "在 hb_mapper YAML 中添加:\n"
            "  node_info:\n"
            "    \"<Softmax节点名>\": {ON: BPU, InputType: int16, OutputType: int16}\n"
            "节点名可由 step 1-3 的 operators.json 查出，或用 hb_mapper checker 确认"
        ),
    },
    "LogSoftmax": {
        "status": "CPU",
        "note": "通常用于分类模型输出层，检测模型中很少出现",
        "workaround": "改用 Softmax + Log，或将 LogSoftmax 移至后处理",
    },

    # ── 归一化 ──────────────────────────────────────────────────────────────
    "BatchNormalization": {
        "status": "BPU",
        "note": "完全支持，训练模式下需设置 training=False",
        "workaround": None,
    },
    "GroupNormalization": {
        "status": "BPU",
        "note": "OE v1.2.8 支持 GroupNorm",
        "workaround": None,
    },
    "InstanceNormalization": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "LayerNormalization": {
        "status": "BPU",
        "note": "bayes-e 架构支持（OE v1.2.8）",
        "workaround": None,
    },

    # ── 池化 ──────────────────────────────────────────────────────────────────
    "MaxPool": {
        "status": "BPU",
        "note": "完全支持，kernel≤11×11 最优",
        "workaround": None,
    },
    "AveragePool": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "GlobalAveragePool": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "GlobalMaxPool": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },

    # ── 元素级运算 ────────────────────────────────────────────────────────────
    "Add": {
        "status": "BPU",
        "note": "完全支持，包含 residual 连接场景",
        "workaround": None,
    },
    "Sub": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Mul": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Div": {
        "status": "BPU",
        "note": "完全支持（常量除法折叠为乘法）",
        "workaround": None,
    },
    "Pow": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Sqrt": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Abs": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Neg": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Exp": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Log": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Min": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Max": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },

    # ── Reduce 运算 ───────────────────────────────────────────────────────────
    "ReduceSum": {
        "status": "BPU",
        "note": "支持常规 axis 配置",
        "workaround": None,
    },
    "ReduceMean": {
        "status": "BPU",
        "note": "支持，常见于 GlobalAveragePool 等价",
        "workaround": None,
    },
    "ReduceMax": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "ReduceMin": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "ReduceProd": {
        "status": "BPU_partial",
        "note": "有限支持，建议验证",
        "workaround": "如落 CPU，将乘法展开为逐元素 Mul",
    },

    # ── Shape/Tensor 操作 ─────────────────────────────────────────────────────
    "Reshape": {
        "status": "BPU",
        "note": "完全支持（静态 shape）",
        "workaround": None,
    },
    "Transpose": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Flatten": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Squeeze": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Unsqueeze": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Concat": {
        "status": "BPU",
        "note": "完全支持，多 tensor 拼接",
        "workaround": None,
    },
    "Split": {
        "status": "BPU",
        "note": "完全支持",
        "workaround": None,
    },
    "Slice": {
        "status": "BPU",
        "note": "静态 slice 完全支持；动态 slice 可能落 CPU",
        "workaround": "确保 start/end/axes/step 为常量",
    },
    "Gather": {
        "status": "BPU_partial",
        "note": "索引为常量时 BPU 支持；动态索引落 CPU",
        "workaround": "将动态 gather 移至后处理，或改用静态 slice",
    },
    "GatherElements": {
        "status": "CPU",
        "note": "复杂动态索引，通常落 CPU",
        "workaround": "重构为等价的静态操作",
    },
    "ScatterElements": {
        "status": "CPU",
        "note": "不支持，通常落 CPU",
        "workaround": "移至后处理 Python 代码",
    },
    "ScatterND": {
        "status": "CPU",
        "note": "不支持，落 CPU",
        "workaround": "移至后处理",
    },
    "GatherND": {
        "status": "CPU",
        "note": "不支持，落 CPU",
        "workaround": "移至后处理",
    },
    "Pad": {
        "status": "BPU",
        "note": "完全支持（constant/reflect/edge padding）",
        "workaround": None,
    },

    # ── 上/下采样 ─────────────────────────────────────────────────────────────
    "Resize": {
        "status": "BPU",
        "note": "支持 nearest/bilinear 插值；静态 scale/size；不支持 cubic",
        "workaround": "如使用 bicubic，改为 bilinear",
    },
    "Upsample": {
        "status": "BPU",
        "note": "nearest/bilinear 均支持",
        "workaround": None,
    },

    # ── 常量/类型 ─────────────────────────────────────────────────────────────
    "Constant": {
        "status": "BPU",
        "note": "常量折叠，不占 BPU 算力",
        "workaround": None,
    },
    "ConstantOfShape": {
        "status": "BPU",
        "note": "静态 shape 时支持",
        "workaround": None,
    },
    "Cast": {
        "status": "BPU",
        "note": "常见类型转换支持（float32/int32/int64/bool）",
        "workaround": None,
    },
    "Shape": {
        "status": "CPU",
        "note": "Shape 算子输出张量形状，通常为常量化后消除；若残留则落 CPU",
        "workaround": "确保使用静态 shape 导出，shape 节点会被常量折叠消除",
    },
    "NonZero": {
        "status": "CPU",
        "note": "动态稀疏算子，不支持 BPU",
        "workaround": "移至后处理",
    },
    "Where": {
        "status": "BPU_partial",
        "note": "条件为常量时 BPU 支持；动态条件落 CPU",
        "workaround": "尽量保证条件为常量",
    },
    "NonMaxSuppression": {
        "status": "CPU",
        "note": "NMS 不在 BPU 运行，需在后处理中实现（hbDNN API 提供 NMS 工具）",
        "workaround": "使用 hbDNN 的 NMS 后处理接口，或自行在 CPU 实现",
    },

    # ── 循环/序列（检测模型通常不用）────────────────────────────────────────────
    "LSTM": {
        "status": "BPU_partial",
        "note": "支持标准 LSTM，不支持双向",
        "workaround": None,
    },
    "GRU": {
        "status": "BPU_partial",
        "note": "支持标准 GRU",
        "workaround": None,
    },

    # ── 逻辑 ─────────────────────────────────────────────────────────────────
    "And": {"status": "BPU", "note": "完全支持", "workaround": None},
    "Or":  {"status": "BPU", "note": "完全支持", "workaround": None},
    "Not": {"status": "BPU", "note": "完全支持", "workaround": None},
    "Equal": {"status": "BPU", "note": "完全支持", "workaround": None},
    "Less": {"status": "BPU", "note": "完全支持", "workaround": None},
    "Greater": {"status": "BPU", "note": "完全支持", "workaround": None},
    "LessOrEqual": {"status": "BPU", "note": "完全支持", "workaround": None},
    "GreaterOrEqual": {"status": "BPU", "note": "完全支持", "workaround": None},

    # ── 其他 ─────────────────────────────────────────────────────────────────
    "Identity": {
        "status": "BPU",
        "note": "恒等映射，通常被常量折叠消除",
        "workaround": None,
    },
    "Dropout": {
        "status": "BPU",
        "note": "推理阶段自动转为 Identity，完全支持",
        "workaround": None,
    },
    "TopK": {
        "status": "CPU",
        "note": "通常用于分类 Top-K，检测模型中很少出现",
        "workaround": "移至后处理 Python 代码",
    },
    "ArgMax": {
        "status": "BPU_partial",
        "note": "有限支持，建议验证",
        "workaround": None,
    },
    "ArgMin": {
        "status": "BPU_partial",
        "note": "有限支持，建议验证",
        "workaround": None,
    },
    "Einsum": {
        "status": "CPU",
        "note": "复杂张量运算，通常落 CPU",
        "workaround": "展开为等价的 MatMul + Reshape",
    },
    "Erf": {
        "status": "BPU",
        "note": "bayes-e 支持（用于 GELU）",
        "workaround": None,
    },
}

STATUS_EMOJI = {
    "BPU":         "✅ BPU",
    "BPU_int16":   "⚠️  BPU(int16需配置)",
    "BPU_partial": "🟡 BPU(有限制)",
    "CPU":         "🔴 CPU(退化)",
    "unsupported": "❌ 不支持",
    "unknown":     "❓ 未知",
}


def main():
    log.info("=" * 60)
    log.info("STEP 1-4: RDK X5 (bayes-e) 算子兼容性分析")
    log.info("=" * 60)

    # BP-A: 加载 operators.json
    log.info(f"[BP-A] 加载: {OPERATORS_JSON}")
    if not OPERATORS_JSON.exists():
        log.error("[BP-A] 找不到 operators.json，请先运行 03_extract_operators.py")
        sys.exit(1)

    with OPERATORS_JSON.open(encoding="utf-8") as f:
        ops_data = json.load(f)

    unique_ops: list[str] = ops_data["unique_ops"]
    log.info(f"[BP-A] 待分析算子: {len(unique_ops)} 个")

    # BP-B: 逐一比对兼容性
    log.info("[BP-B] 逐一对照兼容性数据库 ...")
    results: dict[str, dict] = {}
    for op in unique_ops:
        if op in RDKX5_OP_DB:
            info = RDKX5_OP_DB[op].copy()
        else:
            info = {
                "status": "unknown",
                "note": f"未在兼容性数据库中找到 {op}，建议用 hb_mapper checker 验证",
                "workaround": "手动用 hb_mapper checker 确认是否落 BPU",
            }
        info["count"] = ops_data["operators"][op]["count"]
        results[op] = info

    # BP-C: 输出分类结果
    log.info("[BP-C] 兼容性分析结果:")
    log.info("-" * 60)

    by_status: dict[str, list] = {}
    for op, info in results.items():
        s = info["status"]
        by_status.setdefault(s, []).append(op)

    for status in ["BPU", "BPU_int16", "BPU_partial", "CPU", "unsupported", "unknown"]:
        ops_in = by_status.get(status, [])
        if not ops_in:
            continue
        emoji = STATUS_EMOJI.get(status, status)
        log.info(f"\n  {emoji} ({len(ops_in)} 个算子):")
        for op in ops_in:
            info = results[op]
            log.info(f"    [{info['count']:3d}x] {op}")
            log.info(f"          → {info['note']}")
            if info.get("workaround"):
                log.info(f"          替换方案: {info['workaround']}")

    # 汇总统计
    bpu_count  = len(by_status.get("BPU", []))
    warn_count = len(by_status.get("BPU_int16", [])) + len(by_status.get("BPU_partial", []))
    cpu_count  = len(by_status.get("CPU", []))
    bad_count  = len(by_status.get("unsupported", []))
    unk_count  = len(by_status.get("unknown", []))

    log.info("\n" + "-" * 60)
    log.info(f"  汇总: BPU={bpu_count} | 需配置={warn_count} | CPU退化={cpu_count} "
             f"| 不支持={bad_count} | 未知={unk_count}")
    log.info("-" * 60)

    if bad_count > 0:
        log.warning(f"[BP-C] 存在 {bad_count} 个不支持算子，需要替换!")
    if warn_count > 0:
        log.warning(f"[BP-C] 存在 {warn_count} 个算子需特殊配置 (Softmax int16 等)")
    if cpu_count > 0:
        log.info(f"[BP-C] {cpu_count} 个算子将退化到 CPU 运行（不影响正确性，影响性能）")

    # BP-D: 保存完整分析结果
    output = {
        "model_path": ops_data["model_path"],
        "rdkx5_arch": "bayes-e",
        "oe_version": "v1.2.8",
        "summary": {
            "total_unique_ops": len(unique_ops),
            "BPU_native":  bpu_count,
            "BPU_special_config": warn_count,
            "CPU_fallback": cpu_count,
            "unsupported": bad_count,
            "unknown": unk_count,
        },
        "compatibility": results,
        "critical_actions": [],
    }

    # 生成关键行动清单
    for op, info in results.items():
        if info["status"] in ("BPU_int16",):
            nodes = ops_data["operators"][op]["nodes"]
            for n in nodes:
                output["critical_actions"].append({
                    "priority": "HIGH",
                    "op": op,
                    "node_name": n["name"],
                    "action": "在 hb_mapper YAML 的 node_info 中配置 BPU + int16",
                    "yaml_snippet": (
                        f'node_info:\n'
                        f'  "{n["name"]}":\n'
                        f'    ON: BPU\n'
                        f'    InputType: int16\n'
                        f'    OutputType: int16'
                    ),
                })
        elif info["status"] == "unsupported":
            output["critical_actions"].append({
                "priority": "CRITICAL",
                "op": op,
                "node_name": None,
                "action": f"替换算子: {info.get('workaround', '需人工处理')}",
            })

    COMPATIBILITY_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    log.info(f"[BP-D] 兼容性报告已保存: {COMPATIBILITY_JSON} ✓")

    if output["critical_actions"]:
        log.info(f"[BP-D] 关键行动清单: {len(output['critical_actions'])} 条")
        for action in output["critical_actions"]:
            log.info(f"       [{action['priority']}] {action['op']}: {action['action']}")

    log.info("=" * 60)
    log.info("STEP 1-4 完成 → Step 1 全部完成！")
    log.info("下一步: 运行 step/step2/ 进行 ONNX 推理验证")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
