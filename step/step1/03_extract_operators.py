#!/usr/bin/env python3
"""
Step 1-3: 提取 ONNX 图中所有算子
目的: 遍历 ONNX graph，输出 operators.json，供兼容性分析使用

输出文件:
  operators.json — 包含所有唯一算子名 + 每个算子的节点列表及输入输出 shape

断点说明:
  BP-A: 加载 ONNX 模型
  BP-B: 遍历所有节点，收集算子信息
  BP-C: 找到 Softmax 节点（RDK X5 关键算子）
  BP-D: 保存 JSON

运行方式:
  C:\d\Anaconda3\envs\yolo\python.exe step/step1/03_extract_operators.py
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

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
log = logging.getLogger("step1_03")

# ── 路径 ─────────────────────────────────────────────────────────────────────
STEP1_DIR    = Path(__file__).parent
ONNX_PATH    = STEP1_DIR / "best.onnx"
OPERATORS_JSON = STEP1_DIR / "operators.json"


def get_shape_info(model):
    """从 value_info + graph.output + graph.input 建立 tensor → shape 映射"""
    shape_map: dict[str, list] = {}

    def extract(type_proto):
        if type_proto.HasField("tensor_type"):
            dims = type_proto.tensor_type.shape.dim
            return [d.dim_value if d.dim_value > 0 else "?" for d in dims]
        return []

    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        shape_map[vi.name] = extract(vi.type)
    return shape_map


def main():
    log.info("=" * 60)
    log.info("STEP 1-3: 提取 ONNX 算子列表")
    log.info("=" * 60)

    # BP-A: 加载 ONNX
    log.info(f"[BP-A] 加载 ONNX: {ONNX_PATH}")
    if not ONNX_PATH.exists():
        log.error(f"[BP-A] 找不到 ONNX 文件，请先运行 02_export_onnx.py")
        sys.exit(1)

    import onnx
    model = onnx.load(str(ONNX_PATH))
    # 推断 shape (需要完整 graph 信息)
    try:
        model = onnx.shape_inference.infer_shapes(model)
        log.info("[BP-A] shape inference 完成 ✓")
    except Exception as e:
        log.warning(f"[BP-A] shape inference 警告: {e}")
    shape_map = get_shape_info(model)

    total_nodes = len(model.graph.node)
    log.info(f"[BP-A] 图中总节点数: {total_nodes}")

    # BP-B: 遍历所有节点
    log.info("[BP-B] 遍历节点，收集算子信息 ...")

    # 按 op_type 分组
    op_groups: dict[str, list] = defaultdict(list)
    for node in model.graph.node:
        entry = {
            "name":    node.name or "(unnamed)",
            "domain":  node.domain or "ai.onnx",
            "inputs":  [
                {"tensor": t, "shape": shape_map.get(t, [])}
                for t in node.input if t
            ],
            "outputs": [
                {"tensor": t, "shape": shape_map.get(t, [])}
                for t in node.output if t
            ],
        }
        op_groups[node.op_type].append(entry)

    unique_ops = sorted(op_groups.keys())
    log.info(f"[BP-B] 唯一算子数量: {len(unique_ops)}")
    log.info(f"[BP-B] 算子列表: {unique_ops}")

    for op in unique_ops:
        log.info(f"       {op}: {len(op_groups[op])} 个节点")

    # BP-C: 查找 Softmax 节点（RDK X5 必须特殊处理）
    softmax_nodes = op_groups.get("Softmax", [])
    if softmax_nodes:
        log.info(f"[BP-C] 发现 Softmax 节点 ({len(softmax_nodes)} 个) ⚠️")
        for n in softmax_nodes:
            log.info(f"       节点名: '{n['name']}' → 需在 hb_mapper YAML 中配置 node_info (int16)")
    else:
        log.info("[BP-C] 未发现 Softmax 节点 (无需特殊配置)")

    # 整理输出数据结构
    result = {
        "model_path": str(ONNX_PATH),
        "total_nodes": total_nodes,
        "unique_op_count": len(unique_ops),
        "unique_ops": unique_ops,
        "operators": {
            op: {
                "count": len(nodes),
                "nodes": nodes,
            }
            for op, nodes in op_groups.items()
        },
    }

    # BP-D: 保存 JSON
    OPERATORS_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    log.info(f"[BP-D] 算子信息已保存: {OPERATORS_JSON} ✓")
    log.info("=" * 60)
    log.info("STEP 1-3 完成 → 继续运行 04_check_compatibility.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
