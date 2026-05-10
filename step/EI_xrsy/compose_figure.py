#!/usr/bin/env python3
"""
PC 端：将板端推理结果拼成论文对比图
布局（与 make_comparison_figure.py 风格一致）：
  列：9 张测试图（small×3 | middle×3 | large×3）
  行：Original / Config A / Config E（或按 SHOW_CONFIGS 自定义）

用法：
  python compose_figure.py                    # 默认：A vs E
  python compose_figure.py --configs A B C E  # 自定义行
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE        = Path(__file__).parent
TEST_DIR    = BASE / "test_images"
RESULTS_DIR = BASE / "results"
OUT_PATH    = BASE / "figures" / "ablation_detection_figure.png"

# ── 布局参数 ──────────────────────────────────────────────────
CELL_W    = 260
CELL_H    = 260      # 测试图已是正方形 640×640
GAP_CELL  = 4
GAP_GROUP = 18       # small / middle / large 之间的间距
GAP_ROW   = 6
CAT_H     = 32
LABEL_W   = 110
BG        = (255, 255, 255)
LINE_CLR  = (200, 200, 200)
LABEL_FG  = (20, 20, 20)

# 每行对应的边框颜色
BORDER_CLR = {
    "Original": (150, 150, 150),
    "A":        (220, 60,  60),
    "B":        (220, 140, 30),
    "C":        (180, 100, 200),
    "D":        (30,  160, 200),
    "E":        (30,  100, 220),
}
BORDER_W = {
    "Original": 1,
    "A": 2, "B": 2, "C": 2, "D": 2, "E": 2,
}
ROW_LABEL = {
    "Original": "Original",
    "A": "Config A\n(Baseline)",
    "B": "Config B\n(+INT16)",
    "C": "Config C\n(+Shape)",
    "D": "Config D\n(+Calib.)",
    "E": "Config E\n(Ours)",
}

CATEGORIES = ["small", "middle", "large"]
CAT_DISPLAY = {"small": "Small Fire", "middle": "Middle Fire", "large": "Large Fire"}
N_PER_CAT   = 3
DPI         = 300


# ── 辅助 ──────────────────────────────────────────────────────
def get_font(size):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()

def _text_size(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2]-bb[0], bb[3]-bb[1]

def draw_centered(draw, text, x, y, w, h, font, color):
    tw, th = _text_size(draw, text, font)
    draw.text((x+(w-tw)//2, y+(h-th)//2), text, fill=color, font=font)

def make_rotated_label(text, cell_h, label_w, font):
    lines = text.split("\n")
    tmp = Image.new("RGB", (1,1), BG)
    d   = ImageDraw.Draw(tmp)
    lsp = 4
    sizes = [_text_size(d, l, font) for l in lines]
    th_total = sum(s[1] for s in sizes) + lsp*(len(lines)-1)
    tw_max   = max(s[0] for s in sizes)
    tmp2 = Image.new("RGB", (cell_h, label_w), BG)
    d2   = ImageDraw.Draw(tmp2)
    ty   = (label_w - th_total) // 2
    tx0  = (cell_h  - tw_max)   // 2
    for line, (lw, lh) in zip(lines, sizes):
        d2.text((tx0 + (tw_max-lw)//2, ty), line, fill=LABEL_FG, font=font)
        ty += lh + lsp
    return tmp2.rotate(90, expand=True)

def crop_square(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    iw, ih = img.size
    s = min(iw, ih)
    img = img.crop(((iw-s)//2, (ih-s)//2, (iw+s)//2, (ih+s)//2))
    return img.resize((size, size), Image.LANCZOS)


# ── 查找图片 ──────────────────────────────────────────────────
def collect_images(show_configs):
    """返回 ordered_paths[row_key][cat][idx] = Path or None"""
    data = {}

    # Original
    data["Original"] = {}
    for cat in CATEGORIES:
        paths = sorted(TEST_DIR.glob(f"{cat}_*.jpg"))[:N_PER_CAT]
        data["Original"][cat] = paths

    # Board results
    for cfg in show_configs:
        cfg_dir = RESULTS_DIR / f"config_{cfg}"
        data[cfg] = {}
        for cat in CATEGORIES:
            paths = sorted(cfg_dir.glob(f"{cat}_*.jpg"))[:N_PER_CAT] if cfg_dir.exists() else []
            data[cfg][cat] = paths
            if not paths:
                print(f"  [WARN] results/config_{cfg}/{cat}_*.jpg 未找到，该格显示空白")
    return data


# ── 合成 ──────────────────────────────────────────────────────
def compose(show_configs):
    row_keys = ["Original"] + show_configs

    grp_w = N_PER_CAT * CELL_W + (N_PER_CAT-1) * GAP_CELL
    def grp_x(gi): return LABEL_W + gi*(grp_w + GAP_GROUP)

    total_w = grp_x(len(CATEGORIES)-1) + grp_w
    total_h = CAT_H + len(row_keys)*(CELL_H + GAP_ROW) - GAP_ROW

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    font_cat   = get_font(16)
    font_label = get_font(13)

    # ── 顶部类别标签 ──
    for gi, cat in enumerate(CATEGORIES):
        x0 = grp_x(gi)
        draw_centered(draw, CAT_DISPLAY[cat], x0, 0, grp_w, CAT_H, font_cat, (30,30,30))
        draw.line([(x0, CAT_H-1), (x0+grp_w-1, CAT_H-1)], fill=(100,100,100), width=1)
    for gi in range(1, len(CATEGORIES)):
        sx = grp_x(gi) - GAP_GROUP//2
        draw.line([(sx, 0), (sx, total_h)], fill=LINE_CLR, width=1)

    data = collect_images(show_configs)

    # ── 各行图像 ──
    for ri, rk in enumerate(row_keys):
        y_row = CAT_H + ri*(CELL_H + GAP_ROW)
        bdr   = BORDER_CLR.get(rk, (120,120,120))
        bdr_w = BORDER_W.get(rk, 1)

        lbl = make_rotated_label(ROW_LABEL.get(rk, rk), CELL_H, LABEL_W, font_label)
        canvas.paste(lbl, ((LABEL_W-lbl.width)//2, y_row+(CELL_H-lbl.height)//2))
        draw.line([(LABEL_W-1, y_row), (LABEL_W-1, y_row+CELL_H-1)], fill=LINE_CLR)

        for gi, cat in enumerate(CATEGORIES):
            x0    = grp_x(gi)
            paths = data.get(rk, {}).get(cat, [])
            for ci in range(N_PER_CAT):
                x_cell = x0 + ci*(CELL_W + GAP_CELL)
                if ci < len(paths) and paths[ci].exists():
                    img  = Image.open(paths[ci]).convert("RGB")
                    cell = crop_square(img, CELL_W)
                else:
                    cell = Image.new("RGB", (CELL_W, CELL_H), (230,230,230))
                    d2   = ImageDraw.Draw(cell)
                    d2.text((10, CELL_H//2-8), "No result", fill=(150,150,150))

                canvas.paste(cell, (x_cell, y_row))
                draw.rectangle([x_cell, y_row,
                                 x_cell+CELL_W-1, y_row+CELL_H-1],
                                outline=bdr, width=bdr_w)

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["A", "E"],
                    choices=["A","B","C","D","E"],
                    help="要展示的配置行（默认 A E）")
    args = ap.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"合成行：Original + Config {' '.join(args.configs)}")
    fig = compose(args.configs)
    fig.save(str(OUT_PATH), dpi=(DPI, DPI))
    print(f"[OK] → {OUT_PATH}  ({fig.width}×{fig.height} px @ {DPI} dpi)")


if __name__ == "__main__":
    main()
