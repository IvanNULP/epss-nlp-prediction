#!/usr/bin/env python3
"""
Generate fig06_architecture.svg

"""

from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ── Coordinate constants ───────────────────────────────────────
W, H = 1420, 660

# Column centers (label area = 90px, three equal cols of 443px)
CX = {1: 311, 2: 754, 3: 1197}
COL_LEFT  = {1: 90,  2: 533, 3: 976}
COL_W     = 443

BOX_W  = 370   # individual box width
BOX_HW = 185   # half width

# Vertical row centers
Y_HDR  = 40    # header banner center
Y_IN   = 112   # shared input center
Y_ENC  = 225   # encoding row center
Y_MOD  = 345   # model row center
Y_OUT  = 460   # output row center
Y_SHAP = 568   # SHAP row center

# Row vertical extents
Y_IN_T,  Y_IN_B  = 80,  144
Y_ENC_T, Y_ENC_B = 183, 267
Y_MOD_T, Y_MOD_B = 305, 385
Y_OUT_T, Y_OUT_B = 430, 490
Y_SHAP_T,Y_SHAP_B= 530, 606

# Column colors
COL_COLOR = {
    1: {"banner": "#4472C4", "enc": "#DDEEFF", "mod": "#C5DCF5", "out": "#E8EDF8"},
    2: {"banner": "#D46C1A", "enc": "#FFF3E0", "mod": "#FFE0B2", "out": "#FFF5E8"},
    3: {"banner": "#538135", "enc": "#E8F5E9", "mod": "#C8E6C9", "out": "#E0F0E0"},
}
C_INPUT = "#EAF4FF"
C_SHAP  = "#FCE4EC"
C_EDGE  = "#555555"
C_GREY  = "#888888"
C_WHITE = "#FFFFFF"
C_LABEL = "#666666"

lines = []

def ln(*args):
    lines.extend(args)

def arrow_down(cx, y1, y2):
    """Vertical down arrow from y1 to y2."""
    tip = y2
    shaft_end = tip - 11
    ln(
        f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{shaft_end}" '
        f'stroke="{C_EDGE}" stroke-width="1.2"/>',
        f'<polygon points="{cx-7},{shaft_end} {cx+7},{shaft_end} {cx},{tip}" fill="{C_EDGE}"/>'
    )

def parallelogram(x1, y1, x2, y2, skew, fill, stroke, gid=""):
    """Parallelogram: top-left skewed right, bottom-left normal."""
    pts = f"{x1+skew},{y1} {x2},{y1} {x2-skew},{y2} {x1},{y2}"
    g = f' id="{gid}"' if gid else ""
    ln(f'<g{g}>')
    ln(f'  <polygon points="{pts}" fill="{fill}" '
       f'stroke="{stroke}" stroke-width="1.2"/>')
    ln(f'</g>')

def hexagon(cx, cy, w, h, fill, stroke, gid=""):
    """Flat-sided hexagon."""
    hw, hh = w/2, h/2
    qw = w/4
    pts = (f"{cx-hw:.0f},{cy:.0f} "
           f"{cx-qw:.0f},{cy-hh:.0f} "
           f"{cx+qw:.0f},{cy-hh:.0f} "
           f"{cx+hw:.0f},{cy:.0f} "
           f"{cx+qw:.0f},{cy+hh:.0f} "
           f"{cx-qw:.0f},{cy+hh:.0f}")
    g = f' id="{gid}"' if gid else ""
    ln(f'<g{g}>')
    ln(f'  <polygon points="{pts}" fill="{fill}" '
       f'stroke="{stroke}" stroke-width="1.2"/>')
    ln(f'</g>')

def rect_box(cx, cy, w, h, rx, fill, stroke, gid=""):
    """Rounded rectangle centered at cx,cy."""
    x, y = cx - w/2, cy - h/2
    g = f' id="{gid}"' if gid else ""
    ln(f'<g{g}>')
    ln(f'  <rect x="{x:.0f}" y="{y:.0f}" width="{w}" height="{h}" '
       f'rx="{rx}" ry="{rx}" fill="{fill}" '
       f'stroke="{stroke}" stroke-width="1.2"/>')
    ln(f'</g>')

def pill(cx, cy, w, h, fill, stroke, gid=""):
    """Stadium/pill shape."""
    rect_box(cx, cy, w, h, h//2, fill, stroke, gid)

def ellipse_box(cx, cy, rx, ry, fill, stroke, gid=""):
    g = f' id="{gid}"' if gid else ""
    ln(f'<ellipse{g} cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" '
       f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')

def txt(x, y, text, size=11, color="#222222", italic=False,
        anchor="middle", weight="normal"):
    style = f"font-style:italic;" if italic else ""
    ln(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
       f'font-family="Arial" font-size="{size}" '
       f'font-weight="{weight}" '
       f'style="{style}" fill="{color}">{text}</text>')

def txt2(x, y1, line1, y2, line2, size1=11, size2=9.5,
         color1="#222222", color2="#555555", anchor="middle"):
    txt(x, y1, line1, size1, color1, anchor=anchor)
    txt(x, y2, line2, size2, color2, italic=True, anchor=anchor)

# ═══════════════════════════════════════════════════════════════
# SVG OPEN
# ═══════════════════════════════════════════════════════════════
ln(f'<?xml version="1.0" encoding="UTF-8"?>')
ln(f'<svg xmlns="http://www.w3.org/2000/svg" '
   f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
   f'font-family="Arial" font-weight="normal">')
ln(f'  <rect width="{W}" height="{H}" fill="#ffffff"/>')

# ── Column separator dashed lines ──────────────────────────────
for x in [530, 973]:
    ln(f'<line x1="{x}" y1="15" x2="{x}" y2="{Y_SHAP_B+15}" '
       f'stroke="#DDDDDD" stroke-width="0.8" stroke-dasharray="5,4"/>')

# ── Row labels (left margin) ───────────────────────────────────
for (y, label) in [
    (Y_IN,   "Input"),
    (Y_ENC,  "Encoding"),
    (Y_MOD,  "Model"),
    (Y_OUT,  "Output"),
    (Y_SHAP, "Explainability"),
]:
    txt(78, y+4, label, size=9.5, color=C_LABEL,
        italic=True, anchor="end")

# ── HEADER BANNERS ─────────────────────────────────────────────
headers = {
    1: ("M1: TF-IDF Baseline",      "Ridge / Logistic Regression"),
    2: ("M2: Sentence-BERT",         "Sentence Transformer + XGBoost"),
    3: ("M3: SecBERT Fine-tuned",    "Transformer + Linear Head"),
}
for col, (h1, h2) in headers.items():
    x = COL_LEFT[col]
    ln(f'<rect id="banner-m{col}" x="{x}" y="15" '
       f'width="{COL_W}" height="52" rx="6" '
       f'fill="{COL_COLOR[col]["banner"]}" stroke="none"/>')
    txt(CX[col], 38, h1,  size=12, color=C_WHITE)
    txt(CX[col], 56, h2,  size=9,  color=C_WHITE, italic=True)

# ── SHARED INPUT (parallelogram) ───────────────────────────────
ln(f'<g id="box-input">')
ln(f'  <polygon points="112,{Y_IN_T} 1410,{Y_IN_T} '
   f'1388,{Y_IN_B} 90,{Y_IN_B}" '
   f'fill="{C_INPUT}" stroke="{C_EDGE}" stroke-width="1.2"/>')
ln(f'</g>')
txt(750, Y_IN+5, "CVE Description Text  |  preprocessed per method",
    size=11, color="#222222")

# ── ARROWS: Input → Encoding ───────────────────────────────────
for col in [1,2,3]:
    arrow_down(CX[col], Y_IN_B, Y_ENC_T)

# ── ENCODING BOXES (hexagons) ──────────────────────────────────
enc_text = {
    1: ("TF-IDF Vectorizer",       "unigrams + bigrams  |  50k features"),
    2: ("Sentence Transformer",    "all-MiniLM-L6-v2  |  384-dim"),
    3: ("SecBERT Encoder",         "[CLS] token representation"),
}
for col in [1,2,3]:
    hexagon(CX[col], Y_ENC, 370, 78,
            COL_COLOR[col]["enc"], C_EDGE, f"box-enc-m{col}")
    t1, t2 = enc_text[col]
    txt2(CX[col], Y_ENC-8, t1, Y_ENC+14, t2)

# ── ARROWS: Encoding → Model ───────────────────────────────────
for col in [1,2,3]:
    arrow_down(CX[col], Y_ENC_B, Y_MOD_T)

# ── MODEL BOXES (rounded rectangles) ──────────────────────────
mod_text = {
    1: ("Ridge Regression  /  Logistic Reg.",  "L2 regularization  |  class weighting"),
    2: ("XGBoost",                             "grid search: n_est, depth, lr"),
    3: ("Linear Head",                         "regression or classification head"),
}
for col in [1,2,3]:
    rect_box(CX[col], Y_MOD, 370, 76, 10,
             COL_COLOR[col]["mod"], C_EDGE, f"box-mod-m{col}")
    t1, t2 = mod_text[col]
    txt2(CX[col], Y_MOD-8, t1, Y_MOD+14, t2)

# ── ARROWS: Model → Output ────────────────────────────────────
for col in [1,2,3]:
    arrow_down(CX[col], Y_MOD_B, Y_OUT_T)

# ── OUTPUT BOXES (pill/stadium) ────────────────────────────────
out_text = {
    1: "Regression: MAE / RMSE / R2   |   Classification: AUC-ROC / PR-AUC",
    2: "Regression: MAE / RMSE / R2   |   Classification: AUC-ROC / PR-AUC",
    3: "Regression: MAE / RMSE / R2   |   Classification: AUC-ROC / PR-AUC",
}
for col in [1,2,3]:
    pill(CX[col], Y_OUT, 400, 52,
         COL_COLOR[col]["out"], C_EDGE, f"box-out-m{col}")
    txt(CX[col], Y_OUT+4, out_text[col], size=9, color="#333333")

# ── ARROWS: Output → SHAP ─────────────────────────────────────
arrow_down(CX[1], Y_OUT_B, Y_SHAP_T)
arrow_down(CX[2], Y_OUT_B, Y_SHAP_T)
# M3: no arrow (not applied)

# ── SHAP BOXES (ellipses for M1, M2; text for M3) ─────────────
shap_text = {
    1: ("SHAP LinearExplainer",   "token-level feature importance"),
    2: ("SHAP TreeExplainer",     "embedding-level importance"),
}
for col in [1, 2]:
    ellipse_box(CX[col], Y_SHAP, 168, 42,
                C_SHAP, C_EDGE, f"box-shap-m{col}")
    t1, t2 = shap_text[col]
    txt2(CX[col], Y_SHAP-8, t1, Y_SHAP+14, t2,
         size1=10.5, size2=9)

txt(CX[3], Y_SHAP-6,  "SHAP: not applied",
    size=9.5, color=C_GREY, italic=True)
txt(CX[3], Y_SHAP+12, "(embedding space, not token-level)",
    size=8.5, color=C_GREY, italic=True)

# ── LEGEND ────────────────────────────────────────────────────
LY = Y_SHAP_B + 20
ln(f'<line x1="90" y1="{LY}" x2="1410" y2="{LY}" '
   f'stroke="#EEEEEE" stroke-width="1"/>')

LY2 = LY + 24
legend_items = [
    # (x_center, shape_type, label)
    (160,  "para",    "Input data"),
    (370,  "hex",     "Encoding step"),
    (580,  "rect",    "Model / Predictor"),
    (790,  "pill",    "Task output"),
    (1010, "ellipse", "SHAP explainability"),
    (1220, "none",    "Not applicable"),
]
SH, SW = 28, 52   # shape height, width in legend

for (lx, shape, label) in legend_items:
    sx = lx - 35
    sy = LY2
    if shape == "para":
        pts = (f"{sx+4},{sy-SH//2} {sx+SW},{sy-SH//2} "
               f"{sx+SW-4},{sy+SH//2} {sx},{sy+SH//2}")
        ln(f'<polygon points="{pts}" fill="{C_INPUT}" '
           f'stroke="{C_EDGE}" stroke-width="0.8"/>')
    elif shape == "hex":
        hw, hh, qw = SW//2, SH//2, SW//4
        cx2, cy2 = sx+SW//2, sy
        pts = (f"{cx2-hw},{cy2} {cx2-qw},{cy2-hh} "
               f"{cx2+qw},{cy2-hh} {cx2+hw},{cy2} "
               f"{cx2+qw},{cy2+hh} {cx2-qw},{cy2+hh}")
        ln(f'<polygon points="{pts}" fill="#DDEEFF" '
           f'stroke="{C_EDGE}" stroke-width="0.8"/>')
    elif shape == "rect":
        ln(f'<rect x="{sx}" y="{sy-SH//2}" width="{SW}" height="{SH}" '
           f'rx="4" fill="#E8F5E9" stroke="{C_EDGE}" stroke-width="0.8"/>')
    elif shape == "pill":
        rx2 = SH//2
        ln(f'<rect x="{sx}" y="{sy-SH//2}" width="{SW}" height="{SH}" '
           f'rx="{rx2}" fill="#E8EDF8" stroke="{C_EDGE}" stroke-width="0.8"/>')
    elif shape == "ellipse":
        ln(f'<ellipse cx="{sx+SW//2}" cy="{sy}" '
           f'rx="{SW//2}" ry="{SH//2}" '
           f'fill="{C_SHAP}" stroke="{C_EDGE}" stroke-width="0.8"/>')
    elif shape == "none":
        txt(sx + SW//2, sy+4, "N/A", size=9, color=C_GREY, italic=True)

    txt(lx + 24, LY2 + 5, label, size=9, color="#444444", anchor="start")

ln('</svg>')

# ── Write file ────────────────────────────────────────────────
svg_path = DATA_DIR / "fig06_architecture_v2.svg"
with open(svg_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Saved: {svg_path}")
