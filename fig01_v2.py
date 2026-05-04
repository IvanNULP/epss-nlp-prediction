"""
Figure 1 — Conceptual diagram of the study pipeline.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

matplotlib.rcParams.update({
    "font.family":      "Arial",
    "font.size":        10,
    "font.weight":      "normal",
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.15,
})

fig, ax = plt.subplots(figsize=(16, 7.5))
ax.set_xlim(0, 17)
ax.set_ylim(0, 7.5)
ax.axis("off")

C_SOURCE = "#DDEEFF"
C_PIPE   = "#E8F5E9"
C_MODEL  = "#FFF3E0"
C_OUT    = "#F3E5F5"
C_XAI    = "#FCE4EC"
C_EDGE   = "#555555"
C_GREY   = "#777777"

def rbox(ax, cx, cy, w, h, line1, line2=None, color="#EEE", fs=10):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.08",
        linewidth=0.9,
        edgecolor=C_EDGE,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    if line2:
        ax.text(cx, cy + 0.22, line1,
                ha="center", va="center", fontsize=fs,
                color="#222222", zorder=4)
        ax.text(cx, cy - 0.22, line2,
                ha="center", va="center", fontsize=fs - 1.5,
                color="#555555", style="italic", zorder=4)
    else:
        ax.text(cx, cy, line1,
                ha="center", va="center", fontsize=fs,
                color="#222222", zorder=4)

def arr(ax, x1, y1, x2, y2):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        zorder=2,
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_EDGE,
            lw=0.9,
            mutation_scale=12,
        )
    )

def line(ax, pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=C_EDGE, lw=0.9, zorder=2)

def side_label(ax, y, text):
    ax.text(0.15, y, text,
            ha="left", va="center",
            fontsize=8.5, color=C_GREY,
            style="italic", zorder=4)

# ════════════════════════════════════════════
# COLUMN POSITIONS
# ════════════════════════════════════════════
# Main pipeline: centered around x=6.5
# SHAP column: x=13.5

X_NVD  = 2.4
X_EPSS = 6.5
X_KEV  = 10.6
X_MID  = 6.5   # center of main pipeline
X_SHAP = 14.8

BOX_W_SRC  = 3.2
BOX_H_SRC  = 0.85
BOX_W_PIPE = 4.0
BOX_H_PIPE = 0.75
BOX_W_MOD  = 3.0
BOX_H_MOD  = 0.80
BOX_W_OUT  = 3.4
BOX_H_OUT  = 0.70
BOX_W_SHAP = 3.0
BOX_H_SHAP = 0.85

# Y positions (top to bottom)
Y_SRC   = 6.5
Y_MERGE = 5.3
Y_PRE   = 4.1
Y_MOD   = 2.75
Y_OUT   = 1.4
Y_SHAP1 = 4.1
Y_SHAP2 = 2.4

# ════════════════════════════════════════════
# ROW 1 — Data Sources
# ════════════════════════════════════════════
rbox(ax, X_NVD,  Y_SRC, BOX_W_SRC, BOX_H_SRC,
     "NVD API 2.0", "CVE descriptions, CVSS", C_SOURCE)
rbox(ax, X_EPSS, Y_SRC, BOX_W_SRC, BOX_H_SRC,
     "EPSS v4  (FIRST.org)", "Daily exploit probability", C_SOURCE)
rbox(ax, X_KEV,  Y_SRC, BOX_W_SRC, BOX_H_SRC,
     "CISA KEV", "Confirmed exploitations", C_SOURCE)

side_label(ax, Y_SRC, "Data\nsources")

# Arrows: sources down to merge level, then horizontal join
Y_JOIN = Y_MERGE + BOX_H_PIPE/2 + 0.25
line(ax, [(X_NVD,  Y_SRC - BOX_H_SRC/2),
          (X_NVD,  Y_JOIN)])
line(ax, [(X_EPSS, Y_SRC - BOX_H_SRC/2),
          (X_EPSS, Y_JOIN)])
line(ax, [(X_KEV,  Y_SRC - BOX_H_SRC/2),
          (X_KEV,  Y_JOIN)])
line(ax, [(X_NVD, Y_JOIN), (X_KEV, Y_JOIN)])
arr(ax, X_MID, Y_JOIN, X_MID, Y_MERGE + BOX_H_PIPE/2)

# ════════════════════════════════════════════
# ROW 2 — Merge & Filter
# ════════════════════════════════════════════
rbox(ax, X_MID, Y_MERGE, BOX_W_PIPE, BOX_H_PIPE,
     "Merge & Filter", "Join by CVE ID  |  clean, deduplicate", C_PIPE)

side_label(ax, Y_MERGE, "Integration")

arr(ax, X_MID, Y_MERGE - BOX_H_PIPE/2,
        X_MID, Y_PRE   + BOX_H_PIPE/2)

# ════════════════════════════════════════════
# ROW 3 — Preprocessing
# ════════════════════════════════════════════
rbox(ax, X_MID, Y_PRE, BOX_W_PIPE, BOX_H_PIPE,
     "Text Preprocessing", "Normalize  |  lemmatize  |  train/val/test split", C_PIPE)

side_label(ax, Y_PRE, "Preprocessing")

# Fan out to 3 models
Y_FAN = Y_MOD + BOX_H_MOD/2 + 0.3
X_M1, X_M2, X_M3 = 2.4, 6.5, 10.6

line(ax, [(X_MID, Y_PRE - BOX_H_PIPE/2),
          (X_MID, Y_FAN)])
line(ax, [(X_M1, Y_FAN), (X_M3, Y_FAN)])
arr(ax, X_M1, Y_FAN, X_M1, Y_MOD + BOX_H_MOD/2)
arr(ax, X_M2, Y_FAN, X_M2, Y_MOD + BOX_H_MOD/2)
arr(ax, X_M3, Y_FAN, X_M3, Y_MOD + BOX_H_MOD/2)

# ════════════════════════════════════════════
# ROW 4 — Three Models
# ════════════════════════════════════════════
rbox(ax, X_M1, Y_MOD, BOX_W_MOD, BOX_H_MOD,
     "M1: TF-IDF", "Ridge / Logistic Reg.", C_MODEL)
rbox(ax, X_M2, Y_MOD, BOX_W_MOD, BOX_H_MOD,
     "M2: Sentence-BERT", "+ XGBoost", C_MODEL)
rbox(ax, X_M3, Y_MOD, BOX_W_MOD, BOX_H_MOD,
     "M3: SecBERT", "Fine-tuned", C_MODEL)

side_label(ax, Y_MOD, "Encoding\nmethods")

# Fan in to output level
Y_FAN2 = Y_OUT + BOX_H_OUT/2 + 0.3
line(ax, [(X_M1, Y_MOD - BOX_H_MOD/2), (X_M1, Y_FAN2)])
line(ax, [(X_M2, Y_MOD - BOX_H_MOD/2), (X_M2, Y_FAN2)])
line(ax, [(X_M3, Y_MOD - BOX_H_MOD/2), (X_M3, Y_FAN2)])

X_OUT1, X_OUT2 = 3.8, 9.2
line(ax, [(X_M1, Y_FAN2), (X_M3, Y_FAN2)])
arr(ax, X_OUT1, Y_FAN2, X_OUT1, Y_OUT + BOX_H_OUT/2)
arr(ax, X_OUT2, Y_FAN2, X_OUT2, Y_OUT + BOX_H_OUT/2)

# ════════════════════════════════════════════
# ROW 5 — Output tasks
# ════════════════════════════════════════════
rbox(ax, X_OUT1, Y_OUT, BOX_W_OUT + 0.2, BOX_H_OUT,
     "Regression: EPSS score [0,1]", None, C_OUT)
rbox(ax, X_OUT2, Y_OUT, BOX_W_OUT + 0.6, BOX_H_OUT,
     "Classification: high-risk (EPSS >= 0.1)", None, C_OUT, fs=9.5)

side_label(ax, Y_OUT, "Prediction\ntasks")

# ════════════════════════════════════════════
# SHAP COLUMN (right side)
# ════════════════════════════════════════════
rbox(ax, X_SHAP, Y_SHAP1, BOX_W_SHAP, BOX_H_SHAP + 0.2,
     "SHAP Analysis", "Token-level explanations", C_XAI)

rbox(ax, X_SHAP, Y_SHAP2, BOX_W_SHAP, BOX_H_SHAP + 0.2,
     "Linguistic Risk Markers", "Practical SOC insights", C_XAI)

# Arrow: from M3 right edge → SHAP left edge (right-angle routing)
ax.annotate("",
    xy   =(X_SHAP - BOX_W_SHAP/2, Y_SHAP1),
    xytext=(X_M3  + BOX_W_MOD/2,  Y_MOD),
    zorder=2,
    arrowprops=dict(
        arrowstyle="-|>",
        color=C_EDGE,
        lw=0.9,
        mutation_scale=12,
        connectionstyle="angle,angleA=0,angleB=-90,rad=8",
    )
)

arr(ax, X_SHAP, Y_SHAP1 - (BOX_H_SHAP + 0.2)/2,
        X_SHAP, Y_SHAP2 + (BOX_H_SHAP + 0.2)/2)

# ════════════════════════════════════════════
# SAVE — no title, no suptitle
# ════════════════════════════════════════════
plt.tight_layout()
plt.savefig("fig01_conceptual_diagram.pdf")
plt.savefig("fig01_conceptual_diagram.png", dpi=300)
print("Saved: fig01_conceptual_diagram.pdf / .png")
