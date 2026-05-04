# =============================================================
# Figure 6 — Architecture diagram of three methods M1 / M2 / M3

# =============================================================

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

matplotlib.rcParams.update({
    "font.family":      "Liberation Sans",
    "font.size":        9,
    "font.weight":      "normal",
    "axes.titleweight": "normal",
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.1,
})

# Colors
C_INPUT  = "#DDEEFF"
C_ENC    = "#FFF3E0"
C_MODEL  = "#E8F5E9"
C_OUT    = "#F3E5F5"
C_SHAP   = "#FCE4EC"
C_EDGE   = "#555555"
C_GREY   = "#888888"

fig, ax = plt.subplots(figsize=(15, 7.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 7.5)
ax.axis("off")

def box(cx, cy, w, h, line1, line2=None, color="#EEE", fs=9):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.07",
        linewidth=0.8, edgecolor=C_EDGE, facecolor=color, zorder=3)
    ax.add_patch(rect)
    if line2:
        ax.text(cx, cy + 0.18, line1, ha="center", va="center",
                fontsize=fs, color="#222222", zorder=4)
        ax.text(cx, cy - 0.18, line2, ha="center", va="center",
                fontsize=fs - 1, color="#555555",
                style="italic", zorder=4)
    else:
        ax.text(cx, cy, line1, ha="center", va="center",
                fontsize=fs, color="#222222", zorder=4)

def arr(x1, y1, x2, y2, rad=0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
        arrowprops=dict(arrowstyle="-|>", color=C_EDGE,
                        lw=0.8, mutation_scale=10,
                        connectionstyle=f"arc3,rad={rad}"))

def label(x, y, text, fs=8.5):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color=C_GREY, style="italic")

# ── Column x-positions ────────────────────────────────────────
X1, X2, X3 = 2.5, 7.5, 12.5   # M1, M2, M3
Y_TITLE = 7.1
Y_IN    = 6.2
Y_ENC   = 4.9
Y_MOD   = 3.5
Y_OUT   = 2.1
Y_SHAP  = 0.7

W_MAIN = 3.8
H_STD  = 0.85

# ── Column titles ─────────────────────────────────────────────
for x, title, sub in [
    (X1, "M1: TF-IDF Baseline", "Ridge / Logistic Regression"),
    (X2, "M2: Sentence-BERT", "XGBoost"),
    (X3, "M3: SecBERT Fine-tuned", "Transformer + head"),
]:
    ax.text(x, Y_TITLE, title, ha="center", va="center",
            fontsize=10, color="#222222", zorder=4)
    ax.text(x, Y_TITLE - 0.28, sub, ha="center", va="center",
            fontsize=8.5, color=C_GREY, style="italic", zorder=4)

# ── Row: Input (shared) ───────────────────────────────────────
box(7.5, Y_IN, 12.0, 0.75,
    "CVE Description Text (preprocessed)",
    None, C_INPUT, fs=9)
label(0.35, Y_IN, "Input")

# Arrows from input to encoders
arr(X1, Y_IN - 0.38, X1, Y_ENC + 0.43)
arr(X2, Y_IN - 0.38, X2, Y_ENC + 0.43)
arr(X3, Y_IN - 0.38, X3, Y_ENC + 0.43)

# ── Row: Encoding ─────────────────────────────────────────────
box(X1, Y_ENC, W_MAIN, H_STD,
    "TF-IDF Vectorizer",
    "unigrams + bigrams, 50k features", C_ENC)
box(X2, Y_ENC, W_MAIN, H_STD,
    "Sentence Transformer",
    "all-MiniLM-L6-v2  |  384-dim", C_ENC)
box(X3, Y_ENC, W_MAIN, H_STD,
    "SecBERT Encoder",
    "[CLS] token representation", C_ENC)
label(0.35, Y_ENC, "Encoding")

# Arrows encoding → model
arr(X1, Y_ENC - 0.43, X1, Y_MOD + 0.43)
arr(X2, Y_ENC - 0.43, X2, Y_MOD + 0.43)
arr(X3, Y_ENC - 0.43, X3, Y_MOD + 0.43)

# ── Row: Model ────────────────────────────────────────────────
box(X1, Y_MOD, W_MAIN, H_STD,
    "Ridge Regression",
    "L2 regularization, class weights", C_MODEL)
box(X2, Y_MOD, W_MAIN, H_STD,
    "XGBoost",
    "grid search: depth, lr, n_est", C_MODEL)
box(X3, Y_MOD, W_MAIN, H_STD,
    "Linear Head",
    "regression / classification head", C_MODEL)
label(0.35, Y_MOD, "Model")

# Arrows model → output
arr(X1, Y_MOD - 0.43, X1, Y_OUT + 0.43)
arr(X2, Y_MOD - 0.43, X2, Y_OUT + 0.43)
arr(X3, Y_MOD - 0.43, X3, Y_OUT + 0.43)

# ── Row: Output (shared label, two tasks) ─────────────────────
box(4.5, Y_OUT, 5.5, 0.75,
    "Regression: EPSS score [0,1]  (MAE, RMSE, R2)",
    None, C_OUT, fs=8.5)
box(10.5, Y_OUT, 5.5, 0.75,
    "Classification: high-risk (EPSS >= 0.1)  (AUC-ROC, PR-AUC)",
    None, C_OUT, fs=8.5)
label(0.35, Y_OUT, "Output")

# Arrows output → SHAP (only M1 and M2, labeled)
arr(X1, Y_OUT - 0.38, X1, Y_SHAP + 0.38)
arr(X2, Y_OUT - 0.38, X2, Y_SHAP + 0.38, rad=0.1)

# ── Row: SHAP ─────────────────────────────────────────────────
box(X1, Y_SHAP, W_MAIN, 0.65,
    "LinearExplainer (SHAP)",
    "token-level feature importance", C_SHAP)
box(X2, Y_SHAP, W_MAIN, 0.65,
    "TreeExplainer (SHAP)",
    "embedding-level importance", C_SHAP)

# M3: no SHAP note
ax.text(X3, Y_SHAP, "SHAP: not applied\n(embedding not token-level)",
        ha="center", va="center",
        fontsize=8, color=C_GREY, style="italic", zorder=4)

label(0.35, Y_SHAP, "Explainability")

# ── Separator lines between columns ───────────────────────────
for x in [5.0, 10.0]:
    ax.plot([x, x], [0.2, 6.7], color="#DDDDDD",
            linewidth=0.6, linestyle="--", zorder=1)

plt.tight_layout()
plt.savefig(DATA_DIR / "fig06_architecture.pdf")
plt.savefig(DATA_DIR / "fig06_architecture.png", dpi=300)
plt.show()
print("Figure 6 saved.")
