# =============================================================
# COLAB 03 — Models M1 (TF-IDF) and M2 (Sentence-BERT)
# =============================================================

# ── CELL 1: Drive + Install ───────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# !pip install scikit-learn xgboost shap sentence-transformers
# !pip install pandas pyarrow matplotlib seaborn scipy

from pathlib import Path
DATA_DIR = Path("/content/drive/MyDrive/article_data")

existing = [f.name for f in DATA_DIR.iterdir() if f.is_file()]
print(f"Файлів на Drive: {len(existing)}")
for f in sorted(existing):
    print(f"  {f}")

# ── CELL 2: Imports ───────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)
import shap
import xgboost as xgb

DATA_DIR     = Path("/content/drive/MyDrive/article_data")
RANDOM_STATE = 42

matplotlib.rcParams.update({
    "font.family":       "Liberation Sans",
    "font.size":         9, "font.weight": "normal",
    "axes.titleweight":  "normal", "axes.labelweight": "normal",
    "axes.linewidth":    0.8, "lines.linewidth": 1.0,
    "legend.frameon":    False,
    "axes.spines.top":   False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
C_BLUE="#4472C4"; C_RED="#C00000"; C_ORANGE="#ED7D31"
C_GREEN="#70AD47"; C_GREY="#888888"

# ── CELL 3: Завантаження даних ────────────────────────────────
train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df   = pd.read_parquet(DATA_DIR / "val.parquet")
test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
print(f"High-risk train: {train_df['high_risk'].mean()*100:.1f}%")
print(f"High-risk test:  {test_df['high_risk'].mean()*100:.1f}%")

X_train_full = train_df["text_full"].fillna("").values
X_val_full   = val_df["text_full"].fillna("").values
X_test_full  = test_df["text_full"].fillna("").values
X_train_min  = train_df["text_minimal"].fillna("").values
X_val_min    = val_df["text_minimal"].fillna("").values
X_test_min   = test_df["text_minimal"].fillna("").values
y_train_reg  = train_df["epss"].values
y_val_reg    = val_df["epss"].values
y_test_reg   = test_df["epss"].values
y_train_clf  = train_df["high_risk"].values
y_val_clf    = val_df["high_risk"].values
y_test_clf   = test_df["high_risk"].values
neg = (y_train_clf==0).sum(); pos = (y_train_clf==1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight: {scale_pos_weight:.1f}")

def reg_metrics(y_true, y_pred, name=""):
    mae=mean_absolute_error(y_true,y_pred)
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    r2=r2_score(y_true,y_pred)
    print(f"{name:25s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return {"name":name,"MAE":mae,"RMSE":rmse,"R2":r2}

def clf_metrics(y_true, y_prob, name="", threshold=0.5):
    y_pred=(y_prob>=threshold).astype(int)
    auc=roc_auc_score(y_true,y_prob)
    prauc=average_precision_score(y_true,y_prob)
    prec=precision_score(y_true,y_pred,zero_division=0)
    rec=recall_score(y_true,y_pred,zero_division=0)
    f1=f1_score(y_true,y_pred,zero_division=0)
    print(f"{name:25s}  AUC={auc:.4f}  PR-AUC={prauc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    return {"name":name,"AUC-ROC":auc,"PR-AUC":prauc,"Precision":prec,"Recall":rec,"F1":f1}

# ── CELL 4: M1 TF-IDF ─────────────────────────────────────────
TFIDF_TRAIN = DATA_DIR / "tfidf_train.npz"

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

print("\n=== M1: TF-IDF ===")
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=50_000,
                         sublinear_tf=True, min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train_full)
X_val_tfidf   = tfidf.transform(X_val_full)
X_test_tfidf  = tfidf.transform(X_test_full)
print(f"Vocab: {len(tfidf.vocabulary_):,}  Shape: {X_train_tfidf.shape}")

# ── CELL 5: M1 Ridge ──────────────────────────────────────────
print("\n=== M1: Ridge regression ===")
ridge = RidgeCV(alphas=[0.1,1.0,10.0,100.0], cv=5)
ridge.fit(X_train_tfidf, y_train_reg)
print(f"Best alpha: {ridge.alpha_}")
m1_reg_val  = reg_metrics(y_val_reg,  ridge.predict(X_val_tfidf),  "M1 Ridge (val)")
m1_reg_test = reg_metrics(y_test_reg, ridge.predict(X_test_tfidf), "M1 Ridge (test)")

# ── CELL 6: M1 Logistic ───────────────────────────────────────
print("\n=== M1: Logistic regression ===")
lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                         random_state=RANDOM_STATE, solver="saga", n_jobs=-1)
lr.fit(X_train_tfidf, y_train_clf)
m1_clf_val  = clf_metrics(y_val_clf,  lr.predict_proba(X_val_tfidf)[:,1],  "M1 LR (val)")
m1_clf_test = clf_metrics(y_test_clf, lr.predict_proba(X_test_tfidf)[:,1], "M1 LR (test)")

# ── CELL 7: M1 SHAP ───────────────────────────────────────────
SHAP_NAMES = DATA_DIR / "m1_shap_top30_names.npy"
SHAP_VALS  = DATA_DIR / "m1_shap_top30_vals.npy"

if SHAP_NAMES.exists() and SHAP_VALS.exists():
    print("\nSKIP: SHAP файли вже є на Drive")
    top_features = list(zip(
        np.load(SHAP_NAMES, allow_pickle=True),
        np.load(SHAP_VALS)
    ))
else:
    print("\n=== M1: SHAP analysis ===")
    N_SHAP = min(2000, X_test_tfidf.shape[0])
    X_shap_sample = X_test_tfidf[:N_SHAP]
    explainer_m1 = shap.LinearExplainer(lr, X_train_tfidf,
                                         feature_perturbation="interventional")
    shap_values_m1 = explainer_m1.shap_values(X_shap_sample)
    sv = shap_values_m1[1] if isinstance(shap_values_m1, list) else shap_values_m1
    if sp.issparse(sv):
        sv = sv.toarray()
    feature_names = tfidf.get_feature_names_out()
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:30]
    top_features = [(feature_names[i], mean_abs_shap[i]) for i in top_idx]
    np.save(SHAP_NAMES, np.array([f for f,_ in top_features]))
    np.save(SHAP_VALS,  np.array([v for _,v in top_features]))
    np.save(DATA_DIR / "m1_shap_matrix.npy", sv[:500])
    np.save(DATA_DIR / "m1_shap_feat_names.npy", feature_names[top_idx[:30]])
    print("SHAP збережено на Drive")

print("\nTop-30 tokens:")
for rank,(feat,val) in enumerate(top_features,1):
    print(f"  {rank:2d}. {feat:<30s}  {val:.5f}")

# ── CELL 8: M2 Sentence-BERT encoding ────────────────────────
EMB_TRAIN = DATA_DIR / "emb_train.npy"
EMB_VAL   = DATA_DIR / "emb_val.npy"
EMB_TEST  = DATA_DIR / "emb_test.npy"

if EMB_TRAIN.exists() and EMB_VAL.exists() and EMB_TEST.exists():
    print("\nSKIP: ембедінги вже є на Drive")
    emb_train = np.load(EMB_TRAIN)
    emb_val   = np.load(EMB_VAL)
    emb_test  = np.load(EMB_TEST)
    print(f"Shape: {emb_train.shape}")
else:
    from sentence_transformers import SentenceTransformer
    print("\n=== M2: Sentence-BERT encoding ===")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    print("Encoding train...")
    emb_train = sbert.encode(X_train_min.tolist(), batch_size=256,
                               show_progress_bar=True, convert_to_numpy=True)
    print("Encoding val...")
    emb_val   = sbert.encode(X_val_min.tolist(),   batch_size=256,
                               show_progress_bar=True, convert_to_numpy=True)
    print("Encoding test...")
    emb_test  = sbert.encode(X_test_min.tolist(),  batch_size=256,
                               show_progress_bar=True, convert_to_numpy=True)
    np.save(EMB_TRAIN, emb_train)
    np.save(EMB_VAL,   emb_val)
    np.save(EMB_TEST,  emb_test)
    print(f"Ембедінги збережено на Drive. Shape: {emb_train.shape}")

# ── CELL 9: M2 XGBoost regression ────────────────────────────
print("\n=== M2: XGBoost regression ===")
best_reg_score=float("inf"); best_reg_params={}
for n_est in [100,300]:
    for depth in [3,5]:
        for lr_val in [0.05,0.1]:
            model=xgb.XGBRegressor(n_estimators=n_est,max_depth=depth,
                learning_rate=lr_val,subsample=0.8,colsample_bytree=0.8,
                random_state=RANDOM_STATE,n_jobs=-1,verbosity=0)
            model.fit(emb_train,y_train_reg,
                      eval_set=[(emb_val,y_val_reg)],verbose=False)
            score=mean_absolute_error(y_val_reg,model.predict(emb_val))
            if score<best_reg_score:
                best_reg_score=score
                best_reg_params={"n_estimators":n_est,"max_depth":depth,"learning_rate":lr_val}
print(f"Best params: {best_reg_params}")
xgb_reg=xgb.XGBRegressor(**best_reg_params,subsample=0.8,colsample_bytree=0.8,
    random_state=RANDOM_STATE,n_jobs=-1,verbosity=0)
xgb_reg.fit(emb_train,y_train_reg)
m2_reg_val  = reg_metrics(y_val_reg,  xgb_reg.predict(emb_val),  "M2 XGB-reg (val)")
m2_reg_test = reg_metrics(y_test_reg, xgb_reg.predict(emb_test), "M2 XGB-reg (test)")

# ── CELL 10: M2 XGBoost classification ───────────────────────
print("\n=== M2: XGBoost classification ===")
best_clf_score=0; best_clf_params={}
for n_est in [100,300]:
    for depth in [3,5]:
        for lr_val in [0.05,0.1]:
            model=xgb.XGBClassifier(n_estimators=n_est,max_depth=depth,
                learning_rate=lr_val,subsample=0.8,colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,random_state=RANDOM_STATE,
                n_jobs=-1,eval_metric="auc",verbosity=0)
            model.fit(emb_train,y_train_clf,
                      eval_set=[(emb_val,y_val_clf)],verbose=False)
            score=roc_auc_score(y_val_clf,model.predict_proba(emb_val)[:,1])
            if score>best_clf_score:
                best_clf_score=score
                best_clf_params={"n_estimators":n_est,"max_depth":depth,"learning_rate":lr_val}
print(f"Best params: {best_clf_params}")
xgb_clf=xgb.XGBClassifier(**best_clf_params,subsample=0.8,colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,random_state=RANDOM_STATE,n_jobs=-1,verbosity=0)
xgb_clf.fit(emb_train,y_train_clf)
m2_clf_val  = clf_metrics(y_val_clf,  xgb_clf.predict_proba(emb_val)[:,1],  "M2 XGB-clf (val)")
m2_clf_test = clf_metrics(y_test_clf, xgb_clf.predict_proba(emb_test)[:,1], "M2 XGB-clf (test)")

# ── CELL 11: Ablation study ───────────────────────────────────
print("\n=== ABLATION STUDY ===")
import scipy.sparse as sp2

cvss_tr=train_df["cvss3"].fillna(0).values.reshape(-1,1)/10.0
cvss_te=test_df["cvss3"].fillna(0).values.reshape(-1,1)/10.0
kev_tr=train_df["kev_flag"].values.reshape(-1,1).astype(float)
kev_te=test_df["kev_flag"].values.reshape(-1,1).astype(float)

def run_ablation(X_tr, X_te, label):
    lr_ab=LogisticRegression(C=1.0,max_iter=1000,class_weight="balanced",
                              random_state=RANDOM_STATE,solver="saga",n_jobs=-1)
    lr_ab.fit(X_tr,y_train_clf)
    prob=lr_ab.predict_proba(X_te)[:,1]
    auc=roc_auc_score(y_test_clf,prob)
    prauc=average_precision_score(y_test_clf,prob)
    print(f"  {label:<35s}  AUC={auc:.4f}  PR-AUC={prauc:.4f}")
    return auc,prauc

run_ablation(X_train_tfidf, X_test_tfidf, "A: Text only")
run_ablation(sp2.hstack([X_train_tfidf,sp2.csr_matrix(cvss_tr)]),
             sp2.hstack([X_test_tfidf, sp2.csr_matrix(cvss_te)]), "B: Text + CVSS")
run_ablation(sp2.hstack([X_train_tfidf,sp2.csr_matrix(kev_tr)]),
             sp2.hstack([X_test_tfidf, sp2.csr_matrix(kev_te)]), "C: Text + KEV flag")
run_ablation(sp2.hstack([X_train_tfidf,sp2.csr_matrix(cvss_tr),sp2.csr_matrix(kev_tr)]),
             sp2.hstack([X_test_tfidf, sp2.csr_matrix(cvss_te),sp2.csr_matrix(kev_te)]), "D: Text + CVSS + KEV")

# ── CELL 12: Figures 7-10 ─────────────────────────────────────
FIG7=DATA_DIR/"fig07_roc_curves.png"
FIG8=DATA_DIR/"fig08_pr_curves.png"
FIG9=DATA_DIR/"fig09_shap_m1.png"
FIG10=DATA_DIR/"fig10_scatter_m1.png"

# Figure 7 — ROC
fig7,ax=plt.subplots(figsize=(7,5.5))
fpr1,tpr1,_=roc_curve(y_test_clf,lr.predict_proba(X_test_tfidf)[:,1])
auc1=roc_auc_score(y_test_clf,lr.predict_proba(X_test_tfidf)[:,1])
fpr2,tpr2,_=roc_curve(y_test_clf,xgb_clf.predict_proba(emb_test)[:,1])
auc2=roc_auc_score(y_test_clf,xgb_clf.predict_proba(emb_test)[:,1])
ax.plot(fpr1,tpr1,color=C_BLUE,lw=1.0,label=f"M1: TF-IDF + LR  (AUC={auc1:.3f})")
ax.plot(fpr2,tpr2,color=C_ORANGE,lw=1.0,label=f"M2: SBERT + XGB  (AUC={auc2:.3f})")
ax.plot([0,1],[0,1],color=C_GREY,lw=0.7,linestyle="--",label="Random classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right",fontsize=8.5)
ax.set_xlim(0,1); ax.set_ylim(0,1.01)
plt.tight_layout()
plt.savefig(FIG7,dpi=300); plt.savefig(DATA_DIR/"fig07_roc_curves.pdf"); plt.show()
print("Figure 7 saved.")

# Figure 8 — PR
fig8,ax=plt.subplots(figsize=(7,5.5))
prec1,rec1,_=precision_recall_curve(y_test_clf,lr.predict_proba(X_test_tfidf)[:,1])
prauc1=average_precision_score(y_test_clf,lr.predict_proba(X_test_tfidf)[:,1])
prec2,rec2,_=precision_recall_curve(y_test_clf,xgb_clf.predict_proba(emb_test)[:,1])
prauc2=average_precision_score(y_test_clf,xgb_clf.predict_proba(emb_test)[:,1])
baseline=y_test_clf.mean()
ax.plot(rec1,prec1,color=C_BLUE,lw=1.0,label=f"M1: TF-IDF + LR  (PR-AUC={prauc1:.3f})")
ax.plot(rec2,prec2,color=C_ORANGE,lw=1.0,label=f"M2: SBERT + XGB  (PR-AUC={prauc2:.3f})")
ax.axhline(baseline,color=C_GREY,lw=0.7,linestyle="--",label=f"Baseline={baseline:.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(loc="upper right",fontsize=8.5)
ax.set_xlim(0,1); ax.set_ylim(0,1.01)
plt.tight_layout()
plt.savefig(FIG8,dpi=300); plt.savefig(DATA_DIR/"fig08_pr_curves.pdf"); plt.show()
print("Figure 8 saved.")

# Figure 9 — SHAP
top30_names=np.load(SHAP_NAMES,allow_pickle=True)
top30_vals=np.load(SHAP_VALS)
fig9,ax=plt.subplots(figsize=(9,7))
colors=[C_RED if v>np.median(top30_vals) else C_BLUE for v in top30_vals]
ax.barh(range(len(top30_names)),top30_vals[::-1],color=colors[::-1],alpha=0.82,edgecolor="none")
ax.set_yticks(range(len(top30_names)))
ax.set_yticklabels(top30_names[::-1],fontsize=8.5)
ax.set_xlabel("Mean |SHAP value|  (impact on high-risk prediction)")
patch_h=mpatches.Patch(color=C_RED,alpha=0.82,label="Above median importance")
patch_l=mpatches.Patch(color=C_BLUE,alpha=0.82,label="Below median importance")
ax.legend(handles=[patch_h,patch_l],loc="lower right",fontsize=8)
plt.tight_layout()
plt.savefig(FIG9,dpi=300); plt.savefig(DATA_DIR/"fig09_shap_m1.pdf"); plt.show()
print("Figure 9 saved.")

# Figure 10 — Scatter
y_pred_s=ridge.predict(X_test_tfidf)
fig10,ax=plt.subplots(figsize=(6,6))
ax.scatter(y_test_reg,y_pred_s,alpha=0.15,s=4,color=C_BLUE,edgecolors="none")
lim=max(y_test_reg.max(),y_pred_s.max())*1.05
ax.plot([0,lim],[0,lim],color=C_RED,lw=0.8,linestyle="--",label="Perfect prediction")
ax.set_xlabel("Actual EPSS score"); ax.set_ylabel("Predicted EPSS score")
ax.set_xlim(0,lim); ax.set_ylim(0,lim)
from sklearn.metrics import r2_score
r2=r2_score(y_test_reg,y_pred_s)
ax.text(0.05,0.92,f"R2 = {r2:.4f}",transform=ax.transAxes,fontsize=9,color=C_GREY)
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(FIG10,dpi=300); plt.savefig(DATA_DIR/"fig10_scatter_m1.pdf"); plt.show()
print("Figure 10 saved.")

# ── CELL 13: Final summary ────────────────────────────────────
print("\n"+"="*65)
print("FINAL RESULTS M1 + M2 — copy into Tables 3 and 4")
print("="*65)
print(f"\nREGRESSION (test):")
print(f"{'Method':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
print("-"*51)
for m in [m1_reg_test,m2_reg_test]:
    print(f"{m['name']:<25} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['R2']:>8.4f}")
print(f"\nCLASSIFICATION (test):")
print(f"{'Method':<25} {'AUC-ROC':>8} {'PR-AUC':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
print("-"*71)
for m in [m1_clf_test,m2_clf_test]:
    print(f"{m['name']:<25} {m['AUC-ROC']:>8.4f} {m['PR-AUC']:>8.4f} {m['Precision']:>8.4f} {m['Recall']:>8.4f} {m['F1']:>8.4f}")

print("\n=== Файли збережені на Drive ===")
for f in sorted(DATA_DIR.iterdir()):
    print(f"  {f.name:<45} {f.stat().st_size/1024:.0f} KB")
