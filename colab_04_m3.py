# =============================================================
# COLAB 04 — Model M3: SecBERT Fine-tuning
# =============================================================

# ── CELL 1: Drive + Install ───────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# !pip install transformers torch datasets scikit-learn
# !pip install pandas pyarrow matplotlib scipy

from pathlib import Path
DATA_DIR = Path("/content/drive/MyDrive/article_data")

existing = [f.name for f in DATA_DIR.iterdir() if f.is_file()]
print(f"Файлів на Drive: {len(existing)}")
for f in sorted(existing):
    print(f"  {f}")

# ── CELL 2: Imports ───────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings("ignore")

DATA_DIR     = Path("/content/drive/MyDrive/article_data")
MODEL_NAME   = "jackaduma/SecBERT"
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.06
N_TRAIN_SUB  = 50_000
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("УВАГА: GPU не знайдено! Увімкни T4: Runtime -> Change runtime type -> GPU")

matplotlib.rcParams.update({
    "font.family":"Liberation Sans","font.size":9,"font.weight":"normal",
    "axes.linewidth":0.8,"lines.linewidth":1.0,"legend.frameon":False,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight",
})
C_BLUE="#4472C4";C_RED="#C00000";C_ORANGE="#ED7D31";C_GREEN="#70AD47";C_GREY="#888888"

# ── CELL 3: Завантаження даних ────────────────────────────────
train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df   = pd.read_parquet(DATA_DIR / "val.parquet")
test_df  = pd.read_parquet(DATA_DIR / "test.parquet")
print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

from sklearn.model_selection import train_test_split
_, train_sub = train_test_split(
    train_df, test_size=N_TRAIN_SUB,
    stratify=train_df["high_risk"], random_state=RANDOM_STATE)
train_sub = train_sub.reset_index(drop=True)
print(f"Subsample: {len(train_sub):,}  high-risk: {train_sub['high_risk'].mean()*100:.1f}%")

y_test_reg = test_df["epss"].values
y_test_clf = test_df["high_risk"].values
y_val_reg  = val_df["epss"].values
y_val_clf  = val_df["high_risk"].values

# ── CELL 4: Dataset ───────────────────────────────────────────
print("\nLoading SecBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class CVEDataset(Dataset):
    def __init__(self, texts, labels_reg, labels_clf, max_len=MAX_LEN):
        self.texts=texts; self.labels_reg=labels_reg
        self.labels_clf=labels_clf; self.max_len=max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc=tokenizer(str(self.texts[idx]),max_length=self.max_len,
                      padding="max_length",truncation=True,return_tensors="pt")
        return {"input_ids":enc["input_ids"].squeeze(),
                "attention_mask":enc["attention_mask"].squeeze(),
                "label_reg":torch.tensor(self.labels_reg[idx],dtype=torch.float),
                "label_clf":torch.tensor(self.labels_clf[idx],dtype=torch.float)}

train_dataset=CVEDataset(train_sub["text_minimal"].fillna("").values,
                          train_sub["epss"].values,train_sub["high_risk"].values)
val_dataset  =CVEDataset(val_df["text_minimal"].fillna("").values,
                          val_df["epss"].values,val_df["high_risk"].values)
test_dataset =CVEDataset(test_df["text_minimal"].fillna("").values,
                          test_df["epss"].values,test_df["high_risk"].values)

train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
val_loader  =DataLoader(val_dataset,  batch_size=BATCH_SIZE*2,shuffle=False,num_workers=2)
test_loader =DataLoader(test_dataset, batch_size=BATCH_SIZE*2,shuffle=False,num_workers=2)
print(f"Batches: train={len(train_loader)}  val={len(val_loader)}  test={len(test_loader)}")

# ── CELL 5: Model ─────────────────────────────────────────────
class SecBERTPredictor(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(model_name)
        hidden_size=self.encoder.config.hidden_size
        self.dropout=nn.Dropout(dropout)
        self.reg_head=nn.Sequential(nn.Linear(hidden_size,1),nn.Sigmoid())
        self.clf_head=nn.Linear(hidden_size,1)
    def forward(self,input_ids,attention_mask):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        cls=self.dropout(out.last_hidden_state[:,0,:])
        return self.reg_head(cls).squeeze(-1), self.clf_head(cls).squeeze(-1)

print("\nLoading SecBERT...")
model=SecBERTPredictor(MODEL_NAME).to(DEVICE)
n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n_params:,}")

# ── CELL 6: Training ──────────────────────────────────────────
neg=(train_sub["high_risk"]==0).sum(); pos=(train_sub["high_risk"]==1).sum()
pos_weight=torch.tensor([neg/pos],dtype=torch.float).to(DEVICE)
print(f"pos_weight: {pos_weight.item():.1f}")

optimizer=AdamW(model.parameters(),lr=LR,weight_decay=0.01)
total_steps=len(train_loader)*EPOCHS
warmup_steps=int(total_steps*WARMUP_RATIO)
scheduler=get_linear_schedule_with_warmup(optimizer,warmup_steps,total_steps)
loss_reg_fn=nn.MSELoss()
loss_clf_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print(f"\nTraining {EPOCHS} epochs | {total_steps} steps | warmup {warmup_steps}")
best_val_auc=0; best_state=None

for epoch in range(EPOCHS):
    model.train(); epoch_loss=0
    for step,batch in enumerate(train_loader):
        input_ids=batch["input_ids"].to(DEVICE)
        attn_mask=batch["attention_mask"].to(DEVICE)
        lab_reg=batch["label_reg"].to(DEVICE)
        lab_clf=batch["label_clf"].to(DEVICE)
        optimizer.zero_grad()
        pred_reg,logits=model(input_ids,attn_mask)
        loss=loss_reg_fn(pred_reg,lab_reg)+loss_clf_fn(logits,lab_clf)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step(); scheduler.step()
        epoch_loss+=loss.item()
        if step%200==0:
            print(f"  Epoch {epoch+1}/{EPOCHS}  Step {step}/{len(train_loader)}  Loss={loss.item():.4f}")
    avg_loss=epoch_loss/len(train_loader)
    model.eval(); val_probs=[]; val_preds=[]
    with torch.no_grad():
        for batch in val_loader:
            pred_reg,logits=model(batch["input_ids"].to(DEVICE),
                                   batch["attention_mask"].to(DEVICE))
            val_probs.extend(torch.sigmoid(logits).cpu().numpy())
            val_preds.extend(pred_reg.cpu().numpy())
    val_auc=roc_auc_score(y_val_clf,np.array(val_probs))
    val_mae=mean_absolute_error(y_val_reg,np.array(val_preds))
    print(f"\nEpoch {epoch+1}: loss={avg_loss:.4f}  val AUC={val_auc:.4f}  val MAE={val_mae:.4f}")
    if val_auc>best_val_auc:
        best_val_auc=val_auc
        best_state={k:v.clone() for k,v in model.state_dict().items()}
        print(f"  Best model saved (val AUC={best_val_auc:.4f})")

# ── CELL 7: Test evaluation ───────────────────────────────────
print("\n=== M3: Test evaluation ===")
model.load_state_dict(best_state); model.eval()
test_probs=[]; test_preds=[]
with torch.no_grad():
    for batch in test_loader:
        pred_reg,logits=model(batch["input_ids"].to(DEVICE),
                               batch["attention_mask"].to(DEVICE))
        test_probs.extend(torch.sigmoid(logits).cpu().numpy())
        test_preds.extend(pred_reg.cpu().numpy())
test_probs=np.array(test_probs); test_preds=np.array(test_preds)
test_pred_clf=(test_probs>=0.5).astype(int)

mae3=mean_absolute_error(y_test_reg,test_preds)
rmse3=np.sqrt(mean_squared_error(y_test_reg,test_preds))
r2_3=r2_score(y_test_reg,test_preds)
auc3=roc_auc_score(y_test_clf,test_probs)
prauc3=average_precision_score(y_test_clf,test_probs)
prec3=precision_score(y_test_clf,test_pred_clf,zero_division=0)
rec3=recall_score(y_test_clf,test_pred_clf,zero_division=0)
f1_3=f1_score(y_test_clf,test_pred_clf,zero_division=0)

print(f"REGRESSION:  MAE={mae3:.4f}  RMSE={rmse3:.4f}  R2={r2_3:.4f}")
print(f"CLASSIFICATION:  AUC={auc3:.4f}  PR-AUC={prauc3:.4f}  P={prec3:.4f}  R={rec3:.4f}  F1={f1_3:.4f}")

np.save(DATA_DIR/"m3_test_probs.npy",test_probs)
np.save(DATA_DIR/"m3_test_preds.npy",test_preds)
print("M3 результати збережено на Drive")

# ── CELL 8: Updated Figures 7+8 with M3 ──────────────────────
# Rebuild M1 for curves
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
X_train_full=train_df["text_full"].fillna("").values
X_test_full =test_df["text_full"].fillna("").values
tfidf_f=TfidfVectorizer(ngram_range=(1,2),max_features=50_000,sublinear_tf=True,min_df=2)
X_tr_f=tfidf_f.fit_transform(X_train_full)
X_te_f=tfidf_f.transform(X_test_full)
lr_f=LogisticRegression(C=1.0,max_iter=1000,class_weight="balanced",
                         random_state=42,solver="saga",n_jobs=-1)
lr_f.fit(X_tr_f,train_df["high_risk"].values)
m1_probs=lr_f.predict_proba(X_te_f)[:,1]

# Load M2
try:
    import xgboost as xgb2
    emb_tr2=np.load(DATA_DIR/"emb_train.npy")
    emb_te2=np.load(DATA_DIR/"emb_test.npy")
    neg2=(train_df["high_risk"]==0).sum(); pos2=(train_df["high_risk"]==1).sum()
    xgb_f=xgb2.XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.1,
        subsample=0.8,colsample_bytree=0.8,scale_pos_weight=neg2/pos2,
        random_state=42,n_jobs=-1,verbosity=0)
    xgb_f.fit(emb_tr2,train_df["high_risk"].values)
    m2_probs=xgb_f.predict_proba(emb_te2)[:,1]
    print("M2 ембедінги завантажено з Drive")
except Exception as e:
    print(f"M2 недоступно: {e}"); m2_probs=None

# Figure 7 updated
fig7,ax=plt.subplots(figsize=(7,5.5))
fpr1,tpr1,_=roc_curve(y_test_clf,m1_probs)
auc1=roc_auc_score(y_test_clf,m1_probs)
ax.plot(fpr1,tpr1,color=C_BLUE,lw=1.0,label=f"M1: TF-IDF + LR     (AUC={auc1:.3f})")
if m2_probs is not None:
    fpr2,tpr2,_=roc_curve(y_test_clf,m2_probs)
    auc2=roc_auc_score(y_test_clf,m2_probs)
    ax.plot(fpr2,tpr2,color=C_ORANGE,lw=1.0,label=f"M2: SBERT + XGB    (AUC={auc2:.3f})")
fpr3,tpr3,_=roc_curve(y_test_clf,test_probs)
ax.plot(fpr3,tpr3,color=C_GREEN,lw=1.0,label=f"M3: SecBERT FT     (AUC={auc3:.3f})")
ax.plot([0,1],[0,1],color=C_GREY,lw=0.7,linestyle="--",label="Random classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right",fontsize=8.5); ax.set_xlim(0,1); ax.set_ylim(0,1.01)
plt.tight_layout()
plt.savefig(DATA_DIR/"fig07_roc_curves.png",dpi=300)
plt.savefig(DATA_DIR/"fig07_roc_curves.pdf"); plt.show()
print("Figure 7 updated.")

# Figure 8 updated
fig8,ax=plt.subplots(figsize=(7,5.5))
baseline=y_test_clf.mean()
prec1r,rec1r,_=precision_recall_curve(y_test_clf,m1_probs)
prauc1=average_precision_score(y_test_clf,m1_probs)
ax.plot(rec1r,prec1r,color=C_BLUE,lw=1.0,label=f"M1: TF-IDF + LR     (PR-AUC={prauc1:.3f})")
if m2_probs is not None:
    prec2r,rec2r,_=precision_recall_curve(y_test_clf,m2_probs)
    prauc2=average_precision_score(y_test_clf,m2_probs)
    ax.plot(rec2r,prec2r,color=C_ORANGE,lw=1.0,label=f"M2: SBERT + XGB    (PR-AUC={prauc2:.3f})")
prec3r,rec3r,_=precision_recall_curve(y_test_clf,test_probs)
ax.plot(rec3r,prec3r,color=C_GREEN,lw=1.0,label=f"M3: SecBERT FT     (PR-AUC={prauc3:.3f})")
ax.axhline(baseline,color=C_GREY,lw=0.7,linestyle="--",label=f"Baseline={baseline:.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(loc="upper right",fontsize=8.5); ax.set_xlim(0,1); ax.set_ylim(0,1.01)
plt.tight_layout()
plt.savefig(DATA_DIR/"fig08_pr_curves.png",dpi=300)
plt.savefig(DATA_DIR/"fig08_pr_curves.pdf"); plt.show()
print("Figure 8 updated.")

# ── CELL 9: Final summary all methods ────────────────────────
print("\n"+"="*70)
print("FINAL RESULTS — ALL THREE METHODS")
print("="*70)
print(f"\nREGRESSION (test):")
print(f"{'Method':<28} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
print("-"*54)
print(f"{'M1: TF-IDF + Ridge':<28} {0.0330:>8.4f} {0.0640:>8.4f} {-0.2993:>8.4f}")
print(f"{'M2: SBERT + XGB':<28} {0.0316:>8.4f} {0.0627:>8.4f} {-0.2476:>8.4f}")
print(f"{'M3: SecBERT FT':<28} {mae3:>8.4f} {rmse3:>8.4f} {r2_3:>8.4f}")
print(f"\nCLASSIFICATION (test):")
print(f"{'Method':<28} {'AUC-ROC':>8} {'PR-AUC':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
print("-"*70)
print(f"{'M1: TF-IDF + LR':<28} {0.8820:>8.4f} {0.1234:>8.4f} {0.0695:>8.4f} {0.6812:>8.4f} {0.1261:>8.4f}")
print(f"{'M2: SBERT + XGB':<28} {0.8323:>8.4f} {0.0818:>8.4f} {0.0614:>8.4f} {0.5428:>8.4f} {0.1104:>8.4f}")
print(f"{'M3: SecBERT FT':<28} {auc3:>8.4f} {prauc3:>8.4f} {prec3:>8.4f} {rec3:>8.4f} {f1_3:>8.4f}")

print("\n=== Всі файли на Drive ===")
for f in sorted(DATA_DIR.iterdir()):
    print(f"  {f.name:<45} {f.stat().st_size/1024:.0f} KB")
