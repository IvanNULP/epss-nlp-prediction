# NLP-based EPSS Prediction from CVE Descriptions

**Replication package for the article:**
> Assessment of Software Vulnerability Exploitation Risk Using Natural Language Processing Methods Based on Open NVD and EPSS Data

---

## Overview

This repository contains the complete replication package for a study that evaluates the predictive potential of CVE textual descriptions for the EPSS (Exploit Prediction Scoring System) score. Three text encoding methods are compared:

- **M1:** TF-IDF + Ridge Regression / Logistic Regression (baseline)
- **M2:** Sentence-BERT (all-MiniLM-L6-v2) + XGBoost
- **M3:** Fine-tuned SecBERT (jackaduma/SecBERT)

---

## Key Results

### Regression (test set, CVE published in 2025)

| Method | MAE | RMSE | R² |
|---|---|---|---|
| M1: TF-IDF + Ridge | 0.0330 | 0.0640 | -0.2993 |
| M2: SBERT + XGBoost | 0.0316 | 0.0627 | -0.2476 |
| M3: SecBERT FT | 0.0180 | 0.0724 | -0.6655 |

### Classification (test set, CVE published in 2025)

| Method | AUC-ROC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| M1: TF-IDF + LR | 0.8820 | 0.1234 | 0.0695 | 0.6812 | 0.1261 |
| M2: SBERT + XGBoost | 0.8323 | 0.0818 | 0.0614 | 0.5428 | 0.1104 |
| M3: SecBERT FT | 0.8657 | 0.1107 | 0.1361 | 0.3989 | 0.2030 |

### Ablation Study (M1, classification)

| Variant | AUC-ROC | PR-AUC |
|---|---|---|
| A: Text only | 0.8820 | 0.1234 |
| B: Text + CVSS | 0.8716 | 0.1119 |
| C: Text + KEV flag | 0.9105 | 0.2751 |
| D: Text + CVSS + KEV | 0.9208 | 0.3082 |

---

## Dataset

The merged CVE-EPSS dataset (218,292 records) is archived on Zenodo:
> DOI: [to be assigned upon publication]

**Sources:**
- NVD API 2.0: https://services.nvd.nist.gov/rest/json/cves/2.0
- EPSS v4 (snapshot 2026-05-01): https://epss.cyentia.com
- CISA KEV: https://www.cisa.gov/known-exploited-vulnerabilities-catalog

**Dataset parameters:**
- Records: 218,292
- Year range: 2010–2026
- EPSS snapshot: 2026-05-01 (EPSSv4)
- Train: 129,509 (2010–2024)
- Val: 28,430 (2010–2024, stratified)
- Test: 41,413 (2025 only)
- High-risk threshold: EPSS >= 0.1
- High-risk in train: 6.3% | test: 1.3%

---

## Repository Structure

```
epss-nlp-prediction/
├── README.md
├── requirements.txt
├── colab_01_data_collection.py   # Data collection: NVD + EPSS + KEV
├── colab_03_m1_m2.py             # Models M1 and M2 + SHAP + Ablation
├── colab_04_m3.py                # Model M3: SecBERT fine-tuning
├── fig01_v2.py                   # Figure 1: Conceptual diagram
├── fig06_architecture.py         # Figure 6: Architecture diagram (Matplotlib)
├── gen_fig06_v2.py               # Figure 6: Architecture diagram (SVG/Visio)
└── results/
    └── tables.md                 # All numerical results
```

---

## How to Reproduce

### Requirements
- Google Colab with T4 GPU (for colab_04_m3.py)
- Google Drive mounted at `/content/drive/MyDrive/article_data`
- Python 3.10+

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2: Run in order
```
colab_01_data_collection.py  →  colab_03_m1_m2.py  →  colab_04_m3.py
```

Each script checks Google Drive for existing files and **skips completed steps automatically**. If interrupted, simply re-run — it will continue from where it stopped.

### Step 3: Expected outputs
After all scripts complete, Google Drive will contain:
- `train.parquet`, `val.parquet`, `test.parquet`, `full_clean.parquet`
- `emb_train.npy`, `emb_val.npy`, `emb_test.npy` (SBERT embeddings)
- `m1_shap_top30_names.npy`, `m1_shap_top30_vals.npy`
- `fig07_scatter_m1.png/.pdf`, `fig08_roc_curves.png/.pdf`
- `fig09_pr_curves.png/.pdf`, `fig10_shap_m1.png/.pdf`

### NVD API Key (optional but recommended)
Without an API key, data collection takes ~90 minutes.
With a free API key (https://nvd.nist.gov/developers/request-an-api-key), it takes ~7 minutes.

In `colab_01_data_collection.py`, set:
```python
API_KEY = "your-key-here"
```

---

## Reproducibility

All stochastic components use `random_state = 42`. M1 and M2 results are exactly reproducible (verified by running twice — all metrics match to 4 decimal places). M3 results may vary slightly between runs due to GPU non-determinism in fine-tuning despite the fixed seed.

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{epss-nlp-2026,
  title   = {Assessment of Software Vulnerability Exploitation Risk Using
             Natural Language Processing Methods Based on Open NVD and EPSS Data},
  journal = {Computers \& Security},
  year    = {2026},
  note    = {manuscript under review}
}
```

---

## License

Code: MIT License
Dataset: Creative Commons Attribution 4.0 (CC BY 4.0)
