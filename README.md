# NLP-based EPSS Prediction from CVE Descriptions

**Replication package for:**
> Assessment of Software Vulnerability Exploitation Risk Using Natural Language Processing Methods Based on Open NVD and EPSS Data

---

## Overview

This repository contains the complete replication package for a study evaluating the predictive potential of CVE textual descriptions for the EPSS (Exploit Prediction Scoring System) score. Three text encoding methods of varying architectural complexity are compared, with SHAP-based interpretability analysis and bootstrap confidence intervals.

**Dataset:** 218,292 CVE records (2010–2026), merged from NVD API 2.0, EPSSv4 (snapshot 2026-05-01), and CISA KEV catalog. Archived on Zenodo: [DOI: 10.5281/zenodo.20019540](https://doi.org/10.5281/zenodo.20019540)

---

## Key Results

### Classification — test set (CVE published in 2025)

| Method | AUC-ROC | 95% CI | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| M1: TF-IDF + LR | **0.8820** | [0.869–0.895] | **0.1234** | 0.0695 | 0.6812 | 0.1261 |
| M2: SBERT + XGBoost | 0.8323 | [0.816–0.848] | 0.0818 | 0.0614 | 0.5428 | 0.1104 |
| M3: SecBERT Full | 0.8775 | [0.863–0.891] | 0.1111 | 0.0673 | **0.7213** | 0.1232 |

> M1 vs M3 difference is **not statistically significant** (bootstrap p = 0.218). Both methods are competitive.

### Regression — test set (CVE published in 2025)

| Method | MAE | RMSE | R² |
|---|---|---|---|
| M1: TF-IDF + Ridge | 0.0330 | 0.0640 | -0.2993 |
| M2: SBERT + XGBoost | 0.0316 | 0.0627 | -0.2476 |
| M3: SecBERT Full | 0.0268 | 0.0677 | -0.4559 |

> Negative R² for all methods is a documented consequence of temporal shift (train: 2010–2024, test: 2025). Regression is reported as a negative result; the meaningful contribution lies in ranking and classification.

### Ablation Study — M1, classification

| Variant | AUC-ROC | PR-AUC |
|---|---|---|
| A: Text only | 0.8820 | 0.1234 |
| B: Text + CVSS | 0.8716 | 0.1119 |
| C: Text + KEV flag | 0.9105 | 0.2751 |
| D: Text + CVSS + KEV | **0.9208** | **0.3082** |

### Precision@k — test set

| Top-k% | CVEs reviewed | M1 | M2 | M3 | Baseline |
|---|---|---|---|---|---|
| 1% | 414 | **21.0%** | 14.3% | 16.9% | 1.3% |
| 2% | 828 | **17.8%** | 12.3% | 17.3% | 1.3% |
| 5% | 2,070 | **12.1%** | 8.9% | 12.1% | 1.3% |
| 10% | 4,141 | 8.3% | 6.5% | **8.5%** | 1.3% |
| 20% | 8,282 | 5.1% | 4.6% | **5.3%** | 1.3% |

**Effort-Coverage (M1):** reviewing top 10% of CVEs by score finds 63% of all high-risk vulnerabilities; top 20% finds 77%.

---

## Repository Structure

```
epss-nlp-prediction/
├── README.md
├── requirements.txt
│
├── colab_01_data_collection.py     # Step 1: Data collection (NVD + EPSS + KEV)
├── colab_03_m1_m2.py               # Step 2: Models M1 and M2 + SHAP + Ablation
├── colab_04_m3.py                  # Step 3: Model M3 — SecBERT fine-tuning (GPU)
│
├── colab_step1_shap_clean.py       # SHAP with CVE-domain stop-list (clean config)
├── colab_step2_precision_k.py      # Precision@k + Effort-Coverage curves
├── colab_step3_secbert_full.py     # SecBERT on FULL dataset (129k samples)
├── colab_step4_bootstrap_ci.py     # Bootstrap 95% CI for all metrics
│
├── fig01_v2.py                     # Figure 1: Conceptual diagram
├── fig06_architecture.py           # Figure 6: Architecture diagram (Matplotlib)
├── gen_fig06_v2.py                 # Figure 6: Architecture diagram (SVG)
│
└── results/
    └── tables.md                   # All numerical results
```

---

## How to Reproduce

### Requirements
- Google Colab (T4 GPU required for `colab_04_m3.py` and `colab_step3_secbert_full.py`)
- Google Drive mounted at `/content/drive/MyDrive/article_data`
- Python 3.10+

### Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Execution order

```
Step 1:  colab_01_data_collection.py    (~60-90 min, no GPU)
Step 2:  colab_03_m1_m2.py             (~50-70 min, GPU recommended)
Step 3:  colab_04_m3.py                (~2-3 hours, T4 GPU required)
Step 4:  colab_step1_shap_clean.py     (~20 min, no GPU)
Step 5:  colab_step2_precision_k.py    (~15 min, no GPU)
Step 6:  colab_step3_secbert_full.py   (~4-5 hours, T4 GPU required)
Step 7:  colab_step4_bootstrap_ci.py   (~10 min, no GPU)
```

Each script checks Google Drive for existing files and **skips completed steps automatically**. If interrupted, re-run and it continues from where it stopped.

### NVD API Key (optional but recommended)
Without a key: ~90 minutes for data collection.
With a free key (https://nvd.nist.gov/developers/request-an-api-key): ~7 minutes.

In `colab_01_data_collection.py` set:
```python
API_KEY = "your-key-here"
```

---

## Dataset Parameters

| Parameter | Value |
|---|---|
| Total records | 218,292 |
| High-risk (EPSS >= 0.1) | 10,547 (4.83%) |
| KEV-flagged | 1,336 (0.61%) |
| Median EPSS | 0.0020 |
| Mean EPSS | 0.0259 |
| Median description length | 43 words |
| Year range | 2010-2026 |
| EPSS snapshot | EPSSv4, 2026-05-01 |
| Train | 129,509 (2010-2024) |
| Val | 28,430 (2010-2024, stratified) |
| Test | 41,413 (2025 only) |
| High-risk in train | 6.3% |
| High-risk in test | 1.3% |

---

## SHAP Analysis — Top Linguistic Risk Markers

SHAP analysis on M1 (CVE-domain stop-list configuration, AUC-ROC = 0.872) identified four semantically coherent groups:

1. **Vulnerability types** — `cross`, `xss`, `cross site`, `csrf`, `stored`, `stored xss`
2. **Attack vectors and conditions** — `unauthenticated`, `remote`, `local`, `remote attackers`
3. **Target platforms** — `php`, `wordpress`, `plugin`
4. **Attack mechanisms** — `code`, `execution`, `execute`, `authentication`

> Note: Main classification metrics in the paper (AUC-ROC = 0.882) correspond to the baseline TF-IDF configuration without stop-list. The stop-list configuration (AUC-ROC = 0.872) is used exclusively for SHAP interpretation to reduce NVD template artifacts.

---

## Reproducibility

All stochastic components use `random_state = 42`. M1 and M2 results are exactly reproducible (verified by independent re-runs — all metrics match to 4 decimal places). M3 results may vary slightly between runs due to GPU non-determinism in fine-tuning despite the fixed seed.

---

## Citation

```bibtex
@article{opirskyy2026epss,
  author  = {Opirskyy, Ivan},
  title   = {Assessment of Software Vulnerability Exploitation Risk
             Using Natural Language Processing Methods Based on
             Open NVD and EPSS Data},
  journal = {Computers & Security},
  year    = {2026},
  note    = {manuscript under review}
}
```

---

## Related Resources

- **Dataset (Zenodo):** https://doi.org/10.5281/zenodo.20019540
- **NVD API:** https://nvd.nist.gov
- **EPSS daily snapshots:** https://epss.cyentia.com
- **CISA KEV catalog:** https://www.cisa.gov/known-exploited-vulnerabilities-catalog
- **FIRST EPSS model description:** https://www.first.org/epss/model

---

## License

Code: MIT License  
Dataset: Creative Commons Attribution 4.0 (CC BY 4.0)
