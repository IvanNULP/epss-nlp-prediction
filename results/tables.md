# Numerical Results

## Table 3 — Regression Results (test set, CVE 2025)

| Method | MAE | RMSE | R² |
|---|---|---|---|
| M1: TF-IDF + Ridge | 0.0330 | 0.0640 | -0.2993 |
| M2: SBERT + XGBoost | 0.0316 | 0.0627 | -0.2476 |
| M3: SecBERT FT | 0.0180 | 0.0724 | -0.6655 |

## Table 4 — Classification Results (test set, CVE 2025)

| Method | AUC-ROC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| M1: TF-IDF + LR | 0.8820 | 0.1234 | 0.0695 | 0.6812 | 0.1261 |
| M2: SBERT + XGBoost | 0.8323 | 0.0818 | 0.0614 | 0.5428 | 0.1104 |
| M3: SecBERT FT | 0.8657 | 0.1107 | 0.1361 | 0.3989 | 0.2030 |

## Table 5 — Ablation Study (M1, classification)

| Variant | AUC-ROC | PR-AUC |
|---|---|---|
| A: Text only | 0.8820 | 0.1234 |
| B: Text + CVSS score | 0.8716 | 0.1119 |
| C: Text + KEV flag | 0.9105 | 0.2751 |
| D: Text + CVSS + KEV | 0.9208 | 0.3082 |

## SHAP Top-30 Tokens (M1)

| Rank | Token | Mean |SHAP| |
|---|---|---|
| 1 | from through | 0.17504 |
| 2 | and | 0.16841 |
| 3 | vulnerability in | 0.15070 |
| 4 | through | 0.14358 |
| 5 | cross | 0.12124 |
| 6 | remote | 0.11243 |
| 7 | unauthenticated | 0.10717 |
| 8 | xss | 0.10517 |
| 9 | xss this | 0.10353 |
| 10 | cross site | 0.10279 |
| 11 | site request | 0.09641 |
| 12 | csrf | 0.08457 |
| 13 | code | 0.08227 |
| 14 | from | 0.07869 |
| 15 | stored | 0.07708 |
| 16 | before | 0.07652 |
| 17 | forgery | 0.07382 |
| 18 | local | 0.07193 |
| 19 | request forgery | 0.06887 |
| 20 | arbitrary | 0.06786 |
| 21 | attackers to | 0.06764 |
| 22 | wordpress | 0.06693 |
| 23 | execute | 0.06348 |
| 24 | for | 0.06263 |
| 25 | csrf vulnerability | 0.05994 |
| 26 | due to | 0.05758 |
| 27 | affects | 0.05755 |
| 28 | it | 0.05612 |
| 29 | via crafted | 0.05378 |
| 30 | request | 0.05078 |

## Dataset Statistics

| Parameter | Value |
|---|---|
| Total records | 218,292 |
| High-risk (EPSS >= 0.1) | 10,547 (4.83%) |
| KEV-flagged | 1,336 (0.61%) |
| Median EPSS | 0.0020 |
| Mean EPSS | 0.0259 |
| Median description length | 43 words |
| Year range | 2010–2026 |
| EPSS snapshot | EPSSv4, 2026-05-01 |
| Train set | 129,509 (2010–2024) |
| Val set | 28,430 (2010–2024) |
| Test set | 41,413 (2025 only) |
| High-risk in train | 6.3% |
| High-risk in test | 1.3% |
