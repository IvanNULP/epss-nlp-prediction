# =============================================================
# COLAB 01 — Data Collection & Preprocessing
# =============================================================

# ── CELL 1: Підключення Google Drive ──────────────────────────
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
DATA_DIR = Path("/content/drive/MyDrive/article_data")
DATA_DIR.mkdir(exist_ok=True)
print(f"Data dir: {DATA_DIR}")

# Перевірка що вже є
existing = [f.name for f in DATA_DIR.iterdir() if f.is_file()]
print(f"\nФайли на Drive ({len(existing)}):")
for f in sorted(existing):
    size = (DATA_DIR / f).stat().st_size / 1024
    print(f"  {f:<35} {size:>8.1f} KB")

# ── CELL 2: Install & imports ─────────────────────────────────
# !pip install requests pandas pyarrow spacy tqdm matplotlib scipy
# !python -m spacy download en_core_web_sm

import requests
import pandas as pd
import json
import gzip
import io
import time
import re
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from datetime import date, datetime, timedelta
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("/content/drive/MyDrive/article_data")

EPSS_SNAPSHOT_DATE = "2026-05-01"
NVD_START_YEAR     = 2010
NVD_END_YEAR       = 2026
CHUNK_DAYS         = 110
PAUSE_SEC          = 0.7
RANDOM_STATE       = 42

matplotlib.rcParams.update({
    "font.family":       "Liberation Sans",
    "font.size":         9,
    "font.weight":       "normal",
    "axes.titleweight":  "normal",
    "axes.labelweight":  "normal",
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.0,
    "legend.frameon":    False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

C_BLUE   = "#4472C4"
C_RED    = "#C00000"
C_ORANGE = "#ED7D31"
C_GREY   = "#888888"

print("Imports OK")

# ── CELL 3: Збір NVD (пропускається якщо nvd_raw.parquet є) ───
NVD_FILE = DATA_DIR / "nvd_raw.parquet"

if NVD_FILE.exists():
    print(f"SKIP: nvd_raw.parquet вже є ({NVD_FILE.stat().st_size/1024/1024:.1f} MB)")
    nvd_df = pd.read_parquet(NVD_FILE)
    print(f"Завантажено: {len(nvd_df):,} записів")
else:
    print("Збираємо NVD CVE через API...")

    BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    API_KEY  = None  # вставте свій ключ для 10x швидкості

    def parse_cve(cve):
        cve_id = cve.get("id", "")
        desc = ""
        for d in cve.get("descriptions", []):
            if d["lang"] == "en":
                desc = d["value"]
                break
        if any(p in desc for p in ["** RESERVED **", "** REJECT **", "DO NOT USE"]):
            return None
        cvss3 = None
        metrics = cve.get("metrics", {})
        for key in ["cvssMetricV31", "cvssMetricV30"]:
            if key in metrics and metrics[key]:
                cvss3 = metrics[key][0]["cvssData"]["baseScore"]
                break
        cwe = None
        weaknesses = cve.get("weaknesses", [])
        if weaknesses:
            for wd in weaknesses[0].get("description", []):
                if wd["lang"] == "en":
                    cwe = wd["value"]
                    break
        pub_date = cve.get("published", "")[:10]
        return {"cve_id": cve_id, "description": desc,
                "cvss3": cvss3, "cwe": cwe, "pub_date": pub_date}

    def fetch_chunk(start_dt, end_dt):
        headers = {"apiKey": API_KEY} if API_KEY else {}
        params = {
            "pubStartDate":   start_dt.strftime("%Y-%m-%dT00:00:00.000"),
            "pubEndDate":     end_dt.strftime("%Y-%m-%dT23:59:59.999"),
            "resultsPerPage": 2000,
            "startIndex":     0,
        }
        results = []
        try:
            r = requests.get(BASE_URL, params=params,
                             headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  Error: {e}")
            return []
        total = data.get("totalResults", 0)
        if total == 0:
            return []
        for item in data.get("vulnerabilities", []):
            rec = parse_cve(item["cve"])
            if rec:
                results.append(rec)
        params["startIndex"] = params["resultsPerPage"]
        while params["startIndex"] < total:
            time.sleep(PAUSE_SEC)
            try:
                r = requests.get(BASE_URL, params=params,
                                 headers=headers, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"  Pagination error: {e}")
                break
            for item in data.get("vulnerabilities", []):
                rec = parse_cve(item["cve"])
                if rec:
                    results.append(rec)
            params["startIndex"] += params["resultsPerPage"]
            time.sleep(PAUSE_SEC)
        return results

    start_dt = datetime(NVD_START_YEAR, 1, 1)
    end_dt   = datetime(NVD_END_YEAR, 12, 31)
    chunks   = []
    cur = start_dt
    while cur <= end_dt:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end_dt)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)

    print(f"Всього чанків: {len(chunks)}")
    all_records = []
    for i, (s, e) in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}: {s.date()} → {e.date()}", end=" ... ")
        recs = fetch_chunk(s, e)
        all_records.extend(recs)
        print(f"{len(recs)} CVEs (total: {len(all_records):,})")
        time.sleep(PAUSE_SEC)

    nvd_df = pd.DataFrame(all_records)
    nvd_df.drop_duplicates(subset="cve_id", inplace=True)
    nvd_df.reset_index(drop=True, inplace=True)
    nvd_df.to_parquet(NVD_FILE, index=False)
    print(f"Збережено: {len(nvd_df):,} записів → {NVD_FILE}")

# ── CELL 4: EPSS (пропускається якщо epss.parquet є) ──────────
EPSS_FILE = DATA_DIR / "epss.parquet"

if EPSS_FILE.exists():
    print(f"SKIP: epss.parquet вже є")
    epss_df = pd.read_parquet(EPSS_FILE)
    print(f"Завантажено: {len(epss_df):,} записів")
else:
    print(f"Завантажуємо EPSS snapshot {EPSS_SNAPSHOT_DATE}...")
    url = f"https://epss.cyentia.com/epss_scores-{EPSS_SNAPSHOT_DATE}.csv.gz"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with gzip.open(io.BytesIO(r.content), "rt") as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("cve"))
    csv_text = "".join(lines[header_idx:])
    epss_df = pd.read_csv(io.StringIO(csv_text))
    epss_df.columns = ["cve_id", "epss", "percentile"]
    epss_df["epss"]       = epss_df["epss"].astype(float)
    epss_df["percentile"] = epss_df["percentile"].astype(float)
    epss_df.to_parquet(EPSS_FILE, index=False)
    print(f"Збережено: {len(epss_df):,} записів → {EPSS_FILE}")

# ── CELL 5: KEV (пропускається якщо kev.parquet є) ────────────
KEV_FILE = DATA_DIR / "kev.parquet"

if KEV_FILE.exists():
    print(f"SKIP: kev.parquet вже є")
    kev_df = pd.read_parquet(KEV_FILE)
    print(f"Завантажено: {len(kev_df):,} записів")
else:
    print("Завантажуємо CISA KEV...")
    url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    cve_ids = [v["cveID"] for v in data["vulnerabilities"]]
    kev_df = pd.DataFrame({"cve_id": cve_ids, "kev_flag": 1})
    kev_df.to_parquet(KEV_FILE, index=False)
    print(f"Збережено: {len(kev_df):,} записів → {KEV_FILE}")

# ── CELL 6: Merge (пропускається якщо merged.parquet є) ───────
MERGED_FILE = DATA_DIR / "merged.parquet"

if MERGED_FILE.exists():
    print(f"SKIP: merged.parquet вже є")
    merged = pd.read_parquet(MERGED_FILE)
    print(f"Завантажено: {len(merged):,} записів")
else:
    print("Зливаємо джерела...")
    merged = nvd_df.merge(epss_df, on="cve_id", how="inner")
    merged = merged.merge(kev_df, on="cve_id", how="left")
    merged["kev_flag"] = merged["kev_flag"].fillna(0).astype(int)
    merged.to_parquet(MERGED_FILE, index=False)
    print(f"Після merge: {len(merged):,} → {MERGED_FILE}")

# ── CELL 7: Clean (пропускається якщо full_clean.parquet є) ───
CLEAN_FILE = DATA_DIR / "full_clean.parquet"

if CLEAN_FILE.exists():
    print(f"SKIP: full_clean.parquet вже є")
    clean_df = pd.read_parquet(CLEAN_FILE)
    print(f"Завантажено: {len(clean_df):,} записів")
else:
    print("Очищення даних...")
    df = merged.copy()
    df = df[df["description"].notna()]
    df = df[df["description"].str.strip() != ""]
    mask = (
        df["description"].str.contains("** RESERVED **", regex=False, na=False) |
        df["description"].str.contains("DO NOT USE THIS CANDIDATE", regex=False, na=False) |
        df["description"].str.contains("** REJECT **", regex=False, na=False)
    )
    df = df[~mask]
    df = df[df["description"].str.split().str.len() >= 20]
    df = df[df["cvss3"].notna()]
    df["pub_year"] = pd.to_datetime(df["pub_date"], errors="coerce").dt.year
    df = df[df["pub_year"].between(NVD_START_YEAR, NVD_END_YEAR)]
    df = df.reset_index(drop=True)
    clean_df = df
    print(f"Після очищення: {len(clean_df):,}")

# ── CELL 8: Text preprocessing ────────────────────────────────
CVE_PATTERN = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
URL_PATTERN  = re.compile(r"http\S+")

def clean_text_minimal(text):
    text = CVE_PATTERN.sub("", str(text))
    text = URL_PATTERN.sub("", text)
    return text.strip()

def clean_text_full(text):
    text = clean_text_minimal(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

if "text_minimal" not in clean_df.columns:
    clean_df["text_minimal"] = clean_df["description"].apply(clean_text_minimal)
    clean_df["text_full"]    = clean_df["description"].apply(clean_text_full)
    clean_df["high_risk"]    = (clean_df["epss"] >= 0.1).astype(int)
    clean_df["desc_len"]     = clean_df["description"].str.split().str.len()
    clean_df.to_parquet(CLEAN_FILE, index=False)
    print(f"Текст оброблено і збережено → {CLEAN_FILE}")
else:
    print("SKIP: text columns вже є")

print(f"High-risk: {clean_df['high_risk'].sum():,} ({clean_df['high_risk'].mean()*100:.2f}%)")

# ── CELL 9: Train/Val/Test split ──────────────────────────────
TRAIN_FILE = DATA_DIR / "train.parquet"
VAL_FILE   = DATA_DIR / "val.parquet"
TEST_FILE  = DATA_DIR / "test.parquet"

if TRAIN_FILE.exists() and VAL_FILE.exists() and TEST_FILE.exists():
    print("SKIP: train/val/test вже є")
    train_df = pd.read_parquet(TRAIN_FILE)
    val_df   = pd.read_parquet(VAL_FILE)
    test_df  = pd.read_parquet(TEST_FILE)
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
else:
    from sklearn.model_selection import train_test_split

    test_mask    = clean_df["pub_year"] == 2025
    exclude_mask = clean_df["pub_year"] >= 2026
    train_pool   = clean_df[~test_mask & ~exclude_mask].copy()
    test_df      = clean_df[test_mask].copy()

    train_df, val_df = train_test_split(
        train_pool, test_size=0.18,
        stratify=train_pool["high_risk"],
        random_state=RANDOM_STATE)

    train_df.to_parquet(TRAIN_FILE, index=False)
    val_df.to_parquet(VAL_FILE,     index=False)
    test_df.to_parquet(TEST_FILE,   index=False)

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    print(f"High-risk train: {train_df['high_risk'].mean()*100:.1f}%")
    print(f"High-risk test:  {test_df['high_risk'].mean()*100:.1f}%")

# ── CELL 10: EDA Figures ──────────────────────────────────────
FIG3 = DATA_DIR / "fig03_epss_distribution.png"
FIG4 = DATA_DIR / "fig04_cve_by_year.png"
FIG5 = DATA_DIR / "fig05_desc_length.png"

if FIG3.exists() and FIG4.exists() and FIG5.exists():
    print("SKIP: Figure 3, 4, 5 вже є на Drive")
else:
    from scipy import stats as scipy_stats

    # Figure 3
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 3.8),
                                gridspec_kw={"width_ratios": [2, 1]})
    ax3a = axes3[0]
    counts, bins, patches = ax3a.hist(clean_df["epss"], bins=120,
                                       color=C_BLUE, alpha=0.85, edgecolor="none")
    for patch, left in zip(patches, bins[:-1]):
        if left >= 0.1:
            patch.set_facecolor(C_RED)
            patch.set_alpha(0.75)
    ax3a.set_yscale("log")
    ax3a.axvline(x=0.1, color=C_RED, linewidth=0.9, linestyle="--")
    ax3a.text(0.105, ax3a.get_ylim()[1]*0.55, "threshold = 0.1",
              fontsize=8, color=C_RED, va="center")
    ax3a.set_xlabel("EPSS score")
    ax3a.set_ylabel("Number of CVEs (log scale)")
    ax3a.set_xlim(0, 1)
    import matplotlib.patches as mpatches
    patch_low  = mpatches.Patch(color=C_BLUE, alpha=0.85, label="Low risk (EPSS < 0.1)")
    patch_high = mpatches.Patch(color=C_RED,  alpha=0.75, label="High risk (EPSS >= 0.1)")
    ax3a.legend(handles=[patch_low, patch_high], loc="upper right", fontsize=8)
    ax3b = axes3[1]
    high = clean_df["high_risk"].mean() * 100
    low  = 100 - high
    kev  = clean_df["kev_flag"].mean() * 100
    bars = ax3b.barh(["Low risk\n(EPSS < 0.1)", "High risk\n(EPSS >= 0.1)", "CISA KEV\nconfirmed"],
                     [low, high, kev], color=[C_BLUE, C_RED, C_ORANGE],
                     alpha=0.8, edgecolor="none", height=0.5)
    for bar, val in zip(bars, [low, high, kev]):
        ax3b.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                  f"{val:.1f}%", va="center", fontsize=8.5)
    ax3b.set_xlabel("Share of dataset (%)")
    ax3b.set_xlim(0, 105)
    ax3b.spines["left"].set_visible(False)
    ax3b.spines["top"].set_visible(False)
    ax3b.spines["right"].set_visible(False)
    ax3b.tick_params(axis="y", length=0)
    plt.tight_layout(pad=1.2)
    plt.savefig(FIG3, dpi=300)
    plt.savefig(DATA_DIR / "fig03_epss_distribution.pdf")
    plt.show()
    print("Figure 3 saved.")

    # Figure 4
    year_stats = (
        clean_df[clean_df["pub_year"].between(2010, 2026)]
        .groupby("pub_year")
        .agg(total=("cve_id","count"), kev=("kev_flag","sum"))
        .reset_index()
    )
    year_stats["kev_pct"] = year_stats["kev"] / year_stats["total"] * 100
    fig4, ax4a = plt.subplots(figsize=(12, 3.8))
    ax4b = ax4a.twinx()
    years = year_stats["pub_year"].astype(int)
    colors_bar = [C_BLUE if y <= 2024 else C_ORANGE if y == 2025 else "#CCCCCC" for y in years]
    ax4a.bar(years, year_stats["total"], color=colors_bar, alpha=0.78, edgecolor="none", width=0.7)
    ax4b.plot(years, year_stats["kev_pct"], color=C_RED, marker="o", markersize=4, linewidth=1.0)
    ax4a.axvline(x=2015.5, color=C_GREY, linewidth=0.8, linestyle=":")
    ax4a.text(2015.6, ax4a.get_ylim()[1]*0.85, "NVD gap\n2015-2016", fontsize=7.5, color=C_GREY, va="top")
    ax4a.set_xlabel("Year of CVE publication")
    ax4a.set_ylabel("Number of CVEs")
    ax4b.set_ylabel("Share in CISA KEV (%)")
    ax4b.set_ylim(0, max(year_stats["kev_pct"])*2.5)
    ax4a.set_xticks(years)
    ax4a.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    import matplotlib.ticker as ticker
    ax4a.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    patch_train = mpatches.Patch(color=C_BLUE, alpha=0.78, label="Train+Val (2010-2024)")
    patch_test  = mpatches.Patch(color=C_ORANGE, alpha=0.78, label="Test (2025)")
    patch_excl  = mpatches.Patch(color="#CCCCCC", alpha=0.78, label="Excluded (2026)")
    line_kev    = matplotlib.lines.Line2D([], [], color=C_RED, marker="o",
                                           markersize=4, linewidth=1.0, label="CISA KEV share (%)")
    ax4a.legend(handles=[patch_train, patch_test, patch_excl, line_kev], loc="upper left", fontsize=8)
    ax4b.spines["top"].set_visible(False)
    plt.tight_layout(pad=1.0)
    plt.savefig(FIG4, dpi=300)
    plt.savefig(DATA_DIR / "fig04_cve_by_year.pdf")
    plt.show()
    print("Figure 4 saved.")

    # Figure 5
    if "desc_len" not in clean_df.columns:
        clean_df["desc_len"] = clean_df["description"].str.split().str.len()
    fig5, axes5 = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax5a = axes5[0]
    ax5a.hist(clean_df["desc_len"], bins=80, range=(0, 200), color=C_BLUE, alpha=0.85, edgecolor="none")
    med = clean_df["desc_len"].median()
    mn  = clean_df["desc_len"].mean()
    ax5a.axvline(med, color=C_RED, linewidth=0.9, linestyle="--", label=f"Median = {int(med)} words")
    ax5a.axvline(mn,  color=C_ORANGE, linewidth=0.9, linestyle=":", label=f"Mean = {mn:.0f} words")
    ax5a.axvline(20,  color=C_GREY, linewidth=0.9, linestyle="-.", label="Min threshold = 20 words")
    ax5a.set_xlabel("Description length (words)")
    ax5a.set_ylabel("Number of CVEs")
    ax5a.set_xlim(0, 200)
    ax5a.legend(loc="upper right", fontsize=8)
    ax5b = axes5[1]
    data_low  = clean_df.loc[clean_df["high_risk"]==0, "desc_len"].dropna()
    data_high = clean_df.loc[clean_df["high_risk"]==1, "desc_len"].dropna()
    bp = ax5b.boxplot([data_low, data_high], vert=True, patch_artist=True, widths=0.45,
                      medianprops=dict(color="#222222", linewidth=1.0),
                      whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
                      flierprops=dict(marker=".", markersize=2, markerfacecolor=C_GREY,
                                      alpha=0.3, linestyle="none"),
                      boxprops=dict(linewidth=0.8))
    bp["boxes"][0].set_facecolor(C_BLUE + "BB")
    bp["boxes"][1].set_facecolor(C_RED  + "BB")
    ax5b.set_xticks([1, 2])
    ax5b.set_xticklabels(["Low risk\n(EPSS < 0.1)", "High risk\n(EPSS >= 0.1)"], fontsize=8.5)
    ax5b.set_ylabel("Description length (words)")
    ax5b.set_ylim(0, 250)
    ax5b.spines["top"].set_visible(False)
    ax5b.spines["right"].set_visible(False)
    u_stat, p_val = scipy_stats.mannwhitneyu(data_low, data_high, alternative="two-sided")
    sig = f"Mann-Whitney U, p {'< 0.001' if p_val < 0.001 else f'= {p_val:.3f}'}"
    ax5b.text(1.5, 235, sig, ha="center", va="center", fontsize=7.5, color=C_GREY, style="italic")
    plt.tight_layout(pad=1.2)
    plt.savefig(FIG5, dpi=300)
    plt.savefig(DATA_DIR / "fig05_desc_length.pdf")
    plt.show()
    print("Figure 5 saved.")

# ── CELL 11: Dataset Summary ───────────────────────────────────
print("\n=== Dataset Summary ===")
print(f"Total records        : {len(clean_df):,}")
print(f"High-risk (EPSS>=0.1): {clean_df['high_risk'].sum():,} ({clean_df['high_risk'].mean()*100:.2f}%)")
print(f"KEV-flagged          : {clean_df['kev_flag'].sum():,} ({clean_df['kev_flag'].mean()*100:.2f}%)")
print(f"Median EPSS          : {clean_df['epss'].median():.4f}")
print(f"Mean EPSS            : {clean_df['epss'].mean():.4f}")
print(f"Median desc length   : {clean_df['desc_len'].median():.0f} words")
print(f"Year range           : {int(clean_df['pub_year'].min())}–{int(clean_df['pub_year'].max())}")
print(f"\nTop-10 CWE categories:")
print(clean_df["cwe"].value_counts().head(10).to_string())
print(f"\nSplit sizes:")
print(f"  Train : {len(train_df):,}")
print(f"  Val   : {len(val_df):,}")
print(f"  Test  : {len(test_df):,}")

# Перевірка що все збережено
print("\n=== Файли на Drive ===")
for f in sorted(DATA_DIR.iterdir()):
    print(f"  {f.name:<40} {f.stat().st_size/1024:.0f} KB")
