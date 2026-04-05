# Credit Risk Causal Analysis
### Does a Higher Interest Rate Cause Default? — A Causal Inference Study on Lending Club Loans

---

## Research Questions

1. **Main**: Among observationally similar approved Lending Club borrowers, how much does a higher interest rate *increase* the probability of default?
2. **Extension**: How does risk-based pricing affect default, and for which borrowers is this effect largest?

---

## Key Finding

Being assigned an **above-median interest rate within one's grade** increases the probability of default by approximately **2 percentage points** (AIPW ATE = +2.08 pp, 95% CI [1.92, 2.24]), after controlling for observable borrower characteristics. This effect is largest for lower-risk grades (A-C) and shorter-term (36-month) loans.

The naive (unadjusted) rate-default correlation overstates the causal effect by approximately **60%** due to grade-level confounding — the core endogeneity problem this project addresses.

---

## Project Structure

```
credit-risk-causal-analysis/
│
├── README.md                        ← this file
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
├── data/
│   ├── README.md                    ← data download instructions
│   └── raw/                         ← place raw CSV files here (git-ignored)
│       └── .gitkeep
│
├── notebooks/                       ← analysis notebooks (run in order)
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_and_features.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_causal_model.ipynb
│   └── 05_robustness.ipynb
│
├── src/                             ← reusable Python modules
│   ├── __init__.py
│   ├── cleaning.py                  ← data cleaning helpers
│   ├── features.py                  ← feature engineering
│   ├── estimators.py                ← AIPW, IPW, DML estimators
│   └── plotting.py                  ← shared visualisation utilities
│
├── docs/                            ← plain-English write-ups per notebook
│   ├── 01_data_cleaning.md
│   ├── 02_eda_and_features.md
│   ├── 03_baseline_model.md
│   ├── 04_causal_model.md
│   ├── 05_robustness.md
│   └── final_summary.md
│
├── outputs/
│   ├── figures/                     ← all saved .png plots
│   ├── tables/                      ← summary statistics, result tables
│   └── intermediate/                ← cleaned parquet files & meta.json
│
└── presentations/
    └── causal_inference_analysis_summary.md
```

---

## Notebooks at a Glance

| # | Notebook | Description |
|---|---|---|
| 01 | `01_data_cleaning.ipynb` | Date filtering (2013–2016), type conversion, missing value & outlier handling |
| 02 | `02_eda_and_features.ipynb` | EDA → feature selection → feature engineering → save modeling dataset |
| 03 | `03_baseline_model.ipynb` | Naive to fully-controlled logistic regression; quantify endogeneity |
| 04 | `04_causal_model.ipynb` | IPW, AIPW, DML, Causal Forest, sub-grade boundary analysis |
| 05 | `05_robustness.ipynb` | 5 robustness checks + selection bias appendix (approved vs. rejected) |

---

## Identification Strategy

**Treatment**: `high_rate` — a loan is treated if its interest rate exceeds the **grade-level median**.

**Why within-grade?**
Grade assignment captures the broad credit-quality confound. Conditional on grade and observable borrower characteristics (FICO, DTI, income, etc.), the residual within-grade rate variation is plausibly quasi-random — driven by pricing discretion and minor scoring differences rather than hard credit-quality differences.

**Causal methods used**:

| Method | Treatment type | Key assumption |
|---|---|---|
| IPW (Horvitz-Thompson) | Binary `high_rate` | Propensity score correctly specified |
| AIPW (doubly robust) | Binary `high_rate` | Either PS model or outcome model correct |
| DML (partially linear) | Continuous `int_rate` | Partially linear structural equation |
| Causal Forest (CATE) | Binary `high_rate` | Conditional unconfoundedness |
| Sub-grade boundary Wald | Continuous `int_rate` | Rate jump at grade boundary is exogenous |

---

## Robustness Checks

1. **36-month loans only** — cleaner maturity horizon (ATE = +2.26 pp)
2. **2014-2015 vintage only** — homogeneous market conditions (ATE = +2.04 pp)
3. **Purpose subsamples** — debt-related vs. others (ATE = +2.14 / +1.74 pp)
4. **High-rate threshold sensitivity** — 25th to 75th percentile cutoffs (monotone decline)

---

## Selection Bias Caveat

All causal estimates apply to the **approved Lending Club sample**. Rejected applicants have substantially lower FICO scores (~670 vs. ~720) and higher DTI (~23% vs. ~18%). An approval model achieves AUC = 0.9215, confirming that observable characteristics nearly perfectly determine approval. The population-level effect of risk-based pricing is likely **larger** than within-sample estimates suggest.

---

## Getting Started

### 1. Clone and set up environment

```bash
git clone https://github.com/<your-username>/credit-risk-causal-analysis.git
cd credit-risk-causal-analysis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download data

See [`data/README.md`](data/README.md) for download instructions. Place the two CSV files in `data/raw/`.

### 3. Run notebooks in order

```bash
jupyter lab
```

Open notebooks in the `notebooks/` directory and run them sequentially: `01` → `02` → `03` → `04` → `05`.

Each notebook reads its input from `outputs/intermediate/` and writes outputs back there or to `outputs/figures/`.

---

## Data Sources

| File | Source | Size |
|---|---|---|
| `accepted_2007_to_2018Q4.csv` | [Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | ~1.6 GB |
| `rejected_2007_to_2018Q4.csv` | Same Kaggle dataset | ~1.7 GB |

---

## Dependencies

See [`requirements.txt`](requirements.txt). Core packages:

- `pandas`, `numpy`, `pyarrow` — data wrangling
- `scikit-learn`, `statsmodels`, `scipy` — modelling & statistics
- `econml` — `CausalForestDML` for heterogeneous treatment effects
- `xgboost` — nuisance estimators for AIPW and DML
- `matplotlib`, `seaborn` — visualisation

---

## License

This project is released under the [MIT License](LICENSE).
