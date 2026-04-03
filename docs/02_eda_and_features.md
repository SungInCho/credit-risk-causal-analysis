# 02 — EDA, Feature Selection, and Feature Engineering

**Notebook**: [`notebooks/02_eda_and_features.ipynb`](../notebooks/02_eda_and_features.ipynb)
**Input**: `outputs/intermediate/accepted_cleaned.parquet`
**Outputs**: `outputs/intermediate/accepted_modeling.parquet`, `outputs/intermediate/meta.json`, `outputs/figures/fig01–fig13_*.png`

---

## Overview

This notebook has three sequential parts. The order is deliberate: EDA findings directly inform which features to select and how to engineer them.

---

## Part A — Exploratory Data Analysis

### Required Plots

| Figure | Key Finding |
|---|---|
| `fig01_grade_int_rate.png` | Strong monotonic relationship: Grade A median ~7%, Grade G median ~28%. Within-grade IQR is 2–4 pp — this within-grade variation is the basis for identification. |
| `fig02_grade_default_rate.png` | Default rate rises from ~5% (A) to ~35% (G). Grade is the dominant credit-quality signal and the main confound for the rate–default relationship. |
| `fig03_int_rate_distribution.png` | Bimodal appearance in the pooled distribution, explained by the 36 vs 60-month term split. Within-grade distributions are unimodal and approximately normal. |
| `fig04_int_rate_vs_default_raw.png` | Strong positive raw association (Pearson r ≈ 0.35). This is heavily confounded by grade — a 1 pp higher rate is associated with ~1.5 pp higher default probability in the raw data, far exceeding the causal estimate. |
| `fig05_term_int_rate_default.png` | At the same rate level, 60-month loans default at higher rates than 36-month loans. Term captures independent credit-risk information beyond the rate. |

### Additional Plots

| Figure | Key Finding |
|---|---|
| `fig06_subgrade_rate_default.png` | Both rate and default rate increase monotonically across A1–G5, with visible jumps at grade boundaries (A5→B1, B5→C1, etc.) — exploited in the boundary analysis of Notebook 04. |
| `fig07_fico_by_grade.png` | FICO distributions are well-separated across grades (A median ~755, G median ~660). FICO and grade are highly collinear (r ≈ −0.70 with grade_num). |
| `fig08_dti_by_default.png` | Defaulted loans have meaningfully higher DTI (Welch t-test p < 0.001). DTI is an important confounder independent of grade. |
| `fig09_purpose_default_rate.png` | Small business loans default at ~30%; wedding and vacation loans also default more. Debt consolidation and home improvement are lower-risk. Purpose should be controlled. |
| `fig10_vintage_default.png` | Default rates are stable across 2013–2016 vintages within grade, validating the pooling assumption. |
| `fig11_treatment_overlap_preview.png` | Within-grade high-rate vs. low-rate borrowers have near-identical distributions on FICO, DTI, income, and utilization (SMD < 0.1 for most variables). Treatment overlap is good. |
| `fig12_correlation_heatmap.png` | Strongest correlates with default: `int_rate` (r≈0.35), `grade_num` (r≈0.33), `fico_mid` (r≈−0.28), `term_months` (r≈0.22), `dti` (r≈0.15). |
| `fig13_feature_importance.png` | GBM importance confirms `fico_mid`, `int_rate`, `dti`, `annual_inc_cap`, and `revol_util` as top predictors. |

---

## Part B — Feature Selection

### Selection Pipeline

1. **Variance filter** — drop near-zero-variance features (< 0.01 variance after cleaning).
2. **Correlation with default** — rank all numeric candidates by |r|.
3. **Pairwise collinearity** — if two features have |r| > 0.85, keep the one with higher default-correlation. (In practice, `fico_range_low` and `fico_range_high` collapse to `fico_mid` here.)
4. **GBM importance** — fit a gradient-boosted classifier, keep features with importance ≥ 0.01.
5. **Domain override** — `grade_num` and `subgrade_num` are always retained regardless of rank, because within-grade identification requires fine grade controls.

### Final Numeric Feature Set (16 variables)

| Feature | Rationale |
|---|---|
| `loan_amnt` | Loan size captures demand signal |
| `term_months` | 36 vs 60 months independently predicts default |
| `grade_num` | Ordinal grade encoding (0–6) for smooth trend |
| `subgrade_num` | Fine-grained tier (0–34), used in boundary analysis |
| `fico_mid` | Strongest credit-quality signal after grade |
| `dti` | Debt burden relative to income |
| `log_annual_inc` | Log-transformed income (engineered in Part C) |
| `revol_util` | Revolving credit utilisation |
| `open_acc` | Number of open credit lines |
| `total_acc` | Total credit accounts |
| `pub_rec` | Public derogatory records |
| `delinq_2yrs` | Delinquencies in past 2 years |
| `inq_last_6mths` | Hard credit inquiries (recent credit-seeking behavior) |
| `ever_delinq` | Binary: ever had a delinquency (engineered) |
| `emp_length_num` | Employment tenure |
| `issue_year` | Vintage fixed effect |

### Categorical Features (one-hot encoded)

| Feature | Levels kept | Rationale |
|---|---|---|
| `home_ownership` | RENT, OWN, MORTGAGE, OTHER | Housing stability proxy |
| `verification_status` | Source Verified, Verified, Not Verified | Income verification affects default |
| `purpose` | 14 purposes | Large between-group default variation (EDA A9) |
| `application_type` | Individual, Joint | Joint applications have different risk profiles |

---

## Part C — Feature Engineering

| Engineered Feature | Formula | Motivation |
|---|---|---|
| `log_annual_inc` | `log(1 + annual_inc_cap)` | Right-skewed income → log scale improves linearity and model fit |
| `ever_delinq` | `mths_since_last_delinq < 999` | Summarises delinquency history as a single binary flag |
| `grade_num` | Ordinal 0–6 | Enables smooth grade trend in regression models |
| `subgrade_num` | Ordinal 0–34 | Fine-grained credit tier for boundary analysis |
| `high_rate` | `int_rate > grade median` | **Primary treatment variable** for IPW/AIPW |
| `high_rate_subgrade` | `int_rate > sub-grade median` | Secondary treatment for boundary analysis |
| `rate_dev` | `int_rate − grade median` | **Continuous treatment** for DML |

### Treatment Variable Design

The primary treatment `high_rate` is defined **within grade**:

```
high_rate_i = 1{int_rate_i > median(int_rate | grade_i)}
```

**Why within grade?**
Grade assignment is itself a function of credit quality — comparing across grades would conflate the price effect with the quality-selection effect. Within each grade, the rate varies by ~2–3 pp, driven by finer scoring differences and pricing discretion. Conditional on the full observable covariate set, this residual variation is plausibly quasi-random.

---

## Saved Outputs

- **`accepted_modeling.parquet`** — final modeling dataset (~700 K rows, ~50 columns) used by all downstream notebooks.
- **`meta.json`** — lists of `COVARIATES`, `NUMERIC_FINAL`, `CAT_COLS`, `GRADE_ORDER`, and `DEFAULT_STATUSES`, ensuring consistent column references across notebooks.
