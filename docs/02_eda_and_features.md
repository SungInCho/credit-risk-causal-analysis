# 02 — EDA, Feature Selection, and Feature Engineering

**Notebook**: [`notebooks/02_eda_and_features.ipynb`](../notebooks/02_eda_and_features.ipynb)
**Input**: `data/processed/accepted_cleaned.parquet`
**Outputs**: `data/processed/accepted_modeling.parquet`, `data/processed/meta.json`, `outputs/figures/fig01–fig13_*.png`

---

## Overview

This notebook has three sequential parts. The order is deliberate: EDA findings directly inform which features to select and how to engineer them.

---

## Part A — Exploratory Data Analysis

### Key Plots

| Figure | Key Finding |
|---|---|
| `fig01_grade_int_rate.png` | Strong monotonic relationship: Grade A median ~7%, Grade G median ~28%. Within-grade IQR is 2–4 pp — this within-grade variation is the basis for identification. |
| `fig02_grade_default_rate.png` | Default rate rises from ~5% (A) to ~35% (G). Grade is the dominant credit-quality signal and the main confound for the rate–default relationship. |
| `fig03_int_rate_distribution.png` | Bimodal appearance in the pooled distribution, explained by the 36 vs. 60-month term split. Within-grade distributions are approximately normal. |
| `fig04_int_rate_vs_default_raw.png` | Strong positive raw association (Pearson r ≈ 0.35). This is heavily confounded by grade — a 1 pp higher rate is associated with ~1.5 pp higher default probability in the raw data, far exceeding the causal estimate. |
| `fig05_term_int_rate_default.png` | At the same rate level, 60-month loans default at higher rates than 36-month loans. Term captures independent credit-risk information beyond the rate. |
| `fig06_subgrade_rate_default.png` | Both rate and default rate increase monotonically across A1–G5, with visible jumps at grade boundaries (A5→B1, B5→C1, etc.) — exploited in the boundary analysis of Notebook 04. |
| `fig07_fico_by_grade.png` | FICO distributions are well-separated across grades (A median ~755, G median ~660). FICO and grade are highly collinear (r ≈ −0.70 with grade_num). |
| `fig08_dti_by_default.png` | Defaulted loans have meaningfully higher DTI (Welch t-test p < 0.001). DTI is an important confounder independent of grade. |
| `fig09_purpose_default_rate.png` | Small business loans default at ~30%; debt consolidation loans are lower-risk. Purpose must be controlled. |
| `fig10_vintage_default.png` | Default rates are stable across 2013–2016 vintages within grade, validating the pooling assumption. |
| `fig11_treatment_overlap_preview.png` | Within-grade high-rate vs. low-rate borrowers have near-identical distributions on FICO, DTI, income, and utilization (SMD < 0.1 for most variables). Treatment overlap is good. |
| `fig12_correlation_heatmap.png` | Strongest correlates with default: `int_rate` (r≈0.35), `grade_num` (r≈0.33), `fico_mid` (r≈−0.28), `term_months` (r≈0.22), `dti` (r≈0.15). |
| `fig13_feature_importance.png` | XGBClassifier importance confirms `fico_mid`, `int_rate`, `dti`, `annual_inc_cap`, and `revol_util` as top predictors. |

---

## Part B — Feature Selection

### Selection Pipeline

1. **Variance filter** — drop near-zero-variance features.
2. **Correlation with default** — rank all numeric candidates by |r|.
3. **Pairwise collinearity** — if two features have |r| > 0.85, keep the one with higher default-correlation. (`fico_range_low` and `fico_range_high` collapse to `fico_mid` here.)
4. **XGBClassifier importance** — fit a gradient-boosted tree classifier; keep features with importance ≥ 0.01.
5. **VIF check** — multicollinearity screen. `subgrade_num` is excluded from the final modeling covariate set due to VIF > 50 (VIF = 53.6), reflecting its near-linear relationship with `grade_num` and `int_rate`. It is retained only for the sub-grade boundary analysis in Notebook 04.

### Final Numeric Feature Set

| Feature | Rationale |
|---|---|
| `loan_amnt` | Loan size captures demand signal |
| `term_months_encoded` | 36 months → 0, 60 months → 1 |
| `grade_num` | Ordinal grade encoding (0–6, A→G) |
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
| `issue_year_encoded` | 2013 → 0, 2014 → 1, 2015 → 2, 2016 → 3 |

### Categorical Features (one-hot encoded)

| Feature | Groups | Rationale |
|---|---|---|
| `home_ownership` | MORTGAGE (baseline), OWN, RENT, OTHER | Housing stability proxy |
| `verification_status` | Not Verified (baseline), Source Verified, Verified | Income verification affects default |
| `purpose` | 6 consolidated groups (see below) | Reduces dummy count while preserving default-rate variation |
| `application_type` | Individual (baseline), Joint | Joint applications have different risk profiles |

**Purpose consolidation** (6 groups, with `debt_consolidation` as baseline):

| Group | Original purposes included |
|---|---|
| `debt_consolidation` (baseline) | debt_consolidation, credit_card |
| `home_asset` | home_improvement, house, moving |
| `medical` | medical |
| `other` | other, vacation, wedding, renewable_energy, educational |
| `planned_purchase` | major_purchase, car, home_improvement |
| `small_business` | small_business |

---

## Part C — Feature Engineering

| Engineered Feature | Formula / Encoding | Motivation |
|---|---|---|
| `log_annual_inc` | `log(1 + annual_inc_cap)` | Right-skewed income → log scale improves linearity and model fit |
| `ever_delinq` | `mths_since_last_delinq < 999` | Summarises delinquency history as a single binary flag |
| `grade_num` | Ordinal 0–6 (A→G) | Enables smooth grade trend in regression models |
| `term_months_encoded` | 36 → 0, 60 → 1 | Eliminates scale difference relative to other binary features |
| `issue_year_encoded` | 2013 → 0, …, 2016 → 3 | Controls for vintage effects as an ordinal variable |
| `high_rate` | `int_rate > grade median` | **Primary treatment variable** for IPW/AIPW |
| `high_rate_subgrade` | `int_rate > sub-grade median` | Secondary treatment for boundary analysis |
| `rate_dev` | `int_rate − grade median` | **Continuous treatment** for DML |

### Outcome Variable

```python
default = 1  if  loan_status in {"Charged Off", "Default"}
default = 0  otherwise
```

"Does not meet the credit policy" variants are kept in the sample but are **not** counted as defaults.

### Treatment Variable Design

The primary treatment `high_rate` is defined **within grade**:

```
high_rate_i = 1{int_rate_i > median(int_rate | grade_i)}
```

**Why within grade?** Grade assignment is itself a function of credit quality — comparing across grades conflates the price effect with the quality-selection effect. Within each grade, the rate varies by ~2–3 pp, driven by finer scoring differences and pricing discretion. Conditional on the full observable covariate set, this residual variation is plausibly quasi-random.

---

## Final Modeling Dataset

| Attribute | Value |
|---|---|
| Rows | 1,025,917 |
| Columns | 29 |
| Overall default rate | 20.09% |
| Saved as | `data/processed/accepted_modeling.parquet` |

The 65-row reduction from 1,025,982 (cleaned) to 1,025,917 (modeling) reflects rows dropped when one-hot encoding produced edge-case category combinations.

---

## Saved Outputs

- **`accepted_modeling.parquet`** — final modeling dataset (1,025,917 rows, 29 columns) used by all downstream notebooks.
- **`meta.json`** — records `N=1025917`, `DEFAULT_RATE=0.2009`, lists of `COVARIATES`, `NUMERIC_COLS`, `CAT_COLS`, `GRADE_ORDER`, ensuring consistent column references across notebooks.
