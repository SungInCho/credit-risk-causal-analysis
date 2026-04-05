# 02. EDA, Feature Selection, and Feature Engineering

## Overview

This notebook has three sequential parts: exploratory data analysis to understand the cleaned data, feature selection to identify a parsimonious covariate set, and feature engineering to build the final modeling dataset. The order is deliberate: EDA findings directly inform which features to select and how to engineer them.

**Input**: `accepted_cleaned.parquet` (1,025,982 rows x 34 columns)
**Output**: `accepted_modeling.parquet` (1,025,917 rows x 29 columns), `meta.json`

---

## Part A — Exploratory Data Analysis

### A1. Raw Relationship: Interest Rate vs. Default Rate

A strong positive relationship exists between interest rate and default rate: Pearson r = 0.2651. However, this is **heavily confounded by grade assignment** — higher-risk borrowers receive both higher rates and default more often.

### A2. Default Rate by Grade

Default rates increase monotonically across grades:

| Grade | Default Rate | Count |
|---|---|---|
| A | 5.9% | 174,351 |
| B | 13.2% | 299,846 |
| C | 22.3% | 291,855 |
| D | 30.6% | 153,985 |
| E | 39.3% | 74,318 |
| F | 46.2% | 25,356 |
| G | 51.2% | 6,271 |

### A3. Interest Rate Distribution by Grade

Each grade has a distinct rate band with limited inter-grade overlap:

| Grade | Median Rate | Std Dev | Min | Max |
|---|---|---|---|---|
| A | 7.26% | 1.00 | 5.32 | 9.25 |
| B | 10.78% | 1.37 | 6.00 | 14.09 |
| C | 13.98% | 1.21 | 6.00 | 17.27 |
| D | 17.57% | 1.36 | 6.00 | 21.49 |
| E | 20.49% | 1.86 | 6.00 | 26.24 |
| F | 24.08% | 1.65 | 6.00 | 30.74 |
| G | 26.06% | 1.67 | 6.00 | 30.99 |

Overall: mean = 13.14%, median = 12.74%.

### A4. Interest Rate vs. Default by Loan Term

| Term | Count | Default Rate | Mean Rate |
|---|---|---|---|
| 36 months | 777,929 | 15.79% | 12.02% |
| 60 months | 248,053 | 33.57% | 16.67% |

60-month loans have substantially higher default rates at the same rate level, indicating term independently predicts default.

### A5. FICO Score Distribution by Grade

FICO medians decline from 717 (Grade A) to 677 (Grade G). Distributions overlap substantially across grades, confirming that FICO must be controlled as a confounder.

### A6. DTI and Default

Defaulting borrowers have significantly higher DTI:
- Non-default mean: 17.96%
- Default mean: 20.33%
- Welch t-test: t = -111.89, p < 0.001

### A7. Default Rate by Purpose

Default rates vary substantially by loan purpose. Small business loans have the highest default rate, while credit card and debt consolidation loans have moderate rates. Purpose must be controlled.

### A8. Vintage Analysis

Default rates increased steadily from 2013 to 2016 across all grades (e.g., Grade A: 5% in 2013 to 7% in 2016; Grade G: 36% in 2013 to 58% in 2016), indicating a time trend that must be controlled.

### A9. Correlation with Default

| Variable | Correlation |
|---|---|
| int_rate | +0.2651 |
| term_months | +0.1899 |
| dti | +0.1119 |
| inq_last_6mths | +0.0670 |
| loan_amnt | +0.0585 |
| revol_util | +0.0510 |
| fico_mid | -0.1274 |
| annual_inc_cap | -0.0659 |

### A10. Within-Grade Treatment Overlap

Standardized mean differences (SMD) between above-median and below-median rate borrowers within grade:

| Covariate | SMD |
|---|---|
| FICO Score | -0.154 |
| Revolving Utilization | +0.132 |
| Inquiries (6 months) | +0.105 |
| Annual Income | -0.058 |
| Loan Amount | +0.030 |
| DTI | +0.008 |

Good overlap (SMD < 0.2 for all covariates) confirms that propensity score approaches are credible.

### EDA Summary

1. **Grade drives most of the rate-default correlation** — within-grade variation is the basis for causal identification.
2. **Term independently predicts default** at the same rate level (60-month loans default more).
3. **FICO and DTI are the strongest credit-quality signals** after grade; high correlation with both rate and default — must be controlled.
4. **Purpose matters** (small business >> debt consolidation in default rate) — include as control.
5. **Between 2013 and 2016, default rates increased steadily** across all credit grades.
6. **Within-grade overlap is good** (SMD < 0.2 for most covariates) — propensity score approach is credible.

---

## Part B — Feature Selection

### Step 1: Variance Filter

All 13 numeric candidate features have variance > 0.01. None dropped.

### Step 2: Correlation with Default

Features ranked by absolute correlation with default. Top features: `fico_mid` (0.127), `dti` (0.112), `inq_last_6mths` (0.067), `annual_inc_cap` (0.066).

### Step 3: Pairwise Collinearity Check

Highest pairwise correlation: `open_acc` and `total_acc` (r = 0.70). Both retained as they capture complementary information (current credit exposure vs. credit history depth). No pairs exceed the 0.85 threshold for exclusion.

### Step 4: Gradient-Boosted Feature Importance

A GBM (100 estimators, max_depth=3) ranked features by gain:

| Feature | Importance |
|---|---|
| fico_mid | 0.2845 |
| loan_amnt | 0.2229 |
| dti | 0.1968 |
| revol_bal | 0.0837 |
| annual_inc_cap | 0.0760 |
| inq_last_6mths | 0.0603 |
| total_acc | 0.0242 |
| emp_length_num | 0.0213 |
| open_acc | 0.0182 |

### Step 5: Final Numeric Feature Set

Keep features with importance >= 0.01 AND |r(default)| >= 0.02.

**Selected (10)**: `fico_mid`, `loan_amnt`, `dti`, `revol_bal`, `annual_inc_cap`, `inq_last_6mths`, `total_acc`, `emp_length_num`, `open_acc`, `revol_util`

**Excluded (3)**: `delinq_2yrs`, `mths_since_last_delinq`, `pub_rec` (low importance and weak correlation)

### Step 6: Categorical Features

| Type | Features |
|---|---|
| Nominal (one-hot) | `home_ownership`, `verification_status`, `purpose` |
| Ordinal | `term_months`, `issue_year` |
| Boolean | `ever_delinq` |

Purpose categories consolidated into 5 groups: debt (consolidation + credit card), home_asset, planned_purchase, small_business, medical, and other.

---

## Part C — Feature Engineering

### C1. Log-Transform Annual Income

`log_annual_inc = log(1 + annual_inc_cap)` — mean = 11.084, std = 0.521.

### C2. One-Hot Encoding

Nominal features encoded with `drop_first=True`, producing 9 dummy columns. Rows with `home_ownership = 'ANY'` dropped (n=65).

### C3. Ordinal Encoding

- `term_months`: 36 -> 0, 60 -> 1
- `issue_year`: 2013 -> 0, 2014 -> 1, 2015 -> 2, 2016 -> 3

### C4. Grade and Sub-Grade Encoding

- `grade_num`: A=0, B=1, ..., G=6
- `subgrade_num`: A1=0, A2=1, ..., G5=34

### C5. Treatment Variables

| Treatment | Definition | Purpose |
|---|---|---|
| `high_rate` | 1 if `int_rate > median(int_rate` within grade) | Primary binary treatment (IPW/AIPW) |
| `high_rate_subgrade` | 1 if `int_rate > median(int_rate` within sub-grade) | Boundary analysis |
| `rate_dev` | `int_rate - grade_median_rate` | Continuous treatment (DML) |

Overall treated fraction: 46.94%.

### C6. Final Covariate List (24 variables)

- **Numeric (10)**: `fico_mid`, `dti`, `inq_last_6mths`, `log_annual_inc`, `loan_amnt`, `revol_util`, `revol_bal`, `open_acc`, `total_acc`, `emp_length_num`
- **Categorical (9)**: `home_ownership_OWN`, `home_ownership_RENT`, `verification_status_Source Verified`, `verification_status_Verified`, `purpose_home_asset`, `purpose_medical`, `purpose_other`, `purpose_planned_purchase`, `purpose_small_business`
- **Ordinal (2)**: `term_months_encoded`, `issue_year_encoded`
- **Boolean (1)**: `ever_delinq`
- **Grade (2)**: `grade_num`, `subgrade_num`

### C7. StandardScaler

All numeric features standardized (mean=0, std=1) before saving.

---

## Output

| Attribute | Value |
|---|---|
| Rows | 1,025,917 |
| Columns | 29 |
| Default rate | 20.09% |
| Treated fraction | 46.94% |
| File | `accepted_modeling.parquet` |
| Metadata | `meta.json` |

The 65-row reduction from 1,025,982 to 1,025,917 reflects rows dropped when removing `home_ownership = 'ANY'`.
