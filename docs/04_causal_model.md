# 04. Causal Models

## Overview

This notebook applies four causal inference methods to estimate the effect of higher interest rates on default, plus a sub-grade boundary analysis for local validation. All methods condition on observable borrower characteristics and exploit within-grade rate variation.

**Input**: `accepted_modeling.parquet` (1,025,917 rows, 24 covariates)
**Treatment**: `high_rate` (binary) / `int_rate` (continuous, for DML)
**Outcome**: `default` (binary)
**Identification assumption**: Conditional on observable borrower characteristics, within-grade rate assignment is as-good-as-random.

---

## Method 1 — Propensity Score Weighting (IPW)

### Propensity Score Estimation

- 5-fold cross-fitted logistic regression (C=0.5)
- PS range: [0.0005, 0.8873], PS AUC = 0.6468
- Clipped to [1st percentile, 99th percentile] = [0.20, 0.76]
- 100% of observations in common support

The propensity score distributions for high-rate and low-rate borrowers show substantial overlap, indicating that treatment assignment is not deterministic given observables. This supports the positivity assumption.

### Stabilized Hajek Estimator

| Estimand | Estimate | 95% CI |
|---|---|---|
| IPW-ATE | **+2.4473 pp** | [2.281, 2.615] (bootstrap, 500 reps) |
| IPW-ATT | **+0.9162 pp** | — |

The IPW-ATE exceeds the naive difference (+1.71 pp), suggesting that risk sorting partially masked the true pricing effect. The smaller IPW-ATT indicates that borrowers who actually received higher pricing are more risk-tolerant on average — consistent with treatment effect heterogeneity.

### Covariate Balance After IPW

| Variable | SMD (Unweighted) | SMD (IPW) |
|---|---|---|
| fico_mid | 0.155 | 0.009 |
| dti | 0.008 | 0.010 |
| log_annual_inc | 0.056 | 0.004 |
| revol_util | 0.132 | 0.004 |
| loan_amnt | 0.030 | 0.008 |
| inq_last_6mths | 0.105 | 0.004 |
| grade_num | 0.037 | 0.030 |

All SMDs reduced well below conventional thresholds after weighting, confirming strong comparability.

---

## Method 2 — Doubly Robust Estimation (AIPW)

AIPW combines a propensity score model (logistic regression) with outcome models (XGBClassifier, n_estimators=200, max_depth=4) under 5-fold cross-fitting. The estimator is **doubly robust**: consistent if either the propensity score or the outcome model is correctly specified.

### Results

| Estimand | Estimate | SE | 95% CI | z | p-value |
|---|---|---|---|---|---|
| **AIPW-ATE** | **+2.0800 pp** | 0.0808 pp | [1.9218, 2.2383] | 25.76 | < 0.001 |
| AIPW-ATT | +2.0353 pp | — | — | — | — |

The similarity between ATE and ATT suggests that the impact of higher interest rates is relatively uniform across the population at the aggregate level.

---

## Method 3 — Double Machine Learning (DML)

### Design

Partially linear model (Robinson 1988; Chernozhukov et al. 2018):
- `Y = theta * T + g(X) + epsilon`, `T = m(X) + v`
- **Continuous treatment**: `int_rate` (not `high_rate`)
- Nuisance models: XGBClassifier for E[Y|X], XGBRegressor for E[T|X]
- 5-fold cross-fitting; heteroskedasticity-robust standard errors

### Results

| Metric | Value |
|---|---|
| theta | **+1.1532 pp** per 1 pp rate increase |
| SE | 0.0358 pp |
| 95% CI | [1.0830, 1.2235] |
| First-stage R-squared | 0.9463 |

After flexibly controlling for borrower risk, each 1 pp increase in interest rate leads to approximately 1.15 pp increase in default probability. The high first-stage R-squared (0.95) means observables explain most rate variation; the remaining unexplained variation is what DML identifies the causal effect on.

---

## Method 4 — Causal Forest (CATE)

### Design

`CausalForestDML` from `econml`:
- model_y: GradientBoostingRegressor (100 estimators, max_depth=4)
- model_t: LogisticRegression (C=0.5)
- 500 trees, min_samples_leaf=50, 5-fold cross-fitting

### Results

| Metric | Value |
|---|---|
| Overall Mean CATE | **+1.8027 pp** |
| Std Dev | 1.03 pp |
| Range | [-3.55 pp, +6.60 pp] |
| Median | +1.83 pp |

### Heterogeneity by Subgroup

**By Grade**: Treatment effects are larger for lower-risk grades (A-C) and smaller for higher-risk grades (D-G). The additional interest burden more directly increases default risk for otherwise safer borrowers, while for already risky borrowers, default is driven by a broader set of underlying factors.

**By Term**: 36-month loans show a larger mean CATE (2.06 pp) than 60-month loans (1.00 pp).

**By DTI and FICO**: Relatively small variation across quantiles, suggesting limited additional heterogeneity along these dimensions beyond what grade captures.

---

## Method 5 — Sub-Grade Boundary Analysis

A quasi-RD approach comparing borrowers just below and above each grade boundary (e.g., A5 vs. B1) where interest rates jump discretely.

| Boundary | N (low) | N (high) | Rate Jump | Default Diff | Wald Estimate | p-value |
|---|---|---|---|---|---|---|
| A5 \| B1 | 49,066 | 54,060 | +0.57 pp | +2.01 pp | 0.035 | < 0.001 |
| B5 \| C1 | 61,310 | 64,688 | +0.73 pp | +2.16 pp | 0.030 | < 0.001 |
| C5 \| D1 | 50,111 | 40,667 | +0.83 pp | +2.05 pp | 0.025 | < 0.001 |
| D5 \| E1 | 22,986 | 19,664 | +0.50 pp | +2.96 pp | 0.059 | < 0.001 |
| E5 \| F1 | 10,265 | 7,828 | +0.89 pp | +0.06 pp | 0.001 | 0.940 |
| F5 \| G1 | 3,002 | 2,071 | +0.51 pp | -0.91 pp | -0.018 | 0.526 |

Discrete rate increases at grade boundaries produce significant default increases for grades A-D, but the effect diminishes and becomes insignificant for E-G. This is consistent with the Causal Forest CATE pattern.

---

## Results Summary

| Method | Treatment | ATE (pp) | 95% CI |
|---|---|---|---|
| Naive difference | high_rate | +1.7073 | N/A |
| IPW-ATE | high_rate | +2.4473 | [2.281, 2.615] |
| **AIPW (doubly robust)** | **high_rate** | **+2.0800** | **[1.922, 2.238]** |
| DML (per 1 pp rate) | int_rate | +1.1532 | [1.083, 1.224] |
| Causal Forest mean CATE | high_rate | +1.8027 | — |

---

## Key Takeaways

1. **All methods find a positive, significant effect** of higher interest rates on default.
2. **Differences between naive and causal estimates** highlight the importance of adjusting for confounding. The naive estimate reflects both rate effects and underlying borrower risk.
3. **Treatment effects exhibit clear heterogeneity across credit grades**, with stronger effects in lower-risk segments and weaker effects in higher-risk segments.
4. **Heterogeneity across DTI and FICO is limited**, indicating that grade captures most of the relevant variation.
5. Interest rates may help screen borrowers at the selection stage, but conditional on loan origination, **higher interest rates increase the probability of default**.
