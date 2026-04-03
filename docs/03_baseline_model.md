# 03 — Baseline Models

**Notebook**: [`notebooks/03_baseline_model.ipynb`](../notebooks/03_baseline_model.ipynb)
**Input**: `data/processed/accepted_modeling.parquet`, `data/processed/meta.json`
**Outputs**: `outputs/figures/fig14–fig16_*.png`

---

## Purpose

Before applying causal methods, we establish a **progression of logistic regression models** — from naive to fully controlled — to:

1. Quantify the raw rate–default association.
2. Show how much the coefficient shrinks as controls are added.
3. Demonstrate that the naive estimate is substantially upward-biased by omitted variables.

This motivates the propensity score and DML approaches in Notebook 04.

---

## VIF Check

Before fitting any model, a Variance Inflation Factor (VIF) screen is performed on all candidate covariates. `subgrade_num` is found to have VIF = 53.6, indicating extreme multicollinearity with `grade_num` and `int_rate`. It is excluded from all regression models in this notebook. The final model uses **23 covariates**.

---

## Models Estimated

### Model 1 — Naive: `default ~ int_rate`

No controls whatsoever. The coefficient captures both the causal effect of the rate *and* all correlations between rate and omitted credit-quality variables.

| Metric | Value |
|---|---|
| AME of `int_rate` | **+2.0825 pp** per 1 pp increase |
| Pseudo R² (McFadden) | 0.0678 |

---

### Model 2 — Grade-Controlled: `default ~ int_rate + grade_num`

Add `grade_num` as a single ordinal control. Grade is the single largest confound: lenders assign higher rates to riskier grades, and riskier grades default more. Conditioning on grade removes this first-order endogeneity.

| Metric | Value |
|---|---|
| AME of `int_rate` | **+0.8141 pp** per 1 pp increase |
| Pseudo R² (McFadden) | 0.0700 |
| Shrinkage from Model 1 | ~61% |

---

### Model 3 — Full Controls: `default ~ int_rate + X` (23 covariates)

Add all 23 selected covariates from Notebook 02: FICO, DTI, income, utilization, employment, purpose dummies, home ownership, verification status, term, vintage, etc.

| Metric | Value |
|---|---|
| AME of `int_rate` | **+0.8484 pp** per 1 pp increase |
| Pseudo R² (McFadden) | 0.0988 |
| Strongest positive predictor | `term_months_encoded` (coef +0.6287) |
| Strongest negative predictor | `fico_mid` (coef −0.2153) |
| Other notable predictor | `home_ownership_RENT` (coef +0.2919) |

The small increase in AME from Model 2 to Model 3 (+0.03 pp) suggests grade captures the vast majority of the confounding; additional controls add predictive power but barely move the rate coefficient.

---

### Model 4 — Binary Treatment: `default ~ high_rate + X` (23 covariates)

Replace continuous `int_rate` with the binary within-grade treatment indicator `high_rate`, while keeping the full covariate set. This is the regression-adjusted analogue of the IPW/AIPW estimand in Notebook 04.

| Metric | Value |
|---|---|
| AME of `high_rate` | **+1.9328 pp** |
| Pseudo R² (McFadden) | 0.0985 |

---

## Summary of Model Progression

| Model | Treatment | AME (pp) | Pseudo R² | Interpretation |
|---|---|---|---|---|
| 1. Naive | `int_rate` (continuous) | **+2.0825** | 0.0678 | Raw association — severely biased |
| 2. + Grade control | `int_rate` (continuous) | **+0.8141** | 0.0700 | Grade confound removed (~61% shrinkage) |
| 3. + Full controls | `int_rate` (continuous) | **+0.8484** | 0.0988 | Best observable-control regression estimate |
| 4. Binary + controls | `high_rate` (binary) | **+1.9328** | 0.0985 | Pre-cursor to IPW/AIPW |

---

## Frisch–Waugh–Lovell (FWL) Decomposition

The FWL theorem states that the OLS coefficient on `T` in `Y ~ T + X` equals the coefficient from regressing *residualised* `Y` on *residualised* `T` (both purged of `X`).

This cleanly decomposes how much of the raw `int_rate` coefficient is absorbed by observable controls:

| Quantity | Value |
|---|---|
| Naive OLS coefficient | 0.017073 |
| Residualised OLS coefficient (after partialling out X) | 0.005806 |
| Share explained by observable controls | **~66%** |

The remaining 34% is the residual rate–default association after removing all observable confounders. This is what the causal methods in Notebook 04 identify on.

---

## Within-Grade Naive Estimates

`default ~ int_rate` is run **separately within each grade** to confirm that:
- The within-grade rate–default association is positive and statistically significant across all 7 grades.
- The within-grade rate standard deviation is 1–3 pp — sufficient variation for identification.
- Within-grade estimates are substantially smaller than the pooled naive estimate, consistent with grade-level confounding.

---

## Cross-Validated AUC

A 5-fold CV comparison checks how much predictive power `high_rate` adds beyond the full covariate set:

| Model | CV AUC |
|---|---|
| Full controls (Model 3) | Baseline |
| Full controls + `high_rate` (Model 4) | Baseline + **0.0009** |

The near-zero AUC gain confirms that the within-grade rate indicator adds virtually no *predictive* information beyond observable borrower characteristics — consistent with the conditional independence assumption underpinning the causal analysis.

---

## Key Takeaway

> The naive regression overstates the causal effect of interest rate on default by approximately 66%. Controlling for grade removes the single largest confound (~61% shrinkage). Adding the full observable covariate set absorbs a further ~5%, leaving an AME of ~0.85 pp per 1 pp of rate. The remaining estimate is still likely upward-biased due to unobservable lender information — motivating the doubly robust and DML approaches in Notebook 04.
