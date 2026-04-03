# 03 — Baseline Models

**Notebook**: [`notebooks/03_baseline_model.ipynb`](../notebooks/03_baseline_model.ipynb)
**Input**: `outputs/intermediate/accepted_modeling.parquet`, `outputs/intermediate/meta.json`
**Outputs**: `outputs/figures/fig14–fig16_*.png`

---

## Purpose

Before applying causal methods, we establish a **progression of logistic regression models** — from naive to fully controlled — to:

1. Quantify the raw rate–default association.
2. Show how much the coefficient shrinks as controls are added.
3. Demonstrate that the naive estimate is substantially upward-biased by omitted variables.

This motivates the propensity score and DML approaches in Notebook 04.

---

## Models Estimated

### Model 1 — Naive: `default ~ int_rate`

No controls whatsoever. The coefficient captures both the causal effect of the rate *and* all correlations between rate and omitted credit-quality variables.

**Expected result**: The average marginal effect (AME) will be large and positive, far exceeding the causal estimate.

---

### Model 2 — Grade-Controlled: `default ~ int_rate + grade`

Add grade fixed effects (one-hot dummies). Grade is the single largest confound: lenders assign higher rates to riskier grades, and riskier grades default more. Conditioning on grade removes this first-order endogeneity.

**Expected result**: AME shrinks substantially from Model 1 — the bulk of the raw association is explained by grade.

---

### Model 3 — Full Controls: `default ~ int_rate + X`

Add all 40+ covariates from Notebook 02: FICO, DTI, income, utilization, employment, purpose, home ownership, verification status, issue year, etc.

**Expected result**: AME shrinks further. Any remaining bias reflects unobservable lender information (soft data, proprietary scores) not captured in the public dataset.

---

### Model 4 — Binary Treatment: `default ~ high_rate + X`

Replace continuous `int_rate` with the binary within-grade treatment indicator `high_rate`, while keeping the full covariate set. This is the controlled analogue of the IPW/AIPW estimand in Notebook 04.

**Expected result**: AME gives the regression-adjusted difference in default probability between above-median-rate and below-median-rate borrowers within the same grade.

---

## Key Metrics

| Model | AME of treatment (pp) | Pseudo R² | Interpretation |
|---|---|---|---|
| 1. Naive | Largest | Lowest | Raw association — severely biased |
| 2. + Grade FE | Smaller | Higher | Grade confound removed |
| 3. + Full controls | Smallest | Highest | Best observable-control estimate |
| 4. Binary `high_rate` + X | Similar to Model 3 | Similar | Pre-cursor to causal methods |

The percentage shrinkage from Model 1 to Model 3 quantifies **how much of the naive association is attributable to endogeneity** rather than a causal effect.

---

## Frisch–Waugh–Lovell (FWL) Decomposition

The FWL theorem states that the OLS coefficient on `T` in `Y ~ T + X` equals the coefficient from regressing *residualised* `Y` on *residualised* `T` (both purged of `X`).

We use this to cleanly decompose:
- **How much** of the raw `int_rate` coefficient is absorbed by the observable controls.
- **How much** residual variation remains after controls — this is what the causal methods in Notebook 04 identify on.

```
rate_resid  = int_rate − E[int_rate | X]
default_resid = default − E[default | X]
β_FWL = Cov(rate_resid, default_resid) / Var(rate_resid)
```

---

## Within-Grade Naive Estimates

We also run `default ~ int_rate` **separately within each grade** to confirm that:
- The within-grade rate–default association is positive and statistically significant across all 7 grades.
- The within-grade rate standard deviation is 1–3 pp — sufficient variation for identification.
- The within-grade effect is smaller than the pooled naive effect, consistent with grade-level confounding.

---

## Cross-Validated AUC

A 5-fold CV comparison of Model 1 (naive) vs. Model 3 (full controls) on ROC-AUC:
- A large gain in AUC from adding controls confirms that the controls capture substantial default-predictive variance previously attributed to `int_rate` alone.

---

## Key Takeaway

> The naive regression substantially overstates the causal effect of interest rate on default. Controlling for grade removes the largest confound; adding full observable controls reduces the estimate further. The remaining estimate is still likely upward-biased due to unobservable lender information — motivating the propensity score and DML approaches in Notebook 04.
