# 03. Baseline Models

## Overview

This notebook establishes naive and progressively controlled estimates of the interest rate-default relationship, quantifying how much endogeneity (omitted-variable bias) inflates the naive estimate. The key question: how much does the estimated effect shrink as controls are added?

**Input**: `accepted_modeling.parquet` (1,025,917 rows, 24 covariates)

---

## Logistic Regression Models

### Model 1 — Naive: `default ~ int_rate`

No controls. Captures the raw association.

| Metric | Value |
|---|---|
| AME(int_rate) | **+2.0825 pp** per 1 pp increase |
| Pseudo R-squared | 0.0678 |
| int_rate coefficient | 0.1400 (z = 255.7, p < 0.001) |

Interpretation: a 1 pp increase in interest rate is associated with a 2.08 pp increase in P(default) — but this is **not causal**, as it conflates rate effects with underlying borrower risk.

### Model 2 — Grade-Controlled: `default ~ int_rate + grade`

Adding grade as a single control dramatically reduces the estimated effect.

| Metric | Value |
|---|---|
| AME(int_rate) | **+0.8141 pp** |
| Pseudo R-squared | 0.0700 |
| Shrinkage from Model 1 | -60.9% |

Grade alone explains most of the naive association.

### Model 3 — Full Controls: `default ~ int_rate + X` (23 covariates)

All covariates except `subgrade_num` (excluded due to high VIF with `grade_num` and `int_rate`).

| Metric | Value |
|---|---|
| AME(int_rate) | **+0.8433 pp** |
| Pseudo R-squared | 0.0988 |
| Shrinkage from Model 1 | **-59.5%** |

The slightly larger AME compared to Model 2 may reflect a suppressor effect, where additional covariates remove opposing variation correlated with interest rates.

### Model 4 — Binary Treatment: `default ~ high_rate + X` (23 covariates)

Using the within-grade binary treatment indicator instead of continuous interest rate.

| Metric | Value |
|---|---|
| AME(high_rate) | **+1.9328 pp** |
| Pseudo R-squared | 0.0985 |

The binary treatment removes cross-grade variation and compares borrowers within the same risk bucket, isolating the effect of relatively higher pricing.

---

## Endogeneity Summary

| Model | Treatment | AME (pp) | Pseudo R-squared | N Controls |
|---|---|---|---|---|
| 1. Naive | int_rate | +2.0825 | 0.0678 | 0 |
| 2. + Grade | int_rate | +0.8141 | 0.0700 | 1 |
| 3. + Full controls | int_rate | +0.8433 | 0.0988 | 23 |
| 4. Binary + Full controls | high_rate | +1.9328 | 0.0985 | 23 |

**AME shrinkage (Model 1 to Model 3): -59.5%** — the naive estimate severely overstates the causal effect.

---

## Frisch-Waugh-Lovell Decomposition

The FWL theorem shows that the treatment coefficient with controls equals the coefficient from regressing residualized outcome on residualized treatment (both purged of controls).

| Metric | Value |
|---|---|
| OLS coefficient (naive) | 0.017073 |
| OLS coefficient (residualized FWL) | 0.005806 |
| Explained by controls | **66.0%** |

Observable controls account for **66% of the naive coefficient**. The remaining 34% is the residual rate-default association that motivates the causal methods in Notebook 04.

---

## Within-Grade Naive Estimates

Separate within-grade regressions (`default ~ int_rate` for each grade):

| Grade | N | Default Rate | Mean Rate | Rate Std | AME (pp) | p-value |
|---|---|---|---|---|---|---|
| A | 174,341 | 5.9% | 7.10% | 1.00 | +1.690 | < 0.001 |
| B | 299,820 | 13.2% | 10.58% | 1.37 | +0.767 | < 0.001 |
| C | 291,840 | 22.3% | 13.91% | 1.21 | +0.678 | < 0.001 |
| D | 153,978 | 30.6% | 17.48% | 1.36 | +0.906 | < 0.001 |
| E | 74,312 | 39.3% | 20.62% | 1.86 | +0.421 | < 0.001 |
| F | 25,356 | 46.2% | 24.43% | 1.65 | +1.914 | < 0.001 |
| G | 6,270 | 51.2% | 27.04% | 1.67 | +3.612 | < 0.001 |

**Key findings**: Rate standard deviation within each grade is 1-2 pp — sufficient for causal identification. The within-grade rate-default association is positive and significant across all grades.

---

## Cross-Validated AUC

| Model | 5-Fold CV AUC |
|---|---|
| Default ~ X only | 0.7172 |
| Default ~ high_rate + X | 0.7181 |
| **Gain from adding high_rate** | **+0.0009** |

Near-zero AUC gain confirms that the within-grade treatment indicator adds virtually no predictive power beyond observable borrower characteristics — supporting the conditional unconfoundedness assumption.

---

## Summary

- The naive logistic AME substantially overstates the causal effect. Adding grade alone causes ~61% shrinkage; full controls cause ~60% shrinkage.
- The FWL decomposition shows that 66% of the raw coefficient is absorbed by observables.
- Residual bias likely remains (lenders observe soft data, proprietary scores) — motivating propensity score and DML methods in Notebook 04.
- Within-grade rate variation of 1-2 pp is sufficient for identification, and the raw rate-default association is positive and significant across all grades.
