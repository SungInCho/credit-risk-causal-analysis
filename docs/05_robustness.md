# 05. Robustness Checks and Selection Bias Appendix

## Overview

This notebook verifies that the AIPW ATE from Notebook 04 is stable across different sample restrictions and specification choices, and characterizes the selection process that produces the approved-loan sample.

**Input**: `accepted_modeling.parquet`, `accepted_with_current.parquet`, `rejected_cleaned.parquet`

---

## Baseline Reference

| Metric | Value |
|---|---|
| N | 1,025,917 |
| AIPW ATE | +2.0836 pp |
| 95% CI | [1.9254, 2.2419] |

---

## Robustness Check 1 — 36-Month Loans Only

Restricting to 36-month loans provides a cleaner maturity horizon with less censoring risk.

| Metric | Value |
|---|---|
| N | 777,875 |
| Default Rate | 15.79% |
| AIPW ATE | **+2.2608 pp** |
| 95% CI | [2.0899, 2.4317] |

Slightly larger than baseline, consistent with the Causal Forest finding of larger treatment effects for shorter-term loans.

---

## Robustness Check 2 — 2014-2015 Vintage Only

Restricting to a narrower issuance window ensures homogeneous market conditions.

| Metric | Value |
|---|---|
| N | 598,375 |
| Default Rate | 19.54% |
| AIPW ATE | **+2.0372 pp** |
| 95% CI | [1.7941, 2.2804] |

Nearly identical to the baseline estimate.

---

## Robustness Check 3 — Purpose Subsamples

| Subsample | N | Default Rate | AIPW ATE | 95% CI |
|---|---|---|---|---|
| Debt-related purposes | 842,797 | 20.09% | **+2.1417 pp** | [1.966, 2.317] |
| Other purposes | 183,120 | 20.11% | **+1.7391 pp** | [1.373, 2.106] |

The effect is slightly weaker for non-debt purposes but remains positive and significant across both groups.

---

## Robustness Check 4 — High-Rate Threshold Sensitivity

The treatment definition is varied from the 25th to 75th percentile cutoff within each grade:

| Quantile | % Treated | AIPW ATE (pp) | 95% CI |
|---|---|---|---|
| 0.25 | 72.2% | +2.1052 | [1.936, 2.274] |
| 0.33 | 62.0% | +2.1675 | [2.011, 2.324] |
| 0.50 (baseline) | 46.9% | +2.0836 | [1.925, 2.242] |
| 0.67 | 32.2% | +1.9208 | [1.740, 2.102] |
| 0.75 | 22.2% | +1.4665 | [1.198, 1.735] |

The monotonic decline as the threshold increases provides evidence of treatment effect heterogeneity. The marginal impact of interest rates appears stronger among relatively lower-risk borrowers and attenuates for higher-risk segments.

---

## Robustness Summary

| Specification | ATE (pp) | 95% CI | N |
|---|---|---|---|
| **Baseline AIPW** | **+2.084** | [1.925, 2.242] | 1,025,917 |
| 36-month loans only | +2.261 | [2.090, 2.432] | 777,875 |
| 2014-2015 vintage | +2.037 | [1.794, 2.280] | 598,375 |
| Debt purposes | +2.142 | [1.966, 2.317] | 842,797 |
| Other purposes | +1.739 | [1.373, 2.106] | 183,120 |

The treatment effect remains stable across all subsamples, including different loan terms, issuance periods, and loan purposes. This consistency strengthens the causal interpretation.

---

## Appendix: Selection Bias

### Approved vs. Rejected Covariate Distributions

Rejected applicants differ substantially from approved borrowers:

1. **FICO Score**: Acts as a hard threshold (around 660) in the approval process, indicating rule-based screening rather than a smooth decision boundary.
2. **DTI**: Both extremely low and extremely high DTI values are associated with rejection, while mid-range borrowers are evaluated more flexibly.
3. **Loan Amount**: Certain commonly requested amounts are disproportionately rejected, suggesting interaction with underwriting rules.
4. **Employment Length**: Having no employment history is a strong negative signal for approval.

### Approval Model

A logistic regression trained on combined approved/rejected data:

| Metric | Value |
|---|---|
| 5-Fold CV AUC | **0.9215** |

| Feature | Log-Odds (Standardized) |
|---|---|
| fico_mid | +1.4953 |
| emp_length_num | +1.1807 |
| loan_amnt | -0.1966 |
| dti | -85.1076 |

The high AUC (0.92) confirms that approval is highly predictable from observables. FICO is the strongest positive predictor, DTI has a strong negative effect.

### Implications for Causal Estimates

Our causal estimates are identified on a selected subset of relatively stable, rule-compliant borrowers rather than the full applicant population. Rejected applicants are more financially fragile, and their default response to a rate increase would plausibly exceed our within-sample estimates. **Causal estimates based on approved loans likely understate the true impact of interest rates in the broader applicant population.**
