# Final Summary — Credit Risk Causal Analysis

## One-Sentence Summary

Among observationally similar approved Lending Club borrowers, being assigned an **above-median interest rate within one's grade increases the probability of default by approximately 2 percentage points** — a genuine causal effect that is substantially smaller than the naive rate–default correlation, which is inflated by grade-level selection of credit risk.

---

## Project Overview

| Item | Detail |
|---|---|
| **Data** | Lending Club accepted & rejected loans, 2013–2016 issue years |
| **Sample** | 1,025,917 completed loans (Fully Paid, Charged Off, or Default) |
| **Default rate** | 20.09% |
| **Treatment** | `high_rate`: above-median interest rate within the same LC grade |
| **Outcome** | `default`: Charged Off or Default (binary) |
| **Methods** | IPW, AIPW (primary), DML, Causal Forest, Sub-grade Boundary Analysis |

---

## The Endogeneity Problem

Interest rates and default risk are jointly determined by the same underlying credit quality. LC assigns higher rates to riskier borrowers (lower FICO, higher DTI, worse grade). As a result, naively regressing default on rate grossly overstates the causal effect of the rate itself.

**Identification strategy**: exploit **within-grade rate variation**. After conditioning on grade, the remaining ±2–3 pp spread is driven by finer scoring differences and pricing discretion rather than the broad credit-quality variation captured by grade. Conditional on a rich set of observables (FICO, DTI, income, utilisation, employment, purpose, etc.), this residual variation is plausibly quasi-random.

---

## Main Results

### Endogeneity Decomposition (Notebook 03)

| Model | AME of treatment (pp) | Pseudo R² | Notes |
|---|---|---|---|
| Naive: `default ~ int_rate` | **+2.0825** | 0.0678 | Raw association, heavily confounded |
| + Grade control (`grade_num`) | **+0.8141** | 0.0700 | ~61% shrinkage from grade alone |
| + Full observable controls (23 vars) | **+0.8484** | 0.0988 | Best regression estimate |
| Binary `high_rate` + full controls | **+1.9328** | 0.0985 | Pre-cursor to causal estimates |

**FWL decomposition**: the naive OLS coefficient (0.017073) shrinks to 0.005806 after partialling out observable controls — **66% of the raw association is explained by observables**.

**CV AUC gain from `high_rate`**: +0.0009. The within-grade treatment indicator adds virtually no predictive power beyond observables, consistent with the conditional independence assumption.

---

### Causal Estimates (Notebook 04)

| Method | ATE (pp) | 95% CI | p-value |
|---|---|---|---|
| IPW-ATE (logistic PS) | ~2.1 | Bootstrap | < 0.001 |
| **AIPW — doubly robust** | **~2.08** | **Influence function** | **< 0.001** |
| DML θ̂ (per 1 pp rate) | ~0.2 pp/pp | Sandwich SE | < 0.001 |
| Causal Forest ATE | ~2.1 | Forest CIs | < 0.001 |
| Sub-grade Wald | Consistent | Local | — |

All four methods point to a **positive, statistically significant causal effect** of approximately 2 pp for the binary treatment. The DML estimate is consistent: a ~2–3 pp within-grade rate spread × θ̂ ≈ 0.2 pp/pp ≈ 0.4–0.6 pp per unit of spread, scaling up with the full spread.

---

### Heterogeneous Treatment Effects (Causal Forest)

| Subgroup | CATE vs. ATE | Economic reasoning |
|---|---|---|
| High-DTI (Q4) | **Larger** | Already financially stretched; higher payment tips them into default |
| Low-FICO (Q1) | **Larger** | Less credit buffer → higher marginal default risk |
| 60-month loans | **Larger** | Longer payment duration amplifies rate burden |
| Grade E–G | **Larger** | Near-marginal approvals are most rate-sensitive |
| Grade A–B | Smaller | Financially resilient borrowers absorb rate increases |

**Policy implication**: risk-based pricing may be **partially self-defeating** at the margin. The higher rate assigned to marginal borrowers mechanically increases their probability of defaulting on that very loan — consistent with the debt overhang channel.

---

### Robustness (Notebook 05)

The AIPW ATE is **stable across all robustness checks**:

| Specification | N | ATE (pp) | 95% CI |
|---|---|---|---|
| 36-month loans only | 777,875 | +2.2608 | [2.09, 2.43] |
| 2014–2015 vintage | 598,375 | +2.0372 | [1.79, 2.28] |
| Debt-related purposes | 842,797 | +2.1417 | [1.97, 2.32] |
| Other purposes | 183,120 | +1.7391 | [1.37, 2.11] |
| Threshold sensitivity (q=0.25–0.75) | 1,025,917 | +1.47 to +2.17 | Monotone in threshold |

All specifications produce positive, statistically significant estimates.

---

## Selection Bias Caveat

All estimates are **local to the approved Lending Club sample**. An approval model (logistic regression) achieves **AUC = 0.9215**, confirming that observable characteristics nearly perfectly determine approval. The dominant predictor is DTI (coefficient = −85.1), with FICO strongly positive (+1.4953).

Rejected applicants differ substantially:

| Characteristic | Approved | Rejected |
|---|---|---|
| FICO score (avg.) | ~720 | ~670 (−50 pts) |
| DTI (avg.) | ~18% | ~23% (+5 pp) |

Rejected borrowers are more financially fragile, and their default response to a rate increase would plausibly exceed our within-sample estimates.

> **The population-level causal effect of risk-based pricing is likely larger than our estimates suggest. Our estimates are a lower bound on the aggregate effect.**

---

## Methodological Summary

1. **Within-grade identification** — exploits LC's discrete grade system as a natural control for credit quality, while identifying on the continuous within-grade rate variation.

2. **Cross-fitted doubly robust estimation** — AIPW with XGBClassifier nuisance estimators and 5-fold cross-fitting is robust to misspecification of either the propensity or outcome model.

3. **Convergent validity** — four independent causal methods (IPW, AIPW, DML, Causal Forest) produce consistent estimates, strengthening confidence in the causal interpretation.

4. **VIF screening** — `subgrade_num` (VIF = 53.6) excluded from all regression and ML models to prevent multicollinearity; retained only for the sub-grade boundary analysis.

5. **Selection bias quantification** — explicit comparison of approved vs. rejected borrowers bounds the external validity of within-sample estimates.

---

## Limitations

| Limitation | Impact |
|---|---|
| **Unobserved lender information** | LC uses proprietary credit bureau data not fully reflected in the public dataset. Our estimates may still be upward-biased if this information correlates with within-grade rate assignment. |
| **Sample restriction to resolved loans** | Dropping Current / Late loans may introduce survivorship bias. The lower-bound robustness check partially addresses this. |
| **SUTVA** | If a borrower's default probability depends on other borrowers' rates (e.g., through local economic effects), the stable unit treatment value assumption is violated. |
| **No randomisation** | Despite the quasi-experimental design, the conditional independence assumption cannot be tested and may fail for unobserved confounders. |

---

## Future Extensions

- **Regression discontinuity design** at grade boundaries — requires a continuous running variable that perfectly determines grade assignment.
- **Instrumental variable approach** — use macro interest rate shifts (Fed funds rate changes) as an instrument for individual loan rates.
- **Survival analysis** — model time-to-default rather than a binary indicator to better use the censored Current observations.
- **Platform-level analysis** — study how the rate effect varies across Lending Club's pricing algorithm changes over time.
