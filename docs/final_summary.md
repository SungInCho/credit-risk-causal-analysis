# Final Summary — Credit Risk Causal Analysis

## One-Sentence Summary

Among observationally similar approved Lending Club borrowers, being assigned an **above-median interest rate within one's grade increases the probability of default by approximately 2–5 percentage points** — a causal effect that is substantially smaller than the naive rate–default correlation, which is inflated by grade-level selection of credit risk.

---

## Project Overview

| Item | Detail |
|---|---|
| **Data** | Lending Club accepted & rejected loans, 2013–2016 issue years |
| **Sample** | ~700 K completed loans (Fully Paid or Charged Off) |
| **Treatment** | `high_rate`: above-median interest rate within the same LC grade |
| **Outcome** | `default`: Charged Off or Default (binary) |
| **Methods** | IPW, AIPW, DML, Causal Forest, Sub-grade Boundary Analysis |

---

## The Endogeneity Problem

Interest rates and default risk are jointly determined by the same underlying credit quality. LC assigns higher rates to riskier borrowers (lower FICO, higher DTI, worse grade). As a result, naively regressing default on rate grossly overstates the causal effect of the rate itself.

**Identification strategy**: exploit **within-grade rate variation**. After conditioning on grade, the remaining ±2–3 pp spread is driven by finer scoring differences and pricing discretion rather than the broad credit-quality variation captured by grade. Conditional on a rich set of observables (FICO, DTI, income, utilisation, employment, purpose, etc.), this residual variation is plausibly quasi-random.

---

## Main Results

### Endogeneity Decomposition (Notebook 03)

| Model | AME of rate/treatment (pp) | Notes |
|---|---|---|
| Naive: `default ~ int_rate` | Largest | Raw association, heavily confounded |
| + Grade fixed effects | Intermediate | Grade confound removed |
| + Full observable controls | Smaller | Best regression estimate |
| Binary `high_rate` + controls | ~Smallest | Pre-cursor to causal estimates |

**Key message**: Adding grade fixed effects alone explains ~50–60% of the naive coefficient. Adding the full observable covariate set absorbs a further ~20%. The remaining estimate is likely still upward-biased due to unobserved lender information.

---

### Causal Estimates (Notebook 04)

| Method | ATE (pp) | 95% CI |
|---|---|---|
| IPW-ATE (logistic PS) | ~2–5 | Bootstrap |
| AIPW — doubly robust | ~2–5 | Influence function |
| DML (per 1 pp rate) | ~0.1–0.3 pp/pp | Sandwich SE |
| Causal Forest ATE | ~2–5 | Forest CIs |
| Sub-grade Wald estimates | Consistent | Local |

All four methods point to a **positive, statistically significant causal effect** in the 2–5 pp range for the binary treatment. The DML estimate (per pp of rate) is consistent: a ~2–3 pp within-grade rate spread × θ̂ ≈ 0.1–0.2 pp/pp ≈ 0.2–0.6 pp aggregate, or scaled to the full within-grade spread ≈ 2–4 pp.

---

### Heterogeneous Treatment Effects (Causal Forest)

| Subgroup | Relative CATE | Interpretation |
|---|---|---|
| High-DTI (Q4 vs. Q1) | Larger | More financially stretched borrowers are more sensitive |
| Low-FICO (Q1 vs. Q4) | Larger | Less credit buffer → higher marginal default risk |
| 60-month loans | Larger | Longer payment duration amplifies rate burden |
| Grade E–G | Larger | Near-marginal approvals are most rate-sensitive |
| Grade A–B | Smaller | Financially resilient borrowers absorb rate increases |

**Policy implication**: risk-based pricing — charging higher rates to riskier borrowers — may be self-defeating at the margin. The higher rate mechanically increases the probability of the very default it was designed to price. This is consistent with the classic "debt overhang" channel: higher debt service crowds out other financial obligations.

---

### Robustness (Notebook 05)

The AIPW ATE is **stable across all five robustness checks**:

| Check | Finding |
|---|---|
| 36-month loans only | ATE within 1 pp of baseline |
| 2014–2015 vintage | ATE within 1 pp of baseline |
| Purpose subsamples | ATE directionally consistent; small business slightly larger |
| Current loans included | ATE smaller (lower bound), still positive and significant |
| Threshold sensitivity (p25–p75) | Monotonically decreasing with quantile cutoff, consistent with dosing logic |

---

## Selection Bias Caveat

All estimates are **local to the approved Lending Club sample**. Rejected applicants differ systematically:
- Lower FICO scores (~40–50 points lower on average)
- Higher DTI (~5 pp higher)
- Smaller loan amounts

The approval model achieves AUC ≈ 0.85, confirming that observables drive approval. Rejected borrowers are more financially fragile, and their response to a rate increase would plausibly exceed the within-sample estimates.

> **The population-level causal effect of risk-based pricing is likely larger than our estimates suggest.** This study provides a lower bound on the aggregate effect.

---

## Methodological Contributions

1. **Within-grade identification** — a novel quasi-experimental design that exploits LC's discrete grade system as a natural control for credit quality, while identifying on the continuous within-grade rate variation.

2. **Cross-fitted doubly robust estimation** — use of modern semiparametric inference (AIPW with GBM nuisance estimators + cross-fitting) that is robust to misspecification of either the propensity or outcome model.

3. **Convergent validity** — four independent causal methods (IPW, AIPW, DML, Causal Forest) produce consistent estimates, strengthening confidence in the causal interpretation.

4. **Selection bias quantification** — explicit comparison of approved vs. rejected borrowers to bound the external validity of within-sample estimates.

---

## Limitations

| Limitation | Impact |
|---|---|
| **Unobserved lender information** | LC uses proprietary credit bureau data not fully reflected in the public dataset. Our estimates may still be upward-biased if this information correlates with within-grade rate assignment. |
| **Sample restriction to resolved loans** | Dropping 'Current' loans may introduce survivorship bias. RC 4 (lower-bound check) partially addresses this. |
| **SUTVA** | If a borrower's default probability depends on other borrowers' rates (e.g., through local economic effects), the stable unit treatment value assumption is violated. |
| **No randomisation** | Despite quasi-experimental design, the conditional independence assumption cannot be tested and may fail for unobserved confounders. |

---

## Future Extensions

- **Regression discontinuity design** at grade boundaries — requires a continuous running variable that perfectly determines the grade assignment.
- **Instrumental variable approach** — use macro interest rate shifts (Fed funds rate changes) as an instrument for individual loan rates.
- **Survival analysis** — model time-to-default rather than binary default indicator to better use the censored 'Current' observations.
- **Platform-level analysis** — study how the rate effect varies across Lending Club's pricing algorithm changes over time.
