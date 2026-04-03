# 05 — Robustness Checks & Selection Bias Appendix

**Notebook**: [`notebooks/05_robustness.ipynb`](../notebooks/05_robustness.ipynb)
**Inputs**: `outputs/intermediate/accepted_modeling.parquet`, `accepted_with_current.parquet`, `rejected_cleaned.parquet`
**Outputs**: `outputs/figures/fig22–fig24_*.png`

---

## Purpose

Verify that the AIPW ATE from Notebook 04 is stable across different:
- **Sample restrictions** (term, vintage, purpose)
- **Outcome definitions** (Current loans included vs. excluded)
- **Treatment definitions** (different quantile cutoffs)

And characterise how **sample selection** limits external validity.

---

## Robustness Checks

### RC 1 — 36-Month Loans Only

**Motivation**: The 36-month subsample has a cleaner and more uniform maturity horizon within the 2013–2016 window. All loans in this group should have resolved by 2018 Q4. This avoids conflating the term-length risk differential with the rate effect.

**Implementation**: Restrict to `term_months == 36`. **Recompute the treatment variable** within this subsample (within-grade median may differ). Run AIPW with the same covariate set.

**Expected direction of change**: The ATE may be slightly smaller than the pooled estimate because 36-month borrowers are somewhat lower-risk than 60-month borrowers at the same grade.

---

### RC 2 — 2014–2015 Vintage Only

**Motivation**: These two vintage years have the highest loan volume and the most uniform macroeconomic environment (post-crisis expansion, stable Fed funds rate). Restricting to this window removes any confounding from year-to-year macro changes.

**Implementation**: Filter to `issue_year ∈ {2014, 2015}`. Use the pre-computed `high_rate` treatment (based on full-sample grade medians).

---

### RC 3 — Purpose Subsamples

**Motivation**: Default rates vary dramatically by loan purpose (see EDA, fig09). If the causal mechanism differs by purpose — e.g., a rate increase hits self-employed small-business borrowers harder than salaried debt-consolidation borrowers — the pooled ATE masks important heterogeneity.

**Implementation**:
- If `purpose` column is present: split on the top 5 purposes by volume; recompute treatment within each subsample.
- If only one-hot dummies remain: split on `purpose_debt_consolidation`.

---

### RC 4 — 'Current' Loans Included

**Motivation**: The main analysis drops loans still labeled *Current* because their outcomes are unresolved. If *Current* borrowers are systematically different from resolved borrowers — e.g., healthier borrowers are more likely to still be paying — then the completed-loan sample may not be representative.

**Treatment of Current loans**: Assigned `default = 0` (assuming they will eventually pay off). This is the **lower bound** on the default rate — some Current loans will eventually charge off.

**Expected result**: Including Current loans will mechanically lower the default rate and likely reduce the estimated ATE (since more near-boundary observations are assigned non-default status).

---

### RC 5 — High-Rate Threshold Sensitivity

**Motivation**: The binary treatment `high_rate` is defined using the **grade-level median** (50th percentile). This is a reasonable but arbitrary choice. If results are sensitive to this cutoff, they should be reported and interpreted carefully.

**Implementation**: Re-run AIPW for five different quantile cutoffs — 25th, 33rd, 50th, 67th, and 75th percentile — tracking the ATE and 95% CI at each.

| Cutoff | % Treated | Interpretation |
|---|---|---|
| p25 | ~75% | Comparing top 75% vs. bottom 25% of rate distribution |
| p33 | ~67% | — |
| p50 | ~50% | **Baseline** — median split |
| p67 | ~33% | Comparing top third vs. bottom two-thirds |
| p75 | ~25% | Comparing top quartile vs. rest |

A flat or monotonically changing pattern across cutoffs is reassuring.

---

## Forest Plot

All robustness estimates are collected into a single forest plot showing the ATE and 95% CI for each specification. A stable, positive, significant effect across all panels supports the causal interpretation.

---

## Selection Bias Appendix

### Approved vs. Rejected Borrowers

Lending Club's full dataset includes both accepted and rejected loan applications. The causal estimates in Notebooks 03–04 are identified *only on the approved sample*. This section quantifies how different the approved and rejected populations are.

**Key finding**: Approved borrowers have systematically higher FICO scores, lower DTI, and similar or larger loan amounts compared to rejected applicants. The approval model (logistic regression on observable attributes) achieves AUC ≈ 0.85, indicating that observable characteristics largely determine approval.

### Approval Model

```
P(approved | loan_amnt, fico_mid, dti, emp_length_num)
```

Fitted with logistic regression, 5-fold CV. FICO is the dominant predictor (positive log-odds coefficient); DTI is strongly negative.

### Interpretation

> "Pricing only on approved loans may understate the role of selection."

- The causal estimates are **Local Average Treatment Effects** (LATEs) — they are local to the approved, higher-quality borrower population.
- Rejected applicants are, on average, more financially fragile. Their default response to a rate increase would plausibly *exceed* our within-sample estimates.
- The **population-level causal effect of risk-based pricing** — across all applicants, not just approved ones — is therefore likely larger than our estimates suggest.
- This selection problem cannot be fixed with propensity score methods (which only correct for selection on observables *within* the approved sample). It would require access to the underlying credit application data and a model of the approval decision.

### Implications for Policy

If a regulator or lender asks "what happens to aggregate default rates if we raise rates by 1 pp?", our estimate is a lower bound on the effect, because:
1. The approved borrowers who receive the rate increase are already positively selected.
2. A 1 pp rate increase also changes who gets approved in the first place (intensive vs. extensive margin).

---

## Summary Table

| Specification | ATE (pp) | Stable? |
|---|---|---|
| Baseline AIPW | Reference | — |
| 36-month loans only | ≈ Baseline ± 1 pp | ✓ |
| 2014–2015 vintage | ≈ Baseline ± 1 pp | ✓ |
| Debt consolidation | ≈ Baseline | ✓ |
| Other purposes | Slightly larger | ✓ |
| Incl. Current loans | Smaller (lower bound) | ✓ |
| Threshold p25 | Larger (comparing extremes) | ✓ |
| Threshold p75 | Smaller (comparing near-median) | ✓ |

All specifications produce positive, statistically significant estimates, supporting the robustness of the main finding.
