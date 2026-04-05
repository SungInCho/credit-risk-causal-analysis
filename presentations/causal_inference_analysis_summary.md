# Does Higher Interest Rate Cause Default?
## A Causal Inference Study on Lending Club Loan Pricing

---

## Executive Summary

This analysis investigates whether the strong correlation between interest rates and loan defaults reflects a genuine causal relationship or merely the underlying credit risk that drives rate assignment. Using 1.03 million Lending Club loans issued between 2013 and 2016, we find that being assigned an above-median interest rate within the same credit grade **increases default probability by approximately 2 percentage points** — a statistically significant and economically meaningful effect confirmed across four independent causal methods and six robustness specifications.

Critically, the naive rate-default correlation **overstates the causal effect by roughly 60%** due to confounding from risk-based pricing. This distinction has direct implications for pricing strategy, credit policy, and portfolio risk management.

---

## Business Context

Lending platforms assign higher interest rates to riskier borrowers. This creates a fundamental analytical challenge:

- **What we observe**: Borrowers with higher rates default more often.
- **The question**: Is the higher rate *causing* more defaults (through increased payment burden), or simply *reflecting* the underlying risk?
- **Why it matters**: Overestimating the causal effect leads to mispriced risk and flawed policy decisions. Underestimating it misses a genuine channel through which aggressive pricing damages loan performance.

---

## Data and Sample

| Item | Detail |
|---|---|
| Source | Lending Club public loan dataset (Kaggle) |
| Period | 2013-2016 issue years (mature, resolved outcomes) |
| Sample | 1,025,917 completed loans (Fully Paid or Charged Off/Default) |
| Default rate | 20.09% |
| Treatment | Above-median interest rate within the same credit grade |

---

## Analytical Approach

### Identification Strategy

We define treatment as receiving an interest rate above the within-grade median. Within each credit grade, the 2-3 percentage point rate spread is driven by finer scoring differences and pricing discretion — not the broad credit-quality variation captured by grade assignment. After conditioning on 24 observable borrower characteristics (FICO, DTI, income, employment, etc.), this residual variation is plausibly quasi-random.

### Methods Applied

| Method | Purpose |
|---|---|
| Logistic regression progression | Quantify confounding (endogeneity decomposition) |
| IPW (propensity score weighting) | Reweight sample for treated/control comparability |
| AIPW (doubly robust estimation) | Primary estimate; robust to model misspecification |
| Double Machine Learning | Continuous treatment (rate per percentage point) |
| Causal Forest | Heterogeneous effects across borrower segments |
| Sub-grade boundary analysis | Local validation at discrete grade cutoffs |

---

## Key Finding 1: The Naive Estimate Is Severely Biased

| Model | AME on P(Default) | Shrinkage |
|---|---|---|
| Naive (no controls) | +2.08 pp | — |
| + Grade control | +0.81 pp | -61% |
| + Full controls (23 variables) | +0.84 pp | -60% |

**66% of the raw rate-default association is explained by observable borrower characteristics.** The naive correlation is not a reliable basis for pricing or policy decisions.

---

## Key Finding 2: The Causal Effect Is Real and Significant

| Method | ATE on P(Default) | 95% CI |
|---|---|---|
| IPW | +2.45 pp | [2.28, 2.62] |
| **AIPW (primary)** | **+2.08 pp** | **[1.92, 2.24]** |
| DML (per 1 pp rate) | +1.15 pp | [1.08, 1.22] |
| Causal Forest | +1.80 pp | — |

All methods converge on a positive, statistically significant effect. An above-median rate within grade increases default probability by approximately **2 percentage points**.

---

## Key Finding 3: Effects Vary Across Borrower Segments

The Causal Forest reveals meaningful heterogeneity:

| Segment | Treatment Effect | Interpretation |
|---|---|---|
| Lower-risk grades (A-C) | **Larger** (above average) | Safer borrowers are more sensitive to rate increases |
| Higher-risk grades (D-G) | Smaller (below average) | Default driven by broader risk factors, not rate alone |
| 36-month loans | 2.06 pp | Larger effect |
| 60-month loans | 1.00 pp | Smaller effect |

The sub-grade boundary analysis confirms this pattern: discrete rate jumps at A-D grade boundaries produce significant default increases, while E-G boundaries show no significant effect.

---

## Key Finding 4: Results Are Robust

The AIPW ATE is stable across all tested specifications:

| Specification | ATE (pp) | N |
|---|---|---|
| Baseline | +2.08 | 1,025,917 |
| 36-month loans only | +2.26 | 777,875 |
| 2014-2015 vintage only | +2.04 | 598,375 |
| Debt-related purposes | +2.14 | 842,797 |
| Other purposes | +1.74 | 183,120 |

Varying the treatment threshold from the 25th to 75th percentile produces a monotonic pattern consistent with dose-response logic.

---

## Key Finding 5: Selection Limits Generalizability

All estimates are based on **approved loans only**. An approval model (AUC = 0.92) shows that rejected applicants have substantially lower FICO scores (~670 vs. ~720) and higher DTI (~23% vs. ~18%). Since rejected borrowers are more financially fragile, the true population-level effect of rate increases is likely **larger** than our within-sample estimates.

---

## Implications

### For Pricing Strategy

Risk-based pricing may be **partially self-defeating** at the margin. The higher rate assigned to borderline borrowers increases their probability of defaulting on that very loan. This suggests that aggressive rate increases on marginal credits may reduce, rather than increase, expected revenue after accounting for higher default losses.

### For Credit Policy

The finding that lower-risk borrowers are more sensitive to rate increases implies that **rate increases have the strongest behavioral impact on the segments least likely to default** — and relatively less impact on the highest-risk segments where default is driven by fundamental creditworthiness.

### For Portfolio Risk Management

The naive rate-default correlation substantially overstates the true pricing effect. Risk models that treat the raw correlation as causal will systematically **overestimate the default impact of rate changes** and misallocate risk capital.

---

## Methodology Summary

| Step | Key Output |
|---|---|
| Data cleaning (NB 01) | 1,025,982 completed loans, 2013-2016 |
| EDA and feature engineering (NB 02) | 24 covariates; `high_rate` treatment; modeling dataset (1,025,917 rows) |
| Endogeneity quantification (NB 03) | 66% of naive coefficient explained by observables |
| Causal estimation (NB 04) | AIPW ATE = +2.08 pp, confirmed across four methods |
| Robustness and selection bias (NB 05) | Stable across six specifications; approval AUC = 0.92 |

---

## Technical Notes

- **Doubly robust estimation**: AIPW is consistent if either the propensity score or outcome model is correctly specified, providing insurance against misspecification.
- **Cross-fitting**: All nuisance models use 5-fold cross-fitting to prevent overfitting bias in semiparametric estimation.
- **Propensity score models**: Logistic regression with L2 regularization; scores clipped to [0.01, 0.99].
- **Outcome models**: XGBClassifier (200 estimators, max_depth=4, learning_rate=0.05).
- **Software**: `xgboost`, `scikit-learn`, `statsmodels`, `econml`, `scipy`, `pandas`, `matplotlib`.
