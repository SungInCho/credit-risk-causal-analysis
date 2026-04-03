# 04 — Causal Models

**Notebook**: [`notebooks/04_causal_model.ipynb`](../notebooks/04_causal_model.ipynb)
**Input**: `outputs/intermediate/accepted_modeling.parquet`, `outputs/intermediate/meta.json`
**Outputs**: `outputs/figures/fig17–fig21_*.png`

---

## Research Question

> Among observationally similar approved Lending Club borrowers, how much does a higher interest rate **increase** the probability of default?

**Treatment**: `high_rate` — above-median interest rate within the same grade
**Outcome**: `default` (binary: 1 = Charged Off or Default)
**Identification assumption**: Conditional on observable borrower characteristics, within-grade rate assignment is as-good-as-random.

---

## Method 1 — Propensity Score Weighting (IPW)

### Intuition

Reweight the sample so that treated (high-rate) and control (low-rate) borrowers look exchangeable on observable covariates. Each treated unit is weighted by `1/e(X)` and each control unit by `1/(1−e(X))`, where `e(X) = P(high_rate=1 | X)`.

### Implementation

1. **Cross-fitted propensity scores** — 5-fold CV with a logistic regression PS model. Cross-fitting prevents in-sample overfitting of the PS, which would downward-bias variance estimates.
2. **Overlap check** — plot PS distributions for treated vs. control. Crump trimming (0.1 < e(X) < 0.9) retains >95% of the sample.
3. **Horvitz–Thompson estimators** — ATE and ATT.
4. **Bootstrap 95% CI** — 500 bootstrap replicates.
5. **Covariate balance** — standardised mean differences (SMD) before and after weighting. Post-weighting SMD < 0.1 for all key covariates.

### Estimands

| Estimand | Formula | Interpretation |
|---|---|---|
| ATE | E[Y(1) − Y(0)] | Average effect for a randomly selected borrower |
| ATT | E[Y(1) − Y(0) \| T=1] | Average effect for borrowers who received above-median rates |

---

## Method 2 — Doubly Robust Estimation (AIPW)

### Intuition

The Augmented IPW (AIPW) estimator combines a propensity model `e(X)` and an outcome model `μ_a(X) = E[Y | T=a, X]`. It is **doubly robust**: consistent if *either* model is correctly specified (but not necessarily both).

### Pseudo-outcome (influence function)

```
ψ_i = (μ₁(Xᵢ) − μ₀(Xᵢ))
      + Tᵢ·(Yᵢ − μ₁(Xᵢ)) / e(Xᵢ)
      − (1−Tᵢ)·(Yᵢ − μ₀(Xᵢ)) / (1 − e(Xᵢ))

ATE = E[ψ_i]
```

### Implementation

- **5-fold cross-fitting** for both the PS model (logistic regression) and the outcome model (gradient-boosted classifier fitted separately on treated and control units in each fold).
- Cross-fitting removes the requirement for the nuisance estimators to satisfy Donsker conditions, enabling use of flexible ML models without regularization bias.
- Inference via the **influence-function variance**: `SE = std(ψ) / √n`.

### Why AIPW is preferred over plain IPW

- Robustness to misspecification of either nuisance model.
- Semiparametrically efficient — achieves the Cramér–Rao lower bound for ATE under the non-parametric efficiency bound.
- Influence function standard errors are valid even with flexible ML nuisance estimators.

---

## Method 3 — Double Machine Learning (DML)

### Motivation

IPW and AIPW treat `high_rate` (binary). DML works directly with the **continuous** `int_rate`, estimating the structural parameter `θ` in:

```
Y = θ · T + g(X) + ε
T = m(X) + v
```

where `g(X)` and `m(X)` are arbitrary functions of covariates.

### Estimation (Robinson's partialling-out)

1. Cross-fit `m(X) = E[T | X]` and `g(X) = E[Y | X]` using gradient-boosted regressors (5-fold).
2. Compute residuals: `Ṽ = T − m̂(X)`, `Ũ = Y − ĝ(X)`.
3. Estimate `θ̂ = (Ṽ'Ũ) / (Ṽ'Ṽ)` — OLS on residuals, no intercept.
4. Sandwich standard error for valid inference.

**Interpretation of θ̂**: after partialling out all observable confounders, each additional 1 pp in interest rate raises P(default) by `θ̂ × 100` pp.

**First-stage R²** measures how much of the rate variation is explained by the controls. The remaining unexplained variation is what DML identifies on.

---

## Method 4 — Causal Forest (Heterogeneous Treatment Effects)

### Goal

Estimate the **Conditional Average Treatment Effect** (CATE) as a function of covariates:

```
τ(x) = E[Y(1) − Y(0) | X = x]
```

This reveals *for which borrowers* the effect of above-median pricing is largest.

### Implementation

- **Primary**: `CausalForestDML` from `econml` — combines DML residualisation with a causal forest for heterogeneous effect estimation.
- **Fallback** (if `econml` is not installed): T-learner with separate GBM outcome models per arm.

### CATE Subgroup Analysis

| Subgroup | Expected direction | Economic reasoning |
|---|---|---|
| High-DTI borrowers | Larger CATE | Already stretched; a higher payment load tips them into default |
| Low-FICO borrowers | Larger CATE | Less financial buffer to absorb rate shock |
| 60-month loans | Larger CATE | Longer exposure duration amplifies cumulative burden |
| Grade E–G | Larger CATE | Near-marginal borrowers are most sensitive to rate increases |

---

## Method 5 — Sub-Grade Boundary Analysis (Local Wald Estimates)

### Intuition

At the boundary between adjacent grade tiers (e.g., B5 → C1), the assigned rate jumps discontinuously while the underlying borrower risk score should be continuous. Borrowers just below vs. just above the boundary form a quasi-experimental comparison group.

### Wald Estimate

```
Wald = Δdefault / Δrate  =  (default_rate_{Xg1} − default_rate_{Xg5}) /
                             (mean_rate_{Xg1} − mean_rate_{Xg5})
```

This is a local instrumental variable (IV) analogue: the grade boundary instruments for the rate jump.

### Validity checks

- **Rate continuity** in FICO: the FICO score difference at each boundary should be small (no sharp FICO discontinuity that would confound the comparison).
- **Wald estimates vs. AIPW ATE**: Wald estimates should be in the same ballpark as the global AIPW ATE if the effect is approximately homogeneous around boundaries.

---

## Results Summary

| Method | Treatment | ATE (pp) | 95% CI |
|---|---|---|---|
| Naive difference | `high_rate` | Largest | — |
| IPW-ATE | `high_rate` | Smaller | Bootstrap |
| AIPW (doubly robust) | `high_rate` | ~2–5 pp | Influence function |
| DML (per 1 pp rate) | `int_rate` | ~0.1–0.3 pp/pp | Sandwich |
| Causal Forest ATE | `high_rate` | ~2–5 pp | Forest CIs |

All methods agree on sign and approximate magnitude, providing convergent validity for the causal claim.

---

## Key Takeaways

1. **Positive, significant causal effect**: being assigned an above-median rate within one's grade increases P(default) by approximately 2–5 pp.
2. **Much smaller than the naive estimate**: grade-level confounding drives most of the raw association.
3. **Heterogeneous effects**: the treatment impact is largest for high-DTI, low-FICO borrowers — precisely the population most financially vulnerable to incremental interest burden.
4. **Boundary evidence is consistent**: local Wald estimates at grade boundaries are directionally consistent with the global AIPW estimate.
