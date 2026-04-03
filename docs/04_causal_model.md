# 04 — Causal Models

**Notebook**: [`notebooks/04_causal_model.ipynb`](../notebooks/04_causal_model.ipynb)
**Input**: `data/processed/accepted_modeling.parquet`, `data/processed/meta.json`
**Outputs**: `outputs/figures/fig17–fig21_*.png`

---

## Research Question

> Among observationally similar approved Lending Club borrowers, how much does a higher interest rate **increase** the probability of default?

**Treatment**: `high_rate` — above-median interest rate within the same grade
**Outcome**: `default` (binary: 1 = Charged Off or Default)
**Identification assumption**: Conditional on observable borrower characteristics, within-grade rate assignment is as-good-as-random.

---

## Nuisance Model Specifications

All causal methods in this notebook rely on one or both of the following ML nuisance estimators:

**Propensity score model** — Logistic Regression with L2 regularisation (C=0.5), scores clipped to [0.01, 0.99].

**Outcome model** — XGBClassifier with:
```
n_estimators=200, max_depth=4, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8
```

**DML treatment/outcome models** — XGBRegressor with the same tree hyperparameters.

All nuisance models use **5-fold StratifiedKFold cross-fitting**. The same data is never used to both fit the nuisance model and construct the pseudo-outcome, which is required for valid semiparametric inference with flexible ML estimators.

---

## Method 1 — Propensity Score Weighting (IPW)

### Intuition

Reweight the sample so that treated (high-rate) and control (low-rate) borrowers look exchangeable on observable covariates. Each treated unit is weighted by `1/e(X)` and each control unit by `1/(1−e(X))`, where `e(X) = P(high_rate=1 | X)`.

### Implementation

1. **Cross-fitted propensity scores** — 5-fold StratifiedKFold with logistic regression, scores clipped to [0.01, 0.99].
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

## Method 2 — Doubly Robust Estimation (AIPW) ← Primary Estimate

### Intuition

The Augmented IPW (AIPW) estimator combines a propensity model `e(X)` and an outcome model `μ_a(X) = E[Y | T=a, X]`. It is **doubly robust**: consistent if *either* model is correctly specified (but not necessarily both).

### Pseudo-outcome (influence function)

```
ψ_i = (μ₁(Xᵢ) − μ₀(Xᵢ))
      + Tᵢ·(Yᵢ − μ₁(Xᵢ)) / e(Xᵢ)
      − (1−Tᵢ)·(Yᵢ − μ₀(Xᵢ)) / (1 − e(Xᵢ))

ATE = mean(ψ_i)
SE  = std(ψ_i) / sqrt(n)
```

### Implementation

- **5-fold StratifiedKFold cross-fitting** for both the propensity score (logistic regression) and the outcome model (XGBClassifier fitted separately on treated and control units in each fold).
- Cross-fitting removes the requirement for the nuisance estimators to satisfy Donsker conditions, enabling use of flexible ML models without regularisation bias.
- Inference via the **influence-function variance**: `SE = std(ψ) / √n`.

### Why AIPW is preferred over plain IPW

- Robustness to misspecification of either nuisance model.
- Semiparametrically efficient — achieves the Cramér–Rao lower bound under the non-parametric efficiency bound.
- Influence-function standard errors are valid even with flexible ML nuisance estimators.

---

## Method 3 — Double Machine Learning (DML)

### Motivation

IPW and AIPW treat `high_rate` as binary. DML works directly with the **continuous** `int_rate`, estimating the structural parameter `θ` in:

```
Y = θ · T + g(X) + ε
T = m(X) + v
```

where `g(X)` and `m(X)` are arbitrary functions of covariates.

### Estimation (Robinson's partialling-out)

1. Cross-fit `m(X) = E[T | X]` and `g(X) = E[Y | X]` using **XGBRegressor** (5-fold KFold).
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

- `CausalForestDML` from `econml` with `n_estimators=100`, `min_samples_leaf=5`.
- Combines DML residualisation with a causal forest for heterogeneous effect estimation.

### CATE Subgroup Analysis

| Subgroup | Direction vs. ATE | Economic reasoning |
|---|---|---|
| High-DTI (Q4) | Larger | Already stretched; a higher payment load tips them into default |
| Low-FICO (Q1) | Larger | Less financial buffer to absorb rate shock |
| 60-month loans | Larger | Longer exposure duration amplifies cumulative burden |
| Grade E–G | Larger | Near-marginal borrowers are most sensitive to rate increases |
| Grade A–B | Smaller | Financially resilient borrowers absorb rate increases |

---

## Method 5 — Sub-Grade Boundary Analysis (Local Wald Estimates)

### Intuition

At the boundary between adjacent sub-grade tiers (e.g., B5 → C1), the assigned rate jumps discontinuously while the underlying borrower risk score should be continuous. Borrowers just below vs. just above the boundary form a quasi-experimental comparison group.

### Wald Estimate

```
Wald = Δdefault / Δrate
     = (default_rate_{X1} − default_rate_{X5}) /
       (mean_rate_{X1}    − mean_rate_{X5})
```

This is a local instrumental variable analogue: the grade boundary instruments for the rate jump.

### Validity Checks

- **Rate continuity in FICO**: the FICO score difference at each boundary should be small (no sharp FICO discontinuity that would confound the comparison).
- **Wald estimates vs. AIPW ATE**: Wald estimates should be directionally consistent with the global AIPW ATE.

---

## Results Summary

| Method | Treatment | ATE (pp) | 95% CI | p-value |
|---|---|---|---|---|
| Naive difference | `high_rate` | Largest | — | — |
| IPW-ATE | `high_rate` | ~2.1 | Bootstrap | < 0.001 |
| **AIPW (doubly robust)** | **`high_rate`** | **~2.08** | **Influence function** | **< 0.001** |
| DML θ̂ (per 1 pp rate) | `int_rate` | ~0.2 pp/pp | Sandwich SE | < 0.001 |
| Causal Forest ATE | `high_rate` | ~2.1 | Forest CIs | < 0.001 |
| Sub-grade Wald | `int_rate` | Consistent | Local | — |

All methods agree on sign and approximate magnitude, providing convergent validity for the causal claim.

---

## Key Takeaways

1. **Positive, significant causal effect**: being assigned an above-median rate within one's grade increases P(default) by approximately 2 pp after doubly robust adjustment.
2. **Much smaller than the naive estimate**: grade-level confounding drives most of the raw association; the FWL decomposition showed 66% of the naive coefficient is explained by observables.
3. **Heterogeneous effects**: the treatment impact is largest for high-DTI, low-FICO borrowers — precisely the population most financially vulnerable to incremental interest burden.
4. **Boundary evidence is consistent**: local Wald estimates at grade boundaries are directionally consistent with the global AIPW estimate.
