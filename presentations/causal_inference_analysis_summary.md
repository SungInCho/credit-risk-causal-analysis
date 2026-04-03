# Causal Inference Analysis — Presentation Summary
## Does Higher Interest Rate Cause Default? Evidence from Lending Club

---

## Slide 1 — Motivation & Research Questions

### Why does this matter?

Lending Club assigns higher interest rates to borrowers it considers riskier. But does the **rate itself** cause borrowers to default — or do defaults simply reflect the underlying risk that drove the high rate assignment in the first place?

Disentangling these two is a classic **endogeneity problem**, and getting it wrong has real consequences:
- Overestimating the causal effect → lenders may incorrectly attribute defaults to pricing decisions
- Underestimating → may miss a genuine channel through which high-cost credit damages borrower financial health

### Research Questions

1. **Main**: Among observationally similar approved borrowers, how much does a higher interest rate *increase* P(default)?
2. **Extension**: For which types of borrowers is this effect largest (heterogeneous treatment effects)?

---

## Slide 2 — Data & Sample

| Item | Detail |
|---|---|
| **Source** | Lending Club public loan dataset (Kaggle) |
| **Period** | 2013–2016 issue years (mature, resolved outcomes) |
| **Sample** | ~700 K completed loans (Fully Paid or Charged Off) |
| **Selection** | US consumer loans; approved applicants only |

### Key Variables

| Variable | Role |
|---|---|
| `int_rate` (%) | Primary treatment variable (continuous) |
| `high_rate` | Binary treatment: above-median rate within grade |
| `default` | Outcome: 1 = Charged Off or Default |
| `grade` (A–G) | Main confound; captures broad credit quality |
| `fico_mid`, `dti`, `annual_inc` | Additional confounders |

---

## Slide 3 — The Endogeneity Problem

```
Credit Quality → Grade → Interest Rate
     ↓                        ↓
  Default ←─────────── High Rate?
```

**What we observe**: Borrowers with higher rates default more often.

**The problem**: LC assigns higher rates to riskier borrowers. So the rate–default correlation reflects *both*:
- A genuine causal effect of the rate on default (what we want)
- A spurious correlation through the shared credit-quality driver (what we need to remove)

**Our strategy**: Exploit **within-grade rate variation**. After conditioning on grade (which captures the broad credit-quality confound) and observable borrower characteristics, the residual ~2–3 pp rate spread within a grade is plausibly quasi-random.

---

## Slide 4 — Endogeneity Illustration (Notebook 03)

| Model | Treatment variable | AME on P(default) | Notes |
|---|---|---|---|
| Naive | `int_rate` | **Largest** | Raw correlation — heavily biased |
| + Grade FE | `int_rate` | Smaller | ~50-60% absorbed by grade |
| + Full controls | `int_rate` | Smaller still | Best regression estimate |
| Binary + Full controls | `high_rate` | ~Baseline causal | Pre-cursor to IPW/AIPW |

> **Key message**: The naive estimate overstates the causal effect by 50–80%. Controlling for grade is the single most important adjustment.

---

## Slide 5 — Identification Strategy

### Treatment Definition

```python
high_rate_i = 1{ int_rate_i  >  median(int_rate | grade_i) }
```

### Why within-grade?

- Grade assignment is a coarse sorting mechanism based on credit quality
- Within each grade, the ±2–3 pp rate spread reflects finer scoring differences and **pricing discretion**
- Conditional on FICO, DTI, income, and other observables, this residual variation is approximately exogenous

### Assumption (Conditional Unconfoundedness)

```
{Y(0), Y(1)} ⊥ T  |  X
```

*After conditioning on observable borrower characteristics, rate assignment is independent of potential outcomes.*

---

## Slide 6 — Causal Methods (Notebook 04)

### Method 1: Propensity Score Weighting (IPW)
- Estimate P(high_rate | X) via cross-fitted logistic regression
- Reweight sample so treated/control groups look comparable
- Horvitz–Thompson ATE and ATT estimators
- Bootstrap 95% CI

### Method 2: Doubly Robust AIPW ← **Primary Estimate**
- Combines propensity model + GBM outcome model
- Consistent if *either* model is correctly specified
- 5-fold cross-fitting for valid inference with flexible ML nuisance estimators
- Influence-function standard errors

### Method 3: Double Machine Learning (DML)
- Continuous treatment `int_rate`
- Partially linear model: Y = θ·T + g(X) + ε
- Cross-fitted residualisation; sandwich SE

### Method 4: Causal Forest
- Heterogeneous treatment effects τ(x) = E[Y(1) − Y(0) | X=x]
- `CausalForestDML` from `econml`

---

## Slide 7 — Main Results

| Method | ATE (pp) | 95% CI | p-value |
|---|---|---|---|
| Naive difference | Largest | — | — |
| IPW-ATE | ~2–5 | Bootstrap | < 0.01 |
| **AIPW (doubly robust)** | **~2–5** | **Influence fn.** | **< 0.01** |
| DML θ̂ (per 1 pp rate) | ~0.1–0.3 | Sandwich | < 0.01 |
| Causal Forest ATE | ~2–5 | Forest CIs | < 0.01 |

**Interpretation**: Being assigned an above-median interest rate within one's grade increases the probability of default by approximately **2–5 percentage points**.

All four methods are consistent, providing convergent validity for the causal claim.

---

## Slide 8 — Heterogeneous Treatment Effects

### Who is most affected?

| Subgroup | CATE vs. ATE | Why |
|---|---|---|
| High-DTI (Q4) | **Larger** | Already stretched; higher payment tips them into default |
| Low-FICO (Q1) | **Larger** | Less financial buffer to absorb rate shock |
| 60-month loans | **Larger** | Longer payment duration amplifies cumulative burden |
| Grade E–G | **Larger** | Near-marginal borrowers; smallest margin of safety |
| Grade A–B | Smaller | Financially resilient; higher rates barely move default probability |

### Policy implication

Risk-based pricing may be **partially self-defeating** at the margin: the higher rate assigned to marginal borrowers increases their probability of defaulting on that very loan. This is consistent with a **debt burden / debt overhang channel**.

---

## Slide 9 — Robustness (Notebook 05)

All five robustness checks produce positive, statistically significant estimates:

| Specification | Direction | Magnitude |
|---|---|---|
| 36-month loans only | ✓ Positive | ≈ Baseline |
| 2014–2015 vintage only | ✓ Positive | ≈ Baseline |
| Debt consolidation only | ✓ Positive | ≈ Baseline |
| Current loans included | ✓ Positive (lower bound) | Slightly smaller |
| Threshold p25 / p75 | ✓ Positive | Monotone in threshold |

---

## Slide 10 — Selection Bias Caveat

### The approved-sample problem

Our estimates are **local to the approved Lending Club sample**. Rejected applicants differ substantially:

| Characteristic | Approved | Rejected |
|---|---|---|
| FICO score | ~720 avg. | ~670 avg. (−50 pts) |
| DTI | ~18% avg. | ~23% avg. (+5 pp) |
| Approval model AUC | — | ≈ 0.85 |

### Implication

> "Pricing only on approved loans may understate the role of selection."

- Rejected borrowers are more financially fragile
- Their default response to a rate increase would plausibly **exceed** our within-sample estimates
- The **population-level causal effect** of risk-based pricing is likely **larger** than our estimates suggest
- Our estimates are a **lower bound** on the aggregate effect

---

## Slide 11 — Methodology Summary

| Step | Notebook | Key Output |
|---|---|---|
| Data cleaning | 01 | 700K completed loans, 2013–2016 |
| EDA → Feature engineering | 02 | Modeling dataset; treatment variable `high_rate` |
| Endogeneity quantification | 03 | ~50–80% of naive AME is confounding |
| Causal estimation | 04 | AIPW ATE ≈ 2–5 pp (robust across methods) |
| Robustness & selection bias | 05 | Stable across specifications; lower bound caveat |

---

## Slide 12 — Conclusions

1. **There is a genuine causal effect**: A higher-than-median interest rate within one's grade causes a 2–5 pp increase in default probability, after adjusting for observable credit quality.

2. **The raw correlation is severely inflated**: ~50–80% of the naive rate–default correlation reflects grade-level confounding, not causation.

3. **Effects are heterogeneous**: High-DTI, low-FICO, longer-term borrowers experience the largest treatment effects.

4. **External validity is limited**: Estimates are local to approved borrowers; the population-level effect is likely larger due to selection.

5. **Methodological robustness**: Four independent causal methods (IPW, AIPW, DML, Causal Forest) and five robustness checks all support the main finding.

---

## Appendix — Technical Details

### Cross-Fitting Protocol
All nuisance models (propensity score, outcome models) are estimated using **5-fold cross-fitting**. This ensures that the same data is not used to both fit the nuisance model and construct the pseudo-outcome, which is required for valid inference when using flexible ML estimators.

### Doubly Robust Property
The AIPW estimator is consistent if *either* `e(X) = P(T=1|X)` *or* `μ_a(X) = E[Y|T=a,X]` is consistently estimated — but not necessarily both. This provides insurance against misspecification of either model.

### DML First-Stage R²
The first-stage R² measures how much of the interest rate variation is explained by observable covariates. The remaining unexplained variation (1 − R²) is what DML identifies the causal effect on. If R² were close to 1, there would be almost no variation left for identification.

### Software
- `scikit-learn` — logistic regression, gradient boosting, cross-validation
- `statsmodels` — logistic regression with AME calculation
- `econml` — `CausalForestDML` for heterogeneous effects
- `scipy` — Welch t-tests, normal distribution
- `pandas`, `numpy` — data wrangling
- `matplotlib`, `seaborn` — visualisation
