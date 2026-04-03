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
| **Sample** | 1,025,917 completed loans (Fully Paid or Charged Off/Default) |
| **Default rate** | 20.09% |
| **Selection** | US consumer loans; approved applicants only |

### Key Variables

| Variable | Role |
|---|---|
| `int_rate` (%) | Continuous treatment variable (used in DML) |
| `high_rate` | Binary treatment: above-median rate within grade |
| `default` | Outcome: 1 = Charged Off or Default |
| `grade_num` (0–6) | Main confound; captures broad credit quality |
| `fico_mid`, `dti`, `log_annual_inc` | Additional confounders |

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

| Model | Treatment variable | AME on P(default) | Pseudo R² |
|---|---|---|---|
| Naive | `int_rate` | **+2.0825 pp** | 0.0678 |
| + Grade control | `int_rate` | **+0.8141 pp** | 0.0700 |
| + Full controls (23 vars) | `int_rate` | **+0.8484 pp** | 0.0988 |
| Binary + Full controls | `high_rate` | **+1.9328 pp** | 0.0985 |

> **Key message**: The naive estimate overstates the causal effect by ~61% just from adding the grade control. The FWL decomposition shows that **66% of the raw association is explained by observable controls**, leaving a residual effect that motivates the causal methods.

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

### Supporting Evidence

- **CV AUC gain from `high_rate`**: +0.0009 — the within-grade treatment indicator adds virtually no predictive power beyond observables.
- **Pre-weighting SMD** < 0.1 for most covariates within grade, confirming near-overlap.

---

## Slide 6 — Causal Methods (Notebook 04)

### Nuisance Models

- **Propensity score**: logistic regression (C=0.5), scores clipped [0.01, 0.99]
- **Outcome model**: XGBClassifier (n_estimators=200, max_depth=4, learning_rate=0.05)
- All models use **5-fold StratifiedKFold cross-fitting**

### Method 1: Propensity Score Weighting (IPW)
- Horvitz–Thompson ATE and ATT estimators
- Bootstrap 95% CI (500 replicates)

### Method 2: Doubly Robust AIPW ← **Primary Estimate**
- Combines propensity model + XGBClassifier outcome model
- Consistent if *either* model is correctly specified
- 5-fold cross-fitting for valid semiparametric inference
- Influence-function standard errors

### Method 3: Double Machine Learning (DML)
- Continuous treatment `int_rate`
- Partially linear model: Y = θ·T + g(X) + ε
- XGBRegressor for E[Y|X] and E[T|X]; cross-fitted residualisation; sandwich SE

### Method 4: Causal Forest
- Heterogeneous treatment effects τ(x) = E[Y(1) − Y(0) | X=x]
- `CausalForestDML` from `econml` (n_estimators=100, min_samples_leaf=5)

---

## Slide 7 — Main Results

| Method | ATE (pp) | 95% CI | p-value |
|---|---|---|---|
| Naive difference | Largest | — | — |
| IPW-ATE | ~2.1 | Bootstrap | < 0.001 |
| **AIPW (doubly robust)** | **~2.08** | **Influence fn.** | **< 0.001** |
| DML θ̂ (per 1 pp rate) | ~0.2 pp/pp | Sandwich | < 0.001 |
| Causal Forest ATE | ~2.1 | Forest CIs | < 0.001 |

**Interpretation**: Being assigned an above-median interest rate within one's grade increases the probability of default by approximately **2 percentage points**.

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

All robustness checks produce positive, statistically significant estimates:

| Specification | N | ATE (pp) | 95% CI |
|---|---|---|---|
| 36-month loans only | 777,875 | +2.2608 | [2.09, 2.43] |
| 2014–2015 vintage only | 598,375 | +2.0372 | [1.79, 2.28] |
| Debt-related purposes | 842,797 | +2.1417 | [1.97, 2.32] |
| Other purposes | 183,120 | +1.7391 | [1.37, 2.11] |
| Threshold q=0.25 | 1,025,917 | +2.1052 | — |
| Threshold q=0.75 | 1,025,917 | +1.4665 | — (monotone) |

---

## Slide 10 — Selection Bias Caveat

### The approved-sample problem

Our estimates are **local to the approved Lending Club sample**. Rejected applicants differ substantially:

| Characteristic | Approved | Rejected |
|---|---|---|
| FICO score | ~720 avg. | ~670 avg. (−50 pts) |
| DTI | ~18% avg. | ~23% avg. (+5 pp) |
| Approval model AUC | — | **0.9215** |
| Dominant approval predictor | — | DTI (coef = −85.1) |

### Implication

> "Our estimates are a lower bound — the population-level causal effect of risk-based pricing is likely larger."

- Rejected borrowers are more financially fragile
- Their default response to a rate increase would plausibly **exceed** our within-sample estimates
- The high approval AUC (0.9215) confirms that observables nearly perfectly determine approval

---

## Slide 11 — Methodology Summary

| Step | Notebook | Key Output |
|---|---|---|
| Data cleaning | 01 | 1,025,982 completed loans, 2013–2016 |
| EDA → Feature engineering | 02 | Modeling dataset (1,025,917 rows, 29 cols); `high_rate` treatment; 24 covariates (VIF screen excludes `subgrade_num`) |
| Endogeneity quantification | 03 | 66% of naive AME is confounding; FWL residualized coef = 0.005806 |
| Causal estimation | 04 | AIPW ATE ≈ 2.08 pp (robust across four methods) |
| Robustness & selection bias | 05 | Stable across 6 specifications; AUC=0.9215; DTI dominates approval |

---

## Slide 12 — Conclusions

1. **There is a genuine causal effect**: A higher-than-median interest rate within one's grade causes approximately a 2 pp increase in default probability, after adjusting for observable credit quality.

2. **The raw correlation is substantially inflated**: 66% of the naive rate–default association reflects observable confounding, not causation. Grade alone explains ~61% of the naive AME.

3. **Effects are heterogeneous**: High-DTI, low-FICO, longer-term borrowers experience the largest treatment effects.

4. **External validity is limited**: Estimates are local to approved borrowers; the population-level effect is likely larger (approval AUC = 0.9215 confirms strong selection on observables).

5. **Methodological robustness**: Four independent causal methods (IPW, AIPW, DML, Causal Forest) and multiple robustness checks all support the main finding.

---

## Appendix — Technical Details

### VIF Screening
`subgrade_num` has VIF = 53.6, indicating near-perfect collinearity with `grade_num` and `int_rate`. It is excluded from all regression and ML covariate sets; 23 covariates are used in the final models.

### Cross-Fitting Protocol
All nuisance models (propensity score, outcome models) are estimated using **5-fold StratifiedKFold cross-fitting**. This ensures that the same data is not used to both fit the nuisance model and construct the pseudo-outcome, which is required for valid inference when using flexible ML estimators.

### Doubly Robust Property
The AIPW estimator is consistent if *either* `e(X) = P(T=1|X)` *or* `μ_a(X) = E[Y|T=a,X]` is consistently estimated — but not necessarily both. This provides insurance against misspecification of either model.

### DML First-Stage R²
The first-stage R² measures how much of the interest rate variation is explained by observable covariates. The remaining unexplained variation (1 − R²) is what DML identifies the causal effect on. If R² were close to 1, there would be almost no variation left for identification.

### Software
- `xgboost` — XGBClassifier / XGBRegressor as nuisance estimators
- `scikit-learn` — logistic regression, cross-validation, VIF screening
- `statsmodels` — logistic regression with AME calculation
- `econml` — `CausalForestDML` for heterogeneous effects
- `scipy` — Welch t-tests, normal distribution
- `pandas`, `numpy` — data wrangling
- `matplotlib`, `seaborn` — visualisation
