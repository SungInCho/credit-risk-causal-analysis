# Final Summary — Credit Risk Causal Analysis

## Research Question

Among observationally similar approved Lending Club borrowers, does a higher interest rate **cause** higher default probability — or does the rate-default correlation simply reflect the underlying credit risk that drove the high rate assignment?

---

## Project Flow

```
01 Data Cleaning  -->  02 EDA & Features  -->  03 Baseline Models  -->  04 Causal Models  -->  05 Robustness
   (raw -> clean)      (clean -> model-ready)   (quantify bias)        (estimate effects)     (validate)
```

---

## Step 1: Data Cleaning (Notebook 01)

Prepared 2013-2016 Lending Club loan data from two raw CSV files.

| Dataset | Final Size |
|---|---|
| Accepted (completed loans) | 1,025,982 rows x 33 columns |
| Accepted (with Current loans) | 1,225,945 rows x 28 columns |
| Rejected | 10,323,895 rows x 14 columns |

**Key decisions**: Restricted to 2013-2016 for resolved outcomes. Capped `annual_inc` at 99th percentile. Created `fico_mid` and `ever_delinq` derived features.

---

## Step 2: EDA and Feature Engineering (Notebook 02)

### Key EDA Findings

- **Default rate**: 20.09% overall, ranging from 5.9% (Grade A) to 51.2% (Grade G).
- **Grade is the dominant confound**: raw Pearson r(int_rate, default) = 0.27, but most of this reflects grade-level sorting.
- **Within-grade overlap is good**: SMD < 0.2 for all covariates between high-rate and low-rate borrowers within each grade.
- **60-month loans default at 33.6%** vs. 15.8% for 36-month loans.

### Feature Selection

From 13 numeric candidates, selected 10 features via variance filtering, correlation analysis, and GBM importance ranking. Excluded `delinq_2yrs`, `mths_since_last_delinq`, `pub_rec` (low importance). Added 9 one-hot categorical dummies, 2 ordinal features, 1 boolean, and 2 grade encodings for 24 total covariates.

### Treatment Construction

**Primary treatment**: `high_rate` = 1 if interest rate exceeds the within-grade median. This exploits the 2-3 pp rate spread within each grade as quasi-random variation after conditioning on observables.

**Output**: Modeling dataset of 1,025,917 rows x 29 columns.

---

## Step 3: Baseline Models (Notebook 03)

Progressively controlled logistic regressions quantified endogeneity:

| Model | Treatment | AME (pp) | Shrinkage |
|---|---|---|---|
| Naive | int_rate | +2.0825 | — |
| + Grade | int_rate | +0.8141 | -60.9% |
| + Full controls (23 vars) | int_rate | +0.8433 | -59.5% |
| Binary + Full controls | high_rate | +1.9328 | — |

**Frisch-Waugh-Lovell decomposition**: Observable controls explain **66% of the naive coefficient**. The remaining 34% motivates formal causal methods.

**Cross-validated AUC gain from `high_rate`**: +0.0009 — the treatment indicator adds virtually no predictive power beyond observables, supporting the conditional unconfoundedness assumption.

---

## Step 4: Causal Estimation (Notebook 04)

Five methods applied, all finding positive, significant effects:

| Method | Treatment | ATE (pp) | 95% CI |
|---|---|---|---|
| IPW-ATE | high_rate | +2.4473 | [2.281, 2.615] |
| **AIPW (primary)** | **high_rate** | **+2.0800** | **[1.922, 2.238]** |
| DML | int_rate | +1.1532 per pp | [1.083, 1.224] |
| Causal Forest | high_rate | +1.8027 | — |
| Sub-grade Wald (A-D) | int_rate | +2-3 pp jumps | p < 0.001 |

### Treatment Effect Heterogeneity (Causal Forest)

- **By grade**: Larger effects for lower-risk grades (A-C), smaller for higher-risk grades (D-G). For already risky borrowers, default is driven by broader factors; for safer borrowers, the additional rate burden has a more direct impact.
- **By term**: 36-month loans (CATE = 2.06 pp) > 60-month loans (CATE = 1.00 pp).
- **By DTI/FICO**: Limited additional heterogeneity beyond what grade captures.

### Sub-Grade Boundary Analysis

Rate jumps at grade boundaries (A5->B1, B5->C1, etc.) produce significant default increases for grades A-D, but the effect diminishes for E-G — consistent with the Causal Forest pattern.

---

## Step 5: Robustness (Notebook 05)

The AIPW ATE is stable across all specifications:

| Specification | ATE (pp) | 95% CI |
|---|---|---|
| Baseline | +2.084 | [1.925, 2.242] |
| 36-month only | +2.261 | [2.090, 2.432] |
| 2014-2015 vintage | +2.037 | [1.794, 2.280] |
| Debt purposes | +2.142 | [1.966, 2.317] |
| Other purposes | +1.739 | [1.373, 2.106] |
| Threshold q=0.25 to q=0.75 | +1.467 to +2.168 | Monotone decline |

### Selection Bias

An approval model achieves AUC = 0.9215, confirming that observables nearly perfectly determine approval. Rejected borrowers have lower FICO (~670 vs. ~720) and higher DTI (~23% vs. ~18%). Our estimates are therefore a **lower bound** on the population-level effect.

---

## Conclusions

1. **There is a genuine causal effect**: An above-median interest rate within grade increases default probability by approximately **2 percentage points** after doubly robust adjustment.

2. **The naive correlation is substantially inflated**: 66% of the raw rate-default association reflects observable confounding, not causation.

3. **Effects are heterogeneous**: Stronger in lower-risk grades (A-C) and for shorter-term loans. DTI and FICO add limited heterogeneity beyond grade.

4. **External validity is limited**: Estimates apply to approved borrowers only. The population-level effect is likely larger.

5. **Methodological convergence**: Four independent causal methods and six robustness specifications all support the main finding.

---

## Limitations

| Limitation | Impact |
|---|---|
| Unobserved lender information | Proprietary data not in public dataset may still correlate with within-grade rate assignment |
| Sample restricted to resolved loans | Dropping Current/Late loans may introduce survivorship bias |
| SUTVA | If borrower default depends on others' rates, the stable unit treatment value assumption is violated |
| No randomization | Conditional independence cannot be tested and may fail for unobserved confounders |
