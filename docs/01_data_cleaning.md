# 01 — Data Cleaning

**Notebook**: [`notebooks/01_data_cleaning.ipynb`](../notebooks/01_data_cleaning.ipynb)
**Inputs**: `data/raw/accepted_2007_to_2018Q4.csv`, `data/raw/rejected_2007_to_2018Q4.csv`
**Outputs**: `outputs/intermediate/accepted_cleaned.parquet`, `accepted_with_current.parquet`, `rejected_cleaned.parquet`

---

## Scope

This notebook covers exactly four cleaning steps and nothing more. Feature engineering and treatment variable construction are deferred to Notebook 02, after exploratory analysis has informed those decisions.

---

## Step 1 — Date Filtering (2013–2016)

**Why 2013–2016?**

The core identification strategy requires loans with *resolved* outcomes (Fully Paid or Charged Off). The data has a cut-off of 2018 Q4.

- A 36-month loan issued in December 2016 matures in December 2019 — so most of the balance in the 2013–2016 cohort has already resolved by the cut-off.
- Loans issued in 2017–2018 are frequently still labeled *Current* at the cut-off, making their outcomes censored. Including them would downward-bias observed default rates.
- Pre-2013 loans reflect a structurally different post-GFC market and are excluded for homogeneity.

**Implementation**:
- Parse `issue_d` (e.g. `"Jan-2015"`) to a `datetime` object.
- Extract `issue_year`.
- Keep rows where `2013 ≤ issue_year ≤ 2016`.
- Separately save a copy that retains *Current* loans (`accepted_with_current.parquet`) for the robustness check in Notebook 05.
- For the main analysis, keep only completed loan statuses: *Fully Paid*, *Charged Off*, *Default*, and the two "does not meet credit policy" variants.

---

## Step 2 — Data Type Conversion

| Column | Raw format | Converted to | Notes |
|---|---|---|---|
| `int_rate` | `"13.56%"` (string) | `float` | Strip `%` and whitespace |
| `term` | `"36 months"` | `int` (`term_months`) | Extract first integer with regex |
| `revol_util` | `"45.2%"` (string) | `float` | Strip `%`; non-numeric → `NaN` |
| `emp_length` | `"3 years"` etc. | `int` (`emp_length_num`) | Ordinal map: `< 1 year` → 0, `10+ years` → 10 |
| `fico_range_low/high` | Two integers | `float` (`fico_mid`) | Midpoint of the reported range |

---

## Step 3 — Missing Value Handling

| Column | Missing reason | Treatment |
|---|---|---|
| `mths_since_last_delinq` | Never delinquent | Fill with `999` (sentinel); also create binary `ever_delinq` flag |
| `emp_length_num` | "n/a" / self-employed | Fill with `-1` to distinguish from "< 1 year" (0) |
| `revol_util` | ~0.3% missing | Fill with **grade-level median** |
| Critical columns (`int_rate`, `grade`, `dti`, `fico_mid`, `annual_inc`, `loan_amnt`, `term_months`, `loan_status`) | Data error | **Drop row** |

Critical columns are defined as those required for both outcome construction and basic covariate control. Less than 0.5% of rows are affected.

---

## Step 4 — Outlier Handling

| Column | Issue | Treatment |
|---|---|---|
| `annual_inc` | A small number of entries exceed $10 M — clear data-entry errors | **Cap at the 99th percentile** (`annual_inc_cap`); original column retained |
| `revol_util` | A handful of values > 100% (data errors) | **Cap at 100%** |
| `dti` | Values > 100 are implausible for consumer loans | **Drop row** |
| `loan_amnt` | A few rows have `$0` (Lending Club minimum is `$1,000`) | **Drop row** |

All decisions are logged with before/after counts printed in the notebook.

---

## Outputs Summary

| File | Rows (approx.) | Description |
|---|---|---|
| `accepted_cleaned.parquet` | ~700 K | Completed loans, 2013–2016, cleaned |
| `accepted_with_current.parquet` | ~800 K | Same but includes Current loans |
| `rejected_cleaned.parquet` | ~8 M | Rejected applications, 2013–2016, basic cleaning |

---

## What Is NOT Done Here

- No feature engineering (log-transforms, encodings, interaction terms)
- No treatment or outcome variable creation
- No train/test split
- No imputation beyond the simple rules above

All of the above are handled in Notebook 02 after EDA.
