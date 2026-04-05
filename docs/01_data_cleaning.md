# 01. Data Cleaning

## Overview

This notebook prepares the raw Lending Club loan data for analysis through four cleaning steps: date filtering, data type conversion, missing value handling, and outlier treatment. Both accepted and rejected loan datasets are processed. Feature engineering and treatment variable construction are deferred to Notebook 02 after exploratory analysis.

**Input**: `accepted_2007_to_2018Q4.csv`, `rejected_2007_to_2018Q4.csv`
**Output**: `accepted_cleaned.parquet`, `accepted_with_current.parquet`, `rejected_cleaned.parquet`

---

## Part A: Accepted Loan Data

### Raw Data

- **Shape**: 2,260,701 rows x 27 selected columns
- **Selected columns**: `id`, `loan_amnt`, `funded_amnt`, `term`, `int_rate`, `grade`, `sub_grade`, `emp_length`, `home_ownership`, `annual_inc`, `verification_status`, `issue_d`, `loan_status`, `purpose`, `dti`, `delinq_2yrs`, `fico_range_low`, `fico_range_high`, `inq_last_6mths`, `mths_since_last_delinq`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `application_type`, `addr_state`

---

### Step 1 — Date Filtering (2013-2016)

**Rationale**:
- 36-month loans issued through 2016 matured by early 2019, well before the 2018Q4 data cutoff, so outcomes are largely resolved.
- 2017+ loans still labeled "Current" introduce censoring bias and downward-biased default rates.
- Pre-2013 loans have smaller volume and structurally different market conditions.

**Result**: 1,225,945 loans after 2013-2016 filter.

A copy including "Current" loans is preserved for robustness checks in Notebook 05. The main analysis retains only completed loans:

| Status | Count |
|---|---|
| Fully Paid | 820,316 |
| Charged Off | 206,230 |
| Default | 12 |
| **Total (completed)** | **1,026,558** |

---

### Step 2 — Data Type Conversion

| Original Field | Transformation | New Field |
|---|---|---|
| `term` ("36 months") | Extract integer via regex | `term_months` (36 or 60) |
| `emp_length` ("< 1 year", ..., "10+ years") | Ordinal map 0-10 | `emp_length_num` |
| `fico_range_low` / `fico_range_high` | Midpoint average | `fico_mid` |

---

### Step 3 — Missing Value Handling

| Variable | Missing Count | Strategy |
|---|---|---|
| `mths_since_last_delinq` | 506,129 | Fill with 999 (never delinquent); create binary `ever_delinq` flag |
| `emp_length` / `emp_length_num` | 58,454 | Fill with "Unknown" / -1 to distinguish from "< 1 year" (0) |
| `revol_util`, `dti`, `inq_last_6mths` | 576 total | Drop rows (trivial count) |

**After handling**: 1,025,982 rows with zero missing values.

---

### Step 4 — Outlier Handling

| Variable | Issue | Treatment |
|---|---|---|
| `annual_inc` | Max = $9,550,000 (data-entry errors) | Cap at 99th percentile ($250,000) -> `annual_inc_cap` |
| `revol_util` | Some entries > 100% | Clip at 100% |
| `dti` | Some entries > 100% | Clip at 100% |

---

### Sample Flow

| Step | Rows |
|---|---|
| Raw accepted CSV | 2,260,701 |
| After date filter (2013-2016) | 1,225,945 |
| After completed-loan filter | 1,026,558 |
| After dropping missing values | **1,025,982** |

---

## Part B: Rejected Loan Data

### Raw Data

- **Shape**: 27,648,741 rows x 9 columns

### Processing

1. **Date filtering**: Restrict to 2013-2016 application dates
2. **Type conversion**: Parse `Debt-To-Income Ratio` (percentage string to numeric), map `Employment Length` to ordinal, extract `Amount Requested` and `Risk_Score`

**Result**: 10,323,895 rejected loan records

---

## Output Summary

| Dataset | Shape | Description |
|---|---|---|
| `accepted_cleaned.parquet` | (1,025,982 x 33) | Completed loans, 2013-2016, outliers handled |
| `accepted_with_current.parquet` | (1,225,945 x 28) | Includes "Current" loans for robustness checks |
| `rejected_cleaned.parquet` | (10,323,895 x 14) | Rejected loans, 2013-2016, basic cleaning |
