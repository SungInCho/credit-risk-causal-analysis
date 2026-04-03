"""
cleaning.py
===========
Data loading, date filtering, type conversion, missing-value handling,
and outlier capping — corresponding to Notebook 01.

All functions are pure (no side effects) and return a new DataFrame.
The original raw files are never modified.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCEPTED_COLS = [
    'id', 'loan_amnt', 'funded_amnt', 'term', 'int_rate',
    'grade', 'sub_grade', 'emp_length', 'home_ownership',
    'annual_inc', 'verification_status', 'issue_d', 'loan_status',
    'purpose', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
    'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc', 'application_type', 'addr_state'
]

COMPLETED_STATUSES = [
    'Charged Off',
    'Default',
    'Fully Paid',
    'Does not meet the credit policy. Status:Charged Off',
    'Does not meet the credit policy. Status:Fully Paid',
]

DEFAULT_STATUSES = [
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off',
]

EMP_MAP = {
    '< 1 year': 0, '1 year': 1,  '2 years': 2,  '3 years': 3,
    '4 years':  4, '5 years': 5, '6 years': 6,  '7 years': 7,
    '8 years':  8, '9 years': 9, '10+ years': 10,
}

CRITICAL_COLS = [
    'int_rate', 'grade', 'sub_grade', 'dti', 'fico_mid',
    'annual_inc', 'loan_amnt', 'term_months', 'loan_status',
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_accepted(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load the accepted-loans CSV with a minimal memory footprint.

    Parameters
    ----------
    path : path to ``accepted_2007_to_2018Q4.csv``
    usecols : columns to read; defaults to ACCEPTED_COLS

    Returns
    -------
    DataFrame — raw, unfiltered
    """
    return pd.read_csv(
        path,
        usecols=usecols or ACCEPTED_COLS,
        low_memory=False,
    )


def load_rejected(path: str | Path) -> pd.DataFrame:
    """Load the rejected-loans CSV."""
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# Date filtering
# ---------------------------------------------------------------------------

def filter_date_range(
    df: pd.DataFrame,
    year_col: str = 'issue_d',
    start: int = 2013,
    end: int = 2016,
) -> pd.DataFrame:
    """Parse ``issue_d`` and keep rows in [start, end] inclusive.

    Also adds an ``issue_year`` integer column.
    """
    df = df.copy()
    df[year_col] = pd.to_datetime(df[year_col], format='%b-%Y', errors='coerce')
    df['issue_year'] = df[year_col].dt.year
    return df[df['issue_year'].between(start, end)].copy()


def filter_completed(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only loans with fully resolved outcomes (excludes Current, Late, etc.)."""
    return df[df['loan_status'].isin(COMPLETED_STATUSES)].copy()


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all string-to-numeric conversions in place on a copy.

    Converts:
    - int_rate   : '13.56%'   → 13.56 (float)
    - term       : '36 months' → 36   (int, new col ``term_months``)
    - revol_util : '45.2%'    → 45.2  (float, invalid → NaN)
    - emp_length  : ordinal string → int (new col ``emp_length_num``)
    - fico_mid    : midpoint of low/high range (new col ``fico_mid``)
    """
    df = df.copy()

    df['int_rate'] = (
        df['int_rate'].astype(str)
        .str.replace('%', '', regex=False).str.strip()
        .astype(float)
    )

    df['term_months'] = (
        df['term'].str.strip().str.extract(r'(\d+)')[0].astype(int)
    )

    df['revol_util'] = pd.to_numeric(
        df['revol_util'].astype(str)
        .str.replace('%', '', regex=False).str.strip(),
        errors='coerce',
    )

    df['emp_length_num'] = df['emp_length'].map(EMP_MAP)

    df['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2

    return df


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply project-specific missing-value rules.

    Rules
    -----
    mths_since_last_delinq : NaN → 999 (never delinquent sentinel);
                              also creates binary ``ever_delinq`` flag.
    emp_length_num          : NaN → -1  (n/a / self-employed).
    revol_util              : NaN → grade-level median.
    Critical columns        : drop row if any are missing.
    """
    df = df.copy()

    # Delinquency sentinel
    df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(999)
    df['ever_delinq'] = (df['mths_since_last_delinq'] < 999).astype(int)

    # Employment length n/a
    df['emp_length_num'] = df['emp_length_num'].fillna(-1)

    # Revolving utilisation — grade-level median imputation
    df['revol_util'] = df.groupby('grade')['revol_util'].transform(
        lambda x: x.fillna(x.median())
    )

    # Drop rows missing critical identification columns
    df = df.dropna(subset=[c for c in CRITICAL_COLS if c in df.columns])

    return df


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap or drop extreme values.

    Rules
    -----
    annual_inc  : cap at 99th percentile → new col ``annual_inc_cap``
    revol_util  : cap at 100 (values > 100% are data errors)
    dti         : drop rows where dti > 100
    loan_amnt   : drop rows where loan_amnt < 1000
    """
    df = df.copy()

    p99 = df['annual_inc'].quantile(0.99)
    df['annual_inc_cap'] = df['annual_inc'].clip(upper=p99)

    df['revol_util'] = df['revol_util'].clip(upper=100)

    df = df[df['dti'] <= 100].copy()
    df = df[df['loan_amnt'] >= 1000].copy()

    return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def clean_accepted(
    path: str | Path,
    start_year: int = 2013,
    end_year: int = 2016,
    completed_only: bool = True,
) -> pd.DataFrame:
    """Full cleaning pipeline for the accepted-loans file.

    Parameters
    ----------
    path          : path to raw CSV
    start_year    : earliest issue year to keep
    end_year      : latest issue year to keep
    completed_only: if True, drop Current / Late / Grace Period loans

    Returns
    -------
    Cleaned DataFrame ready for feature engineering.
    """
    df = load_accepted(path)
    df = filter_date_range(df, start=start_year, end=end_year)
    if completed_only:
        df = filter_completed(df)
    df = convert_types(df)
    df = handle_missing(df)
    df = handle_outliers(df)
    return df
