"""
features.py
===========
Feature engineering and treatment variable construction — corresponding to
Part C of Notebook 02.

All functions operate on the cleaned DataFrame produced by ``cleaning.py``
and return a new DataFrame with additional columns.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRADE_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
SUBGRADES   = [f'{g}{n}' for g in GRADE_ORDER for n in range(1, 6)]

GRADE_MAP    = {g: i for i, g in enumerate(GRADE_ORDER)}
SUBGRADE_MAP = {sg: i for i, sg in enumerate(SUBGRADES)}

CATEGORICAL_FEATURES = [
    'home_ownership',
    'verification_status',
    'purpose',
    'application_type',
]

DEFAULT_STATUSES = [
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off',
]


# ---------------------------------------------------------------------------
# Outcome variable
# ---------------------------------------------------------------------------

def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary ``default`` column (1 = Charged Off or Default)."""
    df = df.copy()
    df['default'] = df['loan_status'].isin(DEFAULT_STATUSES).astype(int)
    return df


# ---------------------------------------------------------------------------
# Numeric transformations
# ---------------------------------------------------------------------------

def add_log_income(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``log_annual_inc = log(1 + annual_inc_cap)``."""
    df = df.copy()
    if 'annual_inc_cap' not in df.columns:
        raise ValueError("Run cleaning.handle_outliers() first to create 'annual_inc_cap'.")
    df['log_annual_inc'] = np.log1p(df['annual_inc_cap'])
    return df


# ---------------------------------------------------------------------------
# Ordinal encodings
# ---------------------------------------------------------------------------

def add_grade_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add ordinal ``grade_num`` (0–6) and ``subgrade_num`` (0–34)."""
    df = df.copy()
    df['grade_num']    = df['grade'].map(GRADE_MAP)
    df['subgrade_num'] = df['sub_grade'].map(SUBGRADE_MAP)
    return df


# ---------------------------------------------------------------------------
# Treatment variables
# ---------------------------------------------------------------------------

def add_treatment_binary(
    df: pd.DataFrame,
    group_col: str = 'grade',
    rate_col: str  = 'int_rate',
    quantile: float = 0.50,
) -> pd.DataFrame:
    """Add ``high_rate`` — above-median rate within ``group_col``.

    Parameters
    ----------
    df        : cleaned DataFrame
    group_col : grouping column for treatment definition (default: 'grade')
    rate_col  : interest rate column (default: 'int_rate')
    quantile  : threshold quantile within group (default: 0.50 = median)

    New columns
    -----------
    ``{group_col}_median_rate`` : group-level rate quantile
    ``high_rate``               : 1 if rate > group median, else 0
    """
    df = df.copy()
    medians = df.groupby(group_col)[rate_col].quantile(quantile)
    df[f'{group_col}_median_rate'] = df[group_col].map(medians)
    df['high_rate'] = (df[rate_col] > df[f'{group_col}_median_rate']).astype(int)
    return df


def add_treatment_subgrade(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``high_rate_subgrade`` — above-median rate within sub-grade.

    Used in the boundary analysis of Notebook 04.
    """
    df = df.copy()
    sg_medians = df.groupby('sub_grade')['int_rate'].median()
    df['subgrade_median_rate']  = df['sub_grade'].map(sg_medians)
    df['high_rate_subgrade']    = (df['int_rate'] > df['subgrade_median_rate']).astype(int)
    return df


def add_rate_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``rate_dev = int_rate − grade_median_rate``.

    The continuous treatment used in DML (Notebook 04).
    Requires ``add_treatment_binary()`` to have been called first.
    """
    df = df.copy()
    if 'grade_median_rate' not in df.columns:
        raise ValueError("Call add_treatment_binary() before add_rate_deviation().")
    df['rate_dev'] = df['int_rate'] - df['grade_median_rate']
    return df


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def add_dummies(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """One-hot encode categorical features (drop_first=True to avoid collinearity).

    Parameters
    ----------
    columns : list of columns to encode; defaults to CATEGORICAL_FEATURES

    Returns
    -------
    DataFrame with original categorical columns replaced by integer dummies.
    """
    df = df.copy()
    cols = [c for c in (columns or CATEGORICAL_FEATURES) if c in df.columns]
    df = pd.get_dummies(df, columns=cols, drop_first=True, dtype=int)
    return df


# ---------------------------------------------------------------------------
# Full feature engineering pipeline
# ---------------------------------------------------------------------------

def build_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the complete feature engineering pipeline.

    Steps (in order)
    ----------------
    1. Add outcome variable (``default``)
    2. Log-transform income
    3. Add grade and sub-grade ordinal encodings
    4. Add binary treatment within grade (``high_rate``)
    5. Add binary treatment within sub-grade (``high_rate_subgrade``)
    6. Add continuous treatment deviation (``rate_dev``)
    7. One-hot encode categorical features

    Parameters
    ----------
    df : cleaned DataFrame from ``cleaning.clean_accepted()``

    Returns
    -------
    DataFrame ready for causal modelling.
    """
    df = add_outcome(df)
    df = add_log_income(df)
    df = add_grade_encodings(df)
    df = add_treatment_binary(df)
    df = add_treatment_subgrade(df)
    df = add_rate_deviation(df)
    df = add_dummies(df)
    return df


# ---------------------------------------------------------------------------
# Covariate list builder
# ---------------------------------------------------------------------------

def get_covariates(df: pd.DataFrame) -> dict:
    """Return the numeric and dummy covariate lists for a modelling DataFrame.

    Returns
    -------
    dict with keys:
        'numeric'    : list of numeric covariate column names
        'cat_cols'   : list of one-hot dummy column names
        'covariates' : combined list (numeric + dummies)
    """
    numeric = [
        'loan_amnt', 'term_months', 'grade_num', 'subgrade_num',
        'fico_mid', 'dti', 'log_annual_inc', 'revol_util',
        'open_acc', 'total_acc', 'pub_rec', 'delinq_2yrs',
        'inq_last_6mths', 'ever_delinq', 'emp_length_num', 'issue_year',
    ]
    numeric = [c for c in numeric if c in df.columns]

    cat_cols = [
        c for c in df.columns
        if any(c.startswith(p + '_') for p in CATEGORICAL_FEATURES)
    ]

    return {
        'numeric'   : numeric,
        'cat_cols'  : cat_cols,
        'covariates': numeric + cat_cols,
    }
