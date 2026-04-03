"""
estimators.py
=============
Causal estimators used in Notebook 04:

  - ipw_ate / ipw_att   : Horvitz–Thompson inverse-probability weighting
  - aipw_ate            : Doubly-robust AIPW with 5-fold cross-fitting
  - dml_theta           : Double Machine Learning (partially linear model)

All estimators accept numpy arrays and return named tuples for easy unpacking.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ATEResult:
    """Container for an average treatment effect estimate."""
    ate   : float
    se    : float
    ci_lo : float
    ci_hi : float
    z     : float
    p     : float

    def __str__(self) -> str:
        return (
            f"ATE = {self.ate*100:+.4f} pp  "
            f"SE = {self.se*100:.4f} pp  "
            f"95% CI [{self.ci_lo*100:.4f}, {self.ci_hi*100:.4f}]  "
            f"z = {self.z:.3f}  p = {self.p:.4f}"
        )


@dataclass
class DMLResult:
    """Container for a DML partially-linear model estimate."""
    theta         : float
    se            : float
    ci_lo         : float
    ci_hi         : float
    r2_first_stage: float

    def __str__(self) -> str:
        return (
            f"θ̂ = {self.theta*100:+.6f} pp/pp  "
            f"SE = {self.se*100:.6f}  "
            f"95% CI [{self.ci_lo*100:+.5f}, {self.ci_hi*100:+.5f}]  "
            f"First-stage R² = {self.r2_first_stage:.4f}"
        )


# ---------------------------------------------------------------------------
# Helper: default nuisance pipelines
# ---------------------------------------------------------------------------

def _default_ps_pipe() -> Pipeline:
    """Logistic regression propensity score model."""
    return Pipeline([
        ('sc',  StandardScaler()),
        ('lr',  LogisticRegression(C=0.5, max_iter=500, solver='lbfgs')),
    ])


def _default_outcome_pipe() -> Pipeline:
    """Gradient-boosted classifier outcome model."""
    return Pipeline([
        ('sc',  StandardScaler()),
        ('gbm', GradientBoostingClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, subsample=0.8, random_state=42,
        )),
    ])


def _default_reg_pipe() -> Pipeline:
    """Gradient-boosted regressor for continuous outcomes/treatments (DML)."""
    return Pipeline([
        ('sc',  StandardScaler()),
        ('gbm', GradientBoostingRegressor(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, subsample=0.8, random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# IPW
# ---------------------------------------------------------------------------

def ipw_ate(
    y: np.ndarray,
    T: np.ndarray,
    ps: np.ndarray,
    clip: tuple[float, float] = (0.01, 0.99),
) -> float:
    """Horvitz–Thompson IPW estimator of the ATE.

    Parameters
    ----------
    y   : (n,) binary outcome
    T   : (n,) binary treatment
    ps  : (n,) propensity scores P(T=1 | X)
    clip: (lo, hi) bounds for propensity score trimming

    Returns
    -------
    ATE estimate (float)
    """
    ps = np.clip(ps, *clip)
    return float(np.mean(T * y / ps) - np.mean((1 - T) * y / (1 - ps)))


def ipw_att(
    y: np.ndarray,
    T: np.ndarray,
    ps: np.ndarray,
    clip: tuple[float, float] = (0.01, 0.99),
) -> float:
    """Horvitz–Thompson IPW estimator of the ATT."""
    ps = np.clip(ps, *clip)
    return float(
        np.sum(T * y) / np.sum(T)
        - np.sum((1 - T) * y * ps / (1 - ps)) / np.sum((1 - T) * ps / (1 - ps))
    )


def ipw_bootstrap_ci(
    y: np.ndarray,
    T: np.ndarray,
    ps: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
    clip: tuple[float, float] = (0.01, 0.99),
) -> tuple[float, float]:
    """Non-parametric bootstrap confidence interval for IPW-ATE."""
    rng = np.random.default_rng(random_state)
    n   = len(y)
    boot = [
        ipw_ate(y[idx := rng.integers(0, n, n)], T[idx], ps[idx], clip)
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# AIPW (doubly robust)
# ---------------------------------------------------------------------------

def aipw_ate(
    y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    n_splits: int = 5,
    ps_model: Pipeline | None = None,
    outcome_model_factory: Callable[[], Pipeline] | None = None,
    random_state: int = 42,
    ps_clip: tuple[float, float] = (0.01, 0.99),
) -> ATEResult:
    """Cross-fitted AIPW ATE estimator (doubly robust).

    Parameters
    ----------
    y                     : (n,) binary outcome
    T                     : (n,) binary treatment
    X                     : (n, p) pre-treatment covariates
    n_splits              : number of cross-fitting folds (default: 5)
    ps_model              : sklearn Pipeline for propensity score; if None,
                            uses logistic regression
    outcome_model_factory : callable returning a fresh sklearn Pipeline for
                            the outcome model; if None, uses GBM classifier
    random_state          : random seed
    ps_clip               : (lo, hi) propensity score clipping bounds

    Returns
    -------
    ATEResult dataclass with ate, se, ci_lo, ci_hi, z, p
    """
    from scipy.stats import norm

    cv    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    ps_cf = np.zeros(len(y))
    mu1   = np.zeros(len(y))
    mu0   = np.zeros(len(y))

    ps_pipe = clone(ps_model) if ps_model is not None else _default_ps_pipe()
    out_fac = outcome_model_factory if outcome_model_factory is not None else _default_outcome_pipe

    for tr_idx, te_idx in cv.split(X, T):
        X_tr, X_te = X[tr_idx], X[te_idx]
        T_tr, y_tr = T[tr_idx], y[tr_idx]

        # Propensity score
        ps_pipe.fit(X_tr, T_tr)
        ps_cf[te_idx] = np.clip(ps_pipe.predict_proba(X_te)[:, 1], *ps_clip)

        # Outcome models per arm
        for arm, store in [(1, mu1), (0, mu0)]:
            idx = T_tr == arm
            m   = out_fac()
            if idx.sum() > 10:
                m.fit(X_tr[idx], y_tr[idx])
                store[te_idx] = m.predict_proba(X_te)[:, 1]
            else:
                store[te_idx] = y_tr[idx].mean() if idx.sum() > 0 else 0.0

    # AIPW pseudo-outcome
    psi = (
        (mu1 - mu0)
        + T * (y - mu1) / ps_cf
        - (1 - T) * (y - mu0) / (1 - ps_cf)
    )

    ate   = psi.mean()
    se    = psi.std() / np.sqrt(len(psi))
    ci_lo = ate - 1.96 * se
    ci_hi = ate + 1.96 * se
    z     = ate / se
    p     = float(2 * (1 - norm.cdf(abs(z))))

    return ATEResult(ate=float(ate), se=float(se),
                     ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                     z=float(z), p=p)


# ---------------------------------------------------------------------------
# Double Machine Learning (partially linear model)
# ---------------------------------------------------------------------------

def dml_theta(
    y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    n_splits: int = 5,
    model_y_factory: Callable[[], Pipeline] | None = None,
    model_t_factory: Callable[[], Pipeline] | None = None,
    random_state: int = 42,
) -> DMLResult:
    """Double Machine Learning estimator for a partially linear model.

    Model: Y = θ·T + g(X) + ε,  T = m(X) + v

    Parameters
    ----------
    y               : (n,) outcome (can be continuous or binary)
    T               : (n,) continuous or binary treatment
    X               : (n, p) covariates
    n_splits        : cross-fitting folds
    model_y_factory : callable → sklearn regressor pipeline for E[Y|X]
    model_t_factory : callable → sklearn regressor pipeline for E[T|X]
    random_state    : random seed

    Returns
    -------
    DMLResult with theta, se, ci_lo, ci_hi, r2_first_stage
    """
    cv       = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_resid  = np.zeros(len(y))
    T_resid  = np.zeros(len(y))

    my_fac = model_y_factory if model_y_factory is not None else _default_reg_pipe
    mt_fac = model_t_factory if model_t_factory is not None else _default_reg_pipe

    for tr_idx, te_idx in cv.split(X):
        m_y = my_fac(); m_y.fit(X[tr_idx], y[tr_idx])
        y_resid[te_idx] = y[te_idx] - m_y.predict(X[te_idx])

        m_t = mt_fac(); m_t.fit(X[tr_idx], T[tr_idx])
        T_resid[te_idx] = T[te_idx] - m_t.predict(X[te_idx])

    # OLS on residuals (no intercept)
    theta = float((T_resid @ y_resid) / (T_resid @ T_resid))

    # Sandwich standard error
    eps = y_resid - theta * T_resid
    se  = float(np.sqrt(
        np.sum(T_resid**2 * eps**2) / (np.sum(T_resid**2))**2
    ))

    r2_first = float(1 - np.var(T_resid) / np.var(T))

    return DMLResult(
        theta          = theta,
        se             = se,
        ci_lo          = theta - 1.96 * se,
        ci_hi          = theta + 1.96 * se,
        r2_first_stage = r2_first,
    )
