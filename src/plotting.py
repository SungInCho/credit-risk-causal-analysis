"""
plotting.py
===========
Shared visualisation helpers used across all notebooks.

Conventions
-----------
- All functions accept a Matplotlib ``ax`` parameter; if None, a new figure
  is created and returned.
- Functions return (fig, ax) so callers can further customise or save.
- ``save_fig(fig, path)`` saves at 150 dpi with tight bounding box.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

GRADE_ORDER   = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
GRADE_PALETTE = sns.color_palette('RdYlGn_r', n_colors=7)


def set_style() -> None:
    """Apply the project-wide Seaborn style."""
    sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save a figure to ``path`` at the given dpi."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')


def _make_ax(figsize=(12, 5)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_grade_boxplot(
    df: pd.DataFrame,
    col: str = 'int_rate',
    ylabel: str = 'Interest Rate (%)',
    title: str = 'Distribution by Grade',
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Box-plot of ``col`` stratified by LC grade."""
    fig, ax = (ax.get_figure(), ax) if ax else _make_ax()

    grade_data = [df[df['grade'] == g][col].values for g in GRADE_ORDER]
    bp = ax.boxplot(
        grade_data, labels=GRADE_ORDER, patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markersize=2, alpha=0.3),
    )
    for patch, color in zip(bp['boxes'], GRADE_PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xlabel('Grade', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    return fig, ax


def plot_default_by_grade(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of default rates by grade with 95% CI error bars."""
    fig, ax = (ax.get_figure(), ax) if ax else _make_ax()

    stats = (
        df.groupby('grade')
        .agg(n=('default', 'count'), dr=('default', 'mean'),
             se=('default', lambda x: x.std() / np.sqrt(len(x))))
        .loc[[g for g in GRADE_ORDER if g in df['grade'].unique()]]
    )

    ax.bar(stats.index, stats['dr'] * 100,
           yerr=stats['se'] * 100 * 1.96,
           color=GRADE_PALETTE[:len(stats)], alpha=0.85,
           capsize=5, edgecolor='white')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel('Grade', fontsize=13)
    ax.set_ylabel('Default Rate (%)', fontsize=13)
    ax.set_title('Default Rate by Grade  (error bars = 95% CI)', fontsize=14)
    return fig, ax


def plot_rate_default_raw(
    df: pd.DataFrame,
    bin_width: float = 1.0,
    min_n: int = 100,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Binned scatter: interest rate vs. default rate with loan-count bars."""
    fig, ax = (ax.get_figure(), ax) if ax else _make_ax()
    ax2 = ax.twinx()

    bins   = np.arange(5, 32, bin_width)
    labels = pd.cut(df['int_rate'], bins=bins, right=False)
    binned = (
        df.assign(_b=labels)
        .groupby('_b', observed=True)
        .agg(n=('default', 'count'), dr=('default', 'mean'))
        .reset_index()
    )
    binned['mid'] = binned['_b'].apply(lambda x: x.mid)
    binned = binned[binned['n'] >= min_n]

    ax2.bar(binned['mid'], binned['n'], width=bin_width * 0.8,
            color='lightgrey', alpha=0.5, label='Loan Count')
    ax2.set_ylabel('Loan Count', color='grey', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='grey')

    ax.plot(binned['mid'], binned['dr'] * 100,
            color='crimson', linewidth=2.5, marker='o', markersize=5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel('Interest Rate (%)  — 1 pp bins', fontsize=13)
    ax.set_ylabel('Default Rate (%)', color='crimson', fontsize=13)
    ax.tick_params(axis='y', labelcolor='crimson')
    ax.set_title('Raw Relationship: Interest Rate vs. Default Rate\n'
                 '(confounded by grade assignment)', fontsize=14)
    return fig, ax


# ---------------------------------------------------------------------------
# Causal model plots
# ---------------------------------------------------------------------------

def plot_ps_overlap(
    ps: np.ndarray,
    T: np.ndarray,
    trim_lo: float = 0.1,
    trim_hi: float = 0.9,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Overlapping histograms of propensity scores by treatment group."""
    fig, ax = (ax.get_figure(), ax) if ax else _make_ax()

    for t, color, lbl in [(1, 'crimson', 'High Rate (T=1)'),
                          (0, 'steelblue', 'Low Rate (T=0)')]:
        ax.hist(ps[T == t], bins=60, alpha=0.5, color=color, density=True, label=lbl)
    ax.axvline(trim_lo, color='grey', linestyle='--', linewidth=1)
    ax.axvline(trim_hi, color='grey', linestyle='--', linewidth=1,
               label=f'Trim bounds ({trim_lo}, {trim_hi})')
    ax.set_xlabel('P(high_rate | X)', fontsize=12)
    ax.set_title('Propensity Score Overlap', fontsize=12)
    ax.legend()
    return fig, ax


def plot_balance(
    balance_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of |SMD| before and after IPW weighting.

    Parameters
    ----------
    balance_df : DataFrame with columns Variable, SMD_Unweighted, SMD_IPW
    """
    fig, ax = (ax.get_figure(), ax) if ax else _make_ax((9, 5))

    x = np.arange(len(balance_df))
    ax.barh(balance_df['Variable'], np.abs(balance_df['SMD_Unweighted']),
            height=0.35, label='Unweighted', color='steelblue', alpha=0.7)
    ax.barh(x - 0.38, np.abs(balance_df['SMD_IPW']),
            height=0.35, label='IPW Weighted', color='crimson', alpha=0.7)
    ax.axvline(0.1, color='orange', linestyle='--', linewidth=1.5,
               label='|SMD| = 0.1 threshold')
    ax.set_xlabel('|Standardized Mean Difference|', fontsize=12)
    ax.set_title('Covariate Balance Before / After IPW', fontsize=12)
    ax.legend()
    return fig, ax


def plot_dml_residuals(
    T_resid: np.ndarray,
    y_resid: np.ndarray,
    theta: float,
    n_bins: int = 30,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter of DML residuals with the estimated slope overlaid."""
    fig, ax = (ax.get_figure(), ax) if ax else _make_ax((8, 5))

    T_df  = pd.DataFrame({'T': T_resid, 'y': y_resid})
    T_df['bin'] = pd.qcut(T_df['T'], q=n_bins, duplicates='drop')
    agg   = T_df.groupby('bin', observed=True).agg(
        T_mid=('T', 'mean'), y_mean=('y', 'mean'), n=('y', 'count')
    ).reset_index()

    x_line = np.linspace(T_resid.min(), T_resid.max(), 100)

    ax.scatter(agg['T_mid'], agg['y_mean'] * 100,
               s=agg['n'] / 100, alpha=0.7, color='steelblue')
    ax.plot(x_line, theta * x_line * 100, color='crimson', linewidth=2.5,
            label=f'DML θ̂ = {theta*100:.4f} pp/pp')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Residualised int_rate  [T − E(T|X)]', fontsize=12)
    ax.set_ylabel('Residualised default  [Y − E(Y|X)]', fontsize=12)
    ax.set_title('DML: Partialled-Out Relationship', fontsize=12)
    ax.legend(fontsize=11)
    return fig, ax


def plot_forest(
    forest_df: pd.DataFrame,
    baseline_ate: float,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Robustness forest plot.

    Parameters
    ----------
    forest_df    : DataFrame with columns Specification, ATE_pp, CI_lo, CI_hi, n
    baseline_ate : reference ATE in percentage points (for vertical reference line)
    """
    fig, ax = (ax.get_figure(), ax) if ax else plt.subplots(
        figsize=(10, len(forest_df) * 0.75 + 2)
    )
    y_pos = np.arange(len(forest_df))[::-1]

    for i, (y, (_, row)) in enumerate(zip(y_pos, forest_df.iterrows())):
        color = 'steelblue' if i == 0 else 'dimgrey'
        lw    = 2.5         if i == 0 else 1.5
        ax.plot([row['CI_lo'], row['CI_hi']], [y, y], color=color, linewidth=lw)
        ax.scatter(row['ATE_pp'], y,
                   color='crimson' if i == 0 else 'black',
                   s=80 if i == 0 else 50, zorder=5)
        ax.text(forest_df['CI_hi'].max() + 0.15, y,
                f'{row["ATE_pp"]:+.3f} pp   (n={int(row["n"]):,})',
                va='center', fontsize=9)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(baseline_ate, color='steelblue', linewidth=1.5,
               linestyle=':', alpha=0.7, label='Baseline ATE')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(forest_df['Specification'], fontsize=10)
    ax.set_xlabel('AIPW ATE on P(Default) (pp)', fontsize=12)
    ax.set_title('Robustness Forest Plot', fontsize=13)
    ax.legend()
    return fig, ax
