"""Statistical analysis for comparing classifiers (Demšar 2006).

Implements:
- Wilcoxon Signed-Rank Test (for comparing two classifiers across folds/datasets)
- Friedman Omnibus Test + Iman-Davenport F-statistic (for comparing multiple classifiers)
- Post-hoc Nemenyi Test (all-pairs comparison) and Bonferroni-Dunn Test (vs control)
- Critical Difference (CD) diagram generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# Studentized range statistic q_alpha / sqrt(2) values from Demšar (2006) Table 5(a) & 5(b)
# For Nemenyi (two-tailed all-pairs) at alpha = 0.05, we can also compute exact values via
# scipy.stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
def get_nemenyi_q_alpha(k, alpha=0.05):
    """Critical value q_alpha for the Nemenyi test with k algorithms."""
    return float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0))


def get_bonferroni_dunn_q_alpha(k, alpha=0.05):
    """Critical value q_alpha for the Bonferroni-Dunn test comparing k-1 models against 1 control."""
    return float(stats.norm.ppf(1.0 - alpha / (2.0 * (k - 1))))


def wilcoxon_test(scores_a, scores_b, name_a="Model A", name_b="Model B", alpha=0.05):
    """Perform a two-sided Wilcoxon signed-rank test between two classifiers across folds.

    Also computes a paired t-test for completeness and reference.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b

    # Handle case where all differences are zero
    if np.allclose(diff, 0.0):
        w_stat, p_val = 0.0, 1.0
    else:
        res = stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
        w_stat, p_val = float(res.statistic), float(res.pvalue)

    t_res = stats.ttest_rel(a, b)
    t_stat = float(t_res.statistic) if not np.isnan(t_res.statistic) else 0.0
    t_pval = float(t_res.pvalue) if not np.isnan(t_res.pvalue) else 1.0

    return {
        "model_a": name_a,
        "model_b": name_b,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "wilcoxon_stat": w_stat,
        "p_value": p_val,
        "significant": bool(p_val < alpha),
        "ttest_stat": t_stat,
        "ttest_p_value": t_pval,
    }


def friedman_test(pivot_df, higher_is_better=True, alpha=0.05):
    """Perform Friedman test and Iman-Davenport F-test on a (n_folds x k_models) DataFrame.

    Parameters
    ----------
    pivot_df : pd.DataFrame
        Rows = folds (N), Columns = models/configurations (k), Values = performance metric.
    higher_is_better : bool, default=True
        If True, highest score gets rank 1. If False (e.g. time/storage), lowest value gets rank 1.
    """
    N, k = pivot_df.shape
    if k < 3:
        raise ValueError(f"Friedman test requires at least 3 models, got {k}.")

    # Rank across columns for each fold (1 = best)
    ranks_df = pivot_df.rank(axis=1, ascending=not higher_is_better, method="average")
    avg_ranks = ranks_df.mean(axis=0).sort_values()

    # Scipy Friedman chi-square
    cols = [pivot_df[col].values for col in pivot_df.columns]
    if np.allclose(ranks_df.values, ranks_df.values[:, [0]]):
        chi2_stat, p_chi2 = 0.0, 1.0
        f_stat, p_f = 0.0, 1.0
    else:
        chi2_res = stats.friedmanchisquare(*cols)
        chi2_stat, p_chi2 = float(chi2_res.statistic), float(chi2_res.pvalue)

        # Iman & Davenport (1980) F_F statistic (Demšar 2006, Eq. 2)
        denom = N * (k - 1) - chi2_stat
        if denom > 1e-12:
            f_stat = float(((N - 1) * chi2_stat) / denom)
            p_f = float(1.0 - stats.f.cdf(f_stat, k - 1, (k - 1) * (N - 1)))
        else:
            f_stat, p_f = np.inf, 0.0

    # Nemenyi and Bonferroni-Dunn Critical Differences
    se = np.sqrt(k * (k + 1.0) / (6.0 * N))
    cd_nemenyi = get_nemenyi_q_alpha(k, alpha=alpha) * se
    cd_bonferroni = get_bonferroni_dunn_q_alpha(k, alpha=alpha) * se

    summary_table = pd.DataFrame({
        "model": avg_ranks.index,
        "average_rank": avg_ranks.values,
        "mean_score": [pivot_df[m].mean() for m in avg_ranks.index],
        "std_score": [pivot_df[m].std() for m in avg_ranks.index],
    })
    best_rank = avg_ranks.iloc[0]
    summary_table["rank_diff_from_best"] = summary_table["average_rank"] - best_rank
    summary_table["within_nemenyi_cd"] = summary_table["rank_diff_from_best"] <= cd_nemenyi
    summary_table["within_bonferroni_cd"] = summary_table["rank_diff_from_best"] <= cd_bonferroni

    return {
        "N": N,
        "k": k,
        "alpha": alpha,
        "friedman_chi2": chi2_stat,
        "friedman_p_value": p_chi2,
        "iman_davenport_f": f_stat,
        "iman_davenport_p_value": p_f,
        "significant": bool(p_f < alpha or p_chi2 < alpha),
        "cd_nemenyi": float(cd_nemenyi),
        "cd_bonferroni_dunn": float(cd_bonferroni),
        "ranks_df": ranks_df,
        "summary_table": summary_table,
    }


def plot_critical_difference_diagram(friedman_result, title="Critical Difference Diagram", save_path=None):
    """Plot a Demšar (2006) Critical Difference (CD) diagram from friedman_test output."""
    summary = friedman_result["summary_table"]
    cd = friedman_result["cd_nemenyi"]
    k = friedman_result["k"]

    names = list(summary["model"])
    ranks = list(summary["average_rank"])

    fig, ax = plt.subplots(figsize=(max(8, k * 1.1), max(3.5, k * 0.45 + 1.5)))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_position(("axes", 0.78))
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.tick_top()
    ax.set_xticks(range(1, k + 1))
    ax.set_yticks([])
    ax.set_title(f"{title}\n(Friedman p={friedman_result['friedman_p_value']:.4f}, Nemenyi CD={cd:.3f})", pad=35, fontsize=11, fontweight="bold")

    # Draw CD bar at top left
    cd_start = 1.0
    cd_end = min(k, 1.0 + cd)
    ax.plot([cd_start, cd_end], [0.92, 0.92], color="crimson", lw=2.5)
    ax.plot([cd_start, cd_start], [0.90, 0.94], color="crimson", lw=2.0)
    ax.plot([cd_end, cd_end], [0.90, 0.94], color="crimson", lw=2.0)
    ax.text((cd_start + cd_end) / 2, 0.95, f"CD = {cd:.2f}", ha="center", va="bottom", color="crimson", fontsize=9, fontweight="bold")

    # Split models into left half (better ranks) and right half (worse ranks)
    mid = (k + 1) // 2
    left_items = list(zip(names[:mid], ranks[:mid]))
    right_items = list(zip(names[mid:], ranks[mid:]))

    y_top = 0.78
    step = 0.55 / max(mid, len(right_items), 1)

    for idx, (name, r) in enumerate(left_items):
        y_pos = 0.62 - idx * step
        ax.plot([r, r], [y_top, y_pos], color="steelblue", lw=1.2, ls="--")
        ax.plot([0.6, r], [y_pos, y_pos], color="steelblue", lw=1.2)
        ax.plot(r, y_top, "o", color="navy", markersize=6)
        ax.text(0.55, y_pos, f"{name} ({r:.2f})", ha="right", va="center", fontsize=9, fontweight="bold" if idx == 0 else "normal")

    for idx, (name, r) in enumerate(reversed(right_items)):
        y_pos = 0.62 - idx * step
        ax.plot([r, r], [y_top, y_pos], color="darkorange", lw=1.2, ls="--")
        ax.plot([r, k + 0.4], [y_pos, y_pos], color="darkorange", lw=1.2)
        ax.plot(r, y_top, "o", color="chocolate", markersize=6)
        ax.text(k + 0.45, y_pos, f"({r:.2f}) {name}", ha="left", va="center", fontsize=9)

    # Connect cliques of statistically indistinguishable classifiers
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) <= cd:
            j += 1
        if j > i:
            # Check if this clique is a subset of an existing clique
            if not any(c_start <= i and j <= c_end for c_start, c_end in cliques):
                cliques.append((i, j))

    for c_idx, (i, j) in enumerate(cliques):
        y_clique = 0.72 - c_idx * 0.045
        ax.plot([ranks[i], ranks[j]], [y_clique, y_clique], color="black", lw=3.0, solid_capstyle="round")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig

