"""
Revision Phase 1 — Statistical utilities.

Implements the statistical machinery required by the Scientific Reports
reviewers:

  • bootstrap_ci        — percentile bootstrap 95% CI (10,000 resamples)
  • cluster_bootstrap_ci— seed-level cluster bootstrap (episodes nested)
  • paired_tests        — Wilcoxon signed-rank AND paired permutation test
  • cliffs_delta        — non-parametric effect size (with rank-biserial
                          equivalence for paired data)
  • holm_bonferroni     — family-wise error correction
  • pairwise_matrix     — all-pairs comparison table with corrected p-values

All functions operate on 1-D arrays of episode-level metrics (one value per
episode / seed). Paired tests require equal-length arrays matched by
environment seed.

Usage:
    from stats_utils import bootstrap_ci, paired_tests, cliffs_delta
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps

DEFAULT_N_BOOT = 10_000
DEFAULT_N_PERM = 20_000
RNG_SEED = 12345


# ─────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    x: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = 0.95,
    statistic=np.mean,
    seed: int = RNG_SEED,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for a statistic of x.

    Returns (point_estimate, ci_low, ci_high).
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = statistic(x[rng.integers(0, n, n)])
    alpha = (1.0 - ci) / 2.0
    return float(statistic(x)), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def cluster_bootstrap_ci(
    x: Sequence[float],
    clusters: Sequence,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = 0.95,
    statistic=np.mean,
    seed: int = RNG_SEED,
) -> Tuple[float, float, float]:
    """Cluster (block) bootstrap CI: resample CLUSTERS with replacement, keeping
    every observation nested inside a drawn cluster.

    `bootstrap_ci` treats the episodes of one training run as independent draws.
    They are not: episodes sharing a training seed share a policy, so the
    quantity a reader cares about — the performance of a *newly trained* agent —
    has its uncertainty driven by seed-level variation. Resampling seeds (with
    their episodes nested) is the interval for that quantity, and it is wider.

    Returns (point_estimate, ci_low, ci_high). With few clusters (here 5 training
    seeds) the interval is coarse by construction; that coarseness is the honest
    reflection of how many independent training runs were performed.
    """
    x = np.asarray(x, dtype=float)
    clusters = np.asarray(clusters)
    if len(x) != len(clusters):
        raise ValueError('x and clusters must have the same length')
    keys = np.unique(clusters)
    if len(keys) < 2:
        raise ValueError('cluster bootstrap needs at least two clusters')
    members = [np.where(clusters == k)[0] for k in keys]
    rng = np.random.default_rng(seed)
    n_clusters = len(keys)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([members[d] for d in drawn])
        boots[b] = statistic(x[idx])
    alpha = (1.0 - ci) / 2.0
    return (float(statistic(x)),
            float(np.quantile(boots, alpha)),
            float(np.quantile(boots, 1 - alpha)))


# ─────────────────────────────────────────────────────────────
# Effect sizes
# ─────────────────────────────────────────────────────────────

def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta: P(x > y) - P(x < y) over all pairs. Range [-1, 1].

    |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large
    (Romano et al. 2006 thresholds).
    """
    x = np.asarray(x, dtype=float)[:, None]
    y = np.asarray(y, dtype=float)[None, :]
    greater = (x > y).sum()
    less = (x < y).sum()
    return float((greater - less) / (x.size * y.size / 1))  # x.size*y.size == n*m


def rank_biserial_paired(x: Sequence[float], y: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation (effect size for Wilcoxon).

    r = (sum of positive-difference ranks - sum of negative) / total rank sum.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    total = ranks.sum()
    return float((r_pos - r_neg) / total)


def interpret_delta(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


# ─────────────────────────────────────────────────────────────
# Paired tests
# ─────────────────────────────────────────────────────────────

def permutation_test_paired(
    x: Sequence[float],
    y: Sequence[float],
    n_perm: int = DEFAULT_N_PERM,
    seed: int = RNG_SEED,
) -> float:
    """Exact-style paired permutation test on the mean difference.

    Sign-flips each paired difference; two-sided p-value. With n pairs there
    are 2^n distinct assignments — for n <= 20 we enumerate exhaustively,
    otherwise Monte-Carlo with n_perm draws.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(d)
    obs = abs(d.mean())
    if n <= 20:
        # Exhaustive sign-flip enumeration
        count = 0
        total = 2 ** n
        for mask in range(total):
            signs = np.fromiter(((1 if (mask >> i) & 1 else -1) for i in range(n)), dtype=float, count=n)
            if abs((d * signs).mean()) >= obs - 1e-12:
                count += 1
        return count / total
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm_means >= obs - 1e-12) + 1) / (n_perm + 1))


@dataclass
class PairedResult:
    name_a: str
    name_b: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    wilcoxon_p: float
    permutation_p: float
    cliffs_delta: float
    rank_biserial: float
    effect_label: str
    # Filled in by holm_bonferroni:
    wilcoxon_p_corr: Optional[float] = None
    permutation_p_corr: Optional[float] = None
    significant: Optional[bool] = None


def paired_tests(x: Sequence[float], y: Sequence[float], name_a: str = "A", name_b: str = "B") -> PairedResult:
    """Wilcoxon signed-rank + paired permutation test + effect sizes."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert len(x) == len(y), "paired tests require matched arrays"
    d = x - y
    if np.allclose(d, 0):
        wp = 1.0
    else:
        try:
            wp = float(sps.wilcoxon(x, y, zero_method="wilcox", method="exact").pvalue)
        except ValueError:
            wp = float(sps.wilcoxon(x, y, zero_method="wilcox").pvalue)
    pp = permutation_test_paired(x, y)
    cd = cliffs_delta(x, y)
    rb = rank_biserial_paired(x, y)
    return PairedResult(
        name_a=name_a, name_b=name_b, n=len(x),
        mean_a=float(x.mean()), mean_b=float(y.mean()),
        mean_diff=float(d.mean()),
        wilcoxon_p=wp, permutation_p=pp,
        cliffs_delta=cd, rank_biserial=rb,
        effect_label=interpret_delta(cd),
    )


# ─────────────────────────────────────────────────────────────
# Multiple-comparison correction
# ─────────────────────────────────────────────────────────────

def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Holm–Bonferroni step-down correction.

    Returns (adjusted_pvals, reject_flags) in the ORIGINAL order.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running_max = max(running_max, min(adj, 1.0))
        adjusted[idx] = running_max
    reject = adjusted <= alpha
    return adjusted.tolist(), reject.tolist()


def correct_family(results: List[PairedResult], alpha: float = 0.05, use: str = "permutation") -> List[PairedResult]:
    """Apply Holm–Bonferroni across a family of PairedResults (in place).

    `use` selects which raw p-value drives the significance flag
    ('permutation' or 'wilcoxon'); both corrected values are stored.
    """
    wp_adj, _ = holm_bonferroni([r.wilcoxon_p for r in results], alpha)
    pp_adj, _ = holm_bonferroni([r.permutation_p for r in results], alpha)
    for r, wa, pa in zip(results, wp_adj, pp_adj):
        r.wilcoxon_p_corr = wa
        r.permutation_p_corr = pa
        r.significant = (pa if use == "permutation" else wa) <= alpha
    return results


# ─────────────────────────────────────────────────────────────
# Convenience: all-pairs matrix
# ─────────────────────────────────────────────────────────────

def pairwise_matrix(
    groups: Dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> List[PairedResult]:
    """All-pairs paired comparisons with a single Holm–Bonferroni family.

    All groups must have equal length (matched by env seed / episode slot).
    """
    results = [
        paired_tests(groups[a], groups[b], name_a=a, name_b=b)
        for a, b in combinations(groups.keys(), 2)
    ]
    return correct_family(results, alpha=alpha)


def results_to_records(results: List[PairedResult]) -> List[dict]:
    """PairedResults → list of flat dicts (for pandas.DataFrame)."""
    return [r.__dict__.copy() for r in results]


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 10)
    b = a - 0.8 + rng.normal(0, 0.3, 10)
    c = a + rng.normal(0, 0.05, 10)

    m, lo, hi = bootstrap_ci(a)
    assert lo < m < hi
    print(f"bootstrap_ci(a) = {m:.3f} [{lo:.3f}, {hi:.3f}]")

    res = pairwise_matrix({"A": a, "B": b, "C": c})
    for r in res:
        print(f"{r.name_a} vs {r.name_b}: diff={r.mean_diff:+.3f} "
              f"wilc_p={r.wilcoxon_p:.4f} perm_p={r.permutation_p:.4f} "
              f"delta={r.cliffs_delta:+.2f} ({r.effect_label}) "
              f"holm_perm_p={r.permutation_p_corr:.4f} sig={r.significant}")
    ab = [r for r in res if {r.name_a, r.name_b} == {"A", "B"}][0]
    ac = [r for r in res if {r.name_a, r.name_b} == {"A", "C"}][0]
    assert ab.significant and not ac.significant
    print("✅ stats_utils self-test passed.")
