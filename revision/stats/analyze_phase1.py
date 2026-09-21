"""
Phase 1 analysis — results table with CIs, significance matrix, forest plot, summary.

Consumes the seeded evaluation outputs written by
`LakeGymLite/src/revision_phase1_eval.py`:

    revision/phase1_stats/raw/<policy>/seed<k>/<protocol>/episode_summary.csv

Episodes are matched across policies by `env_seed` (the runner derives seeds from
(protocol, slot, episode) only), so every cross-policy comparison is paired.

Outputs (into revision/phase1_stats/):
    results_table.csv          mean ± bootstrap 95% CI, per policy × protocol × metric
    significance_matrix.csv    all pairwise paired tests, Holm-corrected
    figures/forest_reward.{pdf,png}
    SUMMARY.md                 numbers + explicit answers to the reviewer questions

Usage:  python3 revision/stats/analyze_phase1.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import (bootstrap_ci, cluster_bootstrap_ci, paired_tests,  # noqa: E402
                         correct_family, results_to_records)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'revision' / 'phase1_stats' / 'raw'
OUT = ROOT / 'revision' / 'phase1_stats'
FIGS = OUT / 'figures'

METRICS = [
    ('avg_global_reward', 'Global reward', True),
    ('avg_latency_ms', 'Latency (ms)', False),
    ('avg_file_count', 'File count', False),
    ('avg_partition_pruning_ratio', 'Pruning ratio', True),
]
PROTOCOLS = ['std1000', 'eval500']

# Policies whose names identify them as learned agents
def is_rl(policy: str) -> bool:
    return policy.startswith(('MultiAgentRL', 'CompactOnlyRL', 'PartitionOnlyRL'))


def load_all() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RAW / '*' / 'seed*' / '*' / 'episode_summary.csv'))):
        parts = Path(f).parts
        policy, seed, protocol = parts[-4], int(parts[-3].replace('seed', '')), parts[-2]
        d = pd.read_csv(f)
        d['policy'], d['seed'], d['protocol'] = policy, seed, protocol
        rows.append(d)
    if not rows:
        sys.exit(f'No episode summaries found under {RAW}')
    return pd.concat(rows, ignore_index=True)


def build_results_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (policy, protocol), g in df.groupby(['policy', 'protocol']):
        rec = {'policy': policy, 'protocol': protocol,
               'n_episodes': len(g), 'n_seeds': g['seed'].nunique()}
        for col, label, _ in METRICS:
            if col not in g:
                continue
            m, lo, hi = bootstrap_ci(g[col].values)
            rec[f'{col}_mean'] = round(m, 6)
            rec[f'{col}_ci_low'] = round(lo, 6)
            rec[f'{col}_ci_high'] = round(hi, 6)
            # Seed-level cluster bootstrap: the interval for the performance of a
            # newly trained agent, as opposed to a newly drawn episode. Only
            # identified where a policy has more than one episode per seed.
            if g['seed'].nunique() >= 2 and len(g) > g['seed'].nunique():
                _, clo, chi = cluster_bootstrap_ci(g[col].values, g['seed'].values)
                rec[f'{col}_cluster_ci_low'] = round(clo, 6)
                rec[f'{col}_cluster_ci_high'] = round(chi, 6)
        # seed-level spread (training-run variability, not just episode noise)
        if 'avg_global_reward' in g:
            per_seed = g.groupby('seed')['avg_global_reward'].mean()
            rec['seed_reward_min'] = round(per_seed.min(), 6)
            rec['seed_reward_max'] = round(per_seed.max(), 6)
            rec['seed_reward_sd'] = round(per_seed.std(ddof=1), 6) if len(per_seed) > 1 else 0.0
        out.append(rec)
    return pd.DataFrame(out).sort_values(['protocol', 'avg_global_reward_mean'], ascending=[True, False])


def paired_frames(df: pd.DataFrame, protocol: str, metric: str):
    """Return {policy: Series indexed by env_seed} for one protocol/metric.

    Pairing is resolved *per comparison* by `significance()`, not globally: a
    single global intersection would truncate every comparison to the coverage
    of the least-covered policy (PartitionOnly ran 3 seeds, so every pair would
    silently drop to n=15 even where 25 matched episodes exist).
    """
    sub = df[df.protocol == protocol]
    per_policy = {}
    for policy, g in sub.groupby('policy'):
        g = g.dropna(subset=[metric]).sort_values('env_seed')
        if g['env_seed'].duplicated().any():
            g = g.groupby('env_seed', as_index=False)[metric].mean()
        per_policy[policy] = g.set_index('env_seed')[metric]
    return per_policy


def significance(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for protocol in PROTOCOLS:
        for col, label, higher_better in METRICS:
            groups = paired_frames(df, protocol, col)
            if len(groups) < 2:
                continue
            names = sorted(groups)
            fam = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    shared = sorted(set(groups[a].index) & set(groups[b].index))
                    if len(shared) < 3:
                        continue
                    fam.append(paired_tests(groups[a].loc[shared].values,
                                            groups[b].loc[shared].values, a, b))
            if not fam:
                continue
            fam = correct_family(fam)   # one Holm family per (protocol, metric)
            for r in results_to_records(fam):
                r.update({'protocol': protocol, 'metric': col,
                          'metric_label': label, 'higher_is_better': higher_better})
                records.append(r)
    return pd.DataFrame(records)


def forest_plot(table: pd.DataFrame, protocol: str = 'eval500'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = table[table.protocol == protocol].dropna(subset=['avg_global_reward_mean'])
    if t.empty:
        return
    t = t.sort_values('avg_global_reward_mean')
    FIGS.mkdir(parents=True, exist_ok=True)

    colors = ['#185FA5' if is_rl(p) else '#888780' for p in t['policy']]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.8))
    y = np.arange(len(t))
    lo = t['avg_global_reward_mean'] - t['avg_global_reward_ci_low']
    hi = t['avg_global_reward_ci_high'] - t['avg_global_reward_mean']
    ax.errorbar(t['avg_global_reward_mean'], y, xerr=[lo, hi], fmt='o',
                ecolor='#B4B2A9', elinewidth=1.4, capsize=3, markersize=0, zorder=1)
    ax.scatter(t['avg_global_reward_mean'], y, c=colors, s=42, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(t['policy'])
    ax.set_xlabel('Mean global reward (bootstrap 95% CI)')
    ax.set_title(f'Policy comparison — {protocol}', fontweight='bold', loc='left')
    ax.axvline(0, color='#D3D1C7', lw=0.8, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#D3D1C7', lw=0.5, alpha=0.6)

    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#185FA5', label='RL agents'),
                       Line2D([], [], marker='o', ls='', color='#888780', label='Heuristics')],
              frameon=False, loc='lower right')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'forest_reward_{protocol}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def write_summary(table: pd.DataFrame, sig: pd.DataFrame, df: pd.DataFrame):
    lines = ['# Phase 1 — multi-seed statistical validation (SUMMARY)', '']
    lines += ['Episodes are matched across policies by environment seed, so every',
              'comparison below is paired. CIs are percentile bootstrap (10,000 resamples);',
              'p-values are paired permutation tests, Holm–Bonferroni corrected within each',
              '(protocol, metric) family. Effect sizes are Cliff\'s delta.', '']

    for protocol in PROTOCOLS:
        t = table[table.protocol == protocol]
        if t.empty:
            continue
        lines += [f'## {protocol}', '',
                  '| policy | reward (95% CI) | latency ms | files | pruning | seeds |',
                  '|---|---|---|---|---|---|']
        for _, r in t.iterrows():
            lines.append(
                f"| {r['policy']} | {r['avg_global_reward_mean']:.4f} "
                f"[{r['avg_global_reward_ci_low']:.4f}, {r['avg_global_reward_ci_high']:.4f}] | "
                f"{r.get('avg_latency_ms_mean', float('nan')):.0f} | "
                f"{r.get('avg_file_count_mean', float('nan')):.1f} | "
                f"{r.get('avg_partition_pruning_ratio_mean', float('nan')):.3f} | "
                f"{int(r['n_seeds'])} |")
        lines.append('')

    # ── reviewer questions ──
    lines += ['## Answers to the reviewer questions', '']
    # Reference protocol = the one where both RL and heuristic policies are present
    # (prefer eval500 for its larger episode count); falls back gracefully while the
    # batch is still filling in.
    ref = None
    for cand in ['eval500', 'std1000']:
        t_c = table[table.protocol == cand]
        if t_c.policy.map(is_rl).any() and (~t_c.policy.map(is_rl)).any():
            ref = cand
            break
    if ref is None:
        lines += ['_(Not yet answerable: no protocol contains both RL and heuristic'
                  ' policies — evaluation batch still in progress.)_', '']
        (OUT / 'SUMMARY.md').write_text('\n'.join(lines))
        return
    lines += [f'_Reference protocol: {ref}._', '']
    t = table[table.protocol == ref]
    rl = t[t.policy.map(is_rl)]
    heur = t[~t.policy.map(is_rl)]

    if not rl.empty and not heur.empty:
        best_h = heur.loc[heur['avg_global_reward_mean'].idxmax()]
        best_rl = rl.loc[rl['avg_global_reward_mean'].idxmax()]
        lines += [f'**(a) Is RL significantly better than the best heuristic on reward?**', '',
                  f'Best heuristic: `{best_h.policy}` = {best_h.avg_global_reward_mean:.4f} '
                  f'[{best_h.avg_global_reward_ci_low:.4f}, {best_h.avg_global_reward_ci_high:.4f}]  ',
                  f'Best RL: `{best_rl.policy}` = {best_rl.avg_global_reward_mean:.4f} '
                  f'[{best_rl.avg_global_reward_ci_low:.4f}, {best_rl.avg_global_reward_ci_high:.4f}]', '']
        s = sig[(sig.protocol == ref) & (sig.metric == 'avg_global_reward')]
        for _, r in s.iterrows():
            pair = {r['name_a'], r['name_b']}
            if best_h.policy in pair and best_rl.policy in pair:
                verdict = 'SIGNIFICANT' if r['significant'] else 'NOT significant'
                lines += [f'Paired test `{r["name_a"]}` vs `{r["name_b"]}`: '
                          f'difference {r["mean_diff"]:+.4f}, Holm-corrected permutation '
                          f'p = {r["permutation_p_corr"]:.4g}, Wilcoxon p = {r["wilcoxon_p_corr"]:.4g}, '
                          f"Cliff's delta = {r['cliffs_delta']:+.2f} ({r['effect_label']}) → **{verdict}**.", '']

    if len(rl) > 1:
        lines += ['**(b) Are the RL architectures distinguishable from each other?**', '']
        s = sig[(sig.protocol == ref) & sig.name_a.map(is_rl) & sig.name_b.map(is_rl)]
        for metric, label, _ in METRICS:
            ss = sig[(sig.protocol == ref) & (sig.metric == metric)
                     & sig.name_a.map(is_rl) & sig.name_b.map(is_rl)]
            if ss.empty:
                continue
            n_sig = int(ss['significant'].sum())
            lines.append(f'- **{label}**: {n_sig}/{len(ss)} RL-vs-RL pairs significant after correction.')
            for _, r in ss.iterrows():
                mark = '✓' if r['significant'] else '✗'
                lines.append(f'  - {mark} {r["name_a"]} vs {r["name_b"]}: '
                             f'{r["mean_diff"]:+.4f}, p={r["permutation_p_corr"]:.4g}, '
                             f"delta={r['cliffs_delta']:+.2f} ({r['effect_label']})")
        lines.append('')

    lines += ['## Seed-level stability', '',
              'Per-seed spread of mean global reward (large spread = unstable training):', '',
              '| policy | protocol | min | max | sd |', '|---|---|---|---|---|']
    for _, r in table.sort_values('seed_reward_sd', ascending=False).iterrows():
        lines.append(f"| {r['policy']} | {r['protocol']} | {r.get('seed_reward_min', float('nan')):.4f} | "
                     f"{r.get('seed_reward_max', float('nan')):.4f} | {r.get('seed_reward_sd', float('nan')):.4f} |")
    lines.append('')

    (OUT / 'SUMMARY.md').write_text('\n'.join(lines))


def main():
    df = load_all()
    print(f'loaded {len(df)} episodes across {df.policy.nunique()} policies')
    print(df.groupby(['policy', 'protocol']).size().to_string())

    table = build_results_table(df)
    table.to_csv(OUT / 'results_table.csv', index=False)

    sig = significance(df)
    if not sig.empty:
        sig.to_csv(OUT / 'significance_matrix.csv', index=False)

    for protocol in PROTOCOLS:
        try:
            forest_plot(table, protocol)
        except Exception as e:  # plotting must never block the numbers
            print(f'  ! forest plot ({protocol}) failed: {e}')

    write_summary(table, sig, df)
    print(f'\nwrote: results_table.csv, significance_matrix.csv, SUMMARY.md, figures/')
    cols = ['policy', 'protocol', 'avg_global_reward_mean', 'avg_global_reward_ci_low',
            'avg_global_reward_ci_high', 'n_seeds']
    print(table[[c for c in cols if c in table]].to_string(index=False))


if __name__ == '__main__':
    main()
