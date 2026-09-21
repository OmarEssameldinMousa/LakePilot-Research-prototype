"""
Rewrite paper Tables 1 and 2 with multi-seed statistics (Phase 1 deliverable).

Published Table 1: policy ranking on the 1,000-step v5 standard workload
    (Rank | Policy | Reward | Latency | Files | Pruning) — single episode per policy.
Published Table 2: three-way RL comparison reported as **step-level mean ± s.d.
    pooled over 5,000 steps**.

The step-level s.d. in the published Table 2 measures step-to-step variability
within episodes, not uncertainty about the mean, and pooling 5,000
autocorrelated steps as if independent is exactly what the reviewers objected
to. The rewritten tables report episode-level means with percentile bootstrap
95% CIs and, separately, the across-seed standard deviation — the quantity that
actually captures training-run variability.

Usage:  python3 revision/stats/build_manuscript_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import bootstrap_ci, cluster_bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'revision' / 'phase1_stats'

# Published values for side-by-side comparison (paper Tables 1 and 2,
# 1,000-step v5 standard workload).
PUBLISHED = {
    'MultiAgentRL_attentive_ppo':      ('AttentivePPO',           0.2201, 323.6, 54.9, 0.191),
    'MultiAgentRL_attentive_ppoclip':  ('AttentivePPO',           0.2201, 323.6, 54.9, 0.191),
    'MultiAgentRL_ddqn':               ('DDQN',                   0.2196, 323.4, 58.8, 0.197),
    'MultiAgentRL_mlp_ppo':            ('MLP-PPO',                0.2124, 387.8, 62.7, 0.203),
    'WorkloadAwareThreshold':          ('WorkloadAwareThreshold', 0.2111, 439.9, 93.4, 0.285),
    'AlwaysCompact_C128':              ('AlwaysCompact_C128',     0.1869, 344.3, 43.5, 0.000),
    'Threshold10_C128':                ('Threshold10_C128',       0.1734, 429.7, 46.8, 0.000),
    'CompactOnlyRL_attentive_ppoclip': ('CompactOnly (ablation)', 0.1699, 344.1, 48.9, 0.000),
    'PartitionOnlyRL_attentive_ppoclip': ('PartitionOnly (ablation)', -0.1765, 5306.9, 1245.0, 0.110),
}

DISPLAY = {
    'MultiAgentRL_ddqn': 'DDQN (hierarchical RL)',
    'MultiAgentRL_attentive_ppoclip': 'AttentivePPO-clip (hierarchical RL)',
    'MultiAgentRL_mlp_ppo': 'MLP-PPO (hierarchical RL)',
    'MultiAgentRL_attentive_ppo': 'AttentivePPO-AWR (published recipe)',
    'CompactOnlyRL_attentive_ppoclip': 'CompactOnly (ablation)',
    'PartitionOnlyRL_attentive_ppoclip': 'PartitionOnly (ablation)',
}
RL = ('MultiAgentRL_', 'CompactOnlyRL_', 'PartitionOnlyRL_')


def episode_frame() -> pd.DataFrame:
    import glob
    rows = []
    for f in sorted(glob.glob(str(OUT / 'raw' / '*' / 'seed*' / '*' / 'episode_summary.csv'))):
        p = Path(f)
        d = pd.read_csv(f)
        d['policy'], d['seed'], d['protocol'] = p.parts[-4], int(p.parts[-3][4:]), p.parts[-2]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def table1(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    sub = df[df.protocol == protocol]
    out = []
    for policy, g in sub.groupby('policy'):
        n_seeds = g['seed'].nunique()
        r, lo, hi = bootstrap_ci(g['avg_global_reward'].values)
        per_seed = g.groupby('seed')['avg_global_reward'].mean()
        # Seed-level cluster interval (see stats_utils.cluster_bootstrap_ci):
        # the uncertainty about a newly *trained* agent rather than a newly
        # drawn episode. Reported alongside, never instead.
        if n_seeds >= 2 and len(g) > n_seeds:
            _, clo, chi = cluster_bootstrap_ci(g['avg_global_reward'].values, g['seed'].values)
        else:
            clo = chi = np.nan
        out.append({
            'policy': DISPLAY.get(policy, policy),
            'reward': round(r, 4),
            'ci_low': round(lo, 4), 'ci_high': round(hi, 4),
            'cluster_ci_low': round(clo, 4) if clo == clo else np.nan,
            'cluster_ci_high': round(chi, 4) if chi == chi else np.nan,
            'seed_sd': round(per_seed.std(ddof=1), 4) if n_seeds > 1 else np.nan,
            'latency_ms': round(g['avg_latency_ms'].mean(), 1),
            'files': round(g['avg_file_count'].mean(), 1),
            'pruning': round(g['avg_partition_pruning_ratio'].mean(), 3),
            'n_seeds': n_seeds, 'n_episodes': len(g),
            'published_reward': PUBLISHED.get(policy, (None,) * 5)[1],
        })
    t = pd.DataFrame(out).sort_values('reward', ascending=False).reset_index(drop=True)
    t.insert(0, 'rank', np.arange(1, len(t) + 1))
    return t


def table2(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    agents = ['MultiAgentRL_ddqn', 'MultiAgentRL_attentive_ppoclip', 'MultiAgentRL_mlp_ppo']
    sub = df[(df.protocol == protocol) & (df.policy.isin(agents))]
    out = []
    for policy in agents:
        g = sub[sub.policy == policy]
        if g.empty:
            continue
        rec = {'agent': DISPLAY[policy], 'n_seeds': g['seed'].nunique(), 'n_episodes': len(g)}
        for col, name in [('avg_global_reward', 'reward'), ('avg_latency_ms', 'latency_ms'),
                          ('avg_file_count', 'files'), ('avg_partition_pruning_ratio', 'pruning')]:
            m, lo, hi = bootstrap_ci(g[col].values)
            per_seed = g.groupby('seed')[col].mean()
            dp = 4 if 'reward' in name or name == 'pruning' else 1
            rec[name] = round(m, dp)
            rec[f'{name}_ci'] = f'[{lo:.{dp}f}, {hi:.{dp}f}]'
            rec[f'{name}_seed_sd'] = round(per_seed.std(ddof=1), dp)
            if g['seed'].nunique() >= 2 and len(g) > g['seed'].nunique():
                _, clo, chi = cluster_bootstrap_ci(g[col].values, g['seed'].values)
                rec[f'{name}_cluster_ci'] = f'[{clo:.{dp}f}, {chi:.{dp}f}]'
        out.append(rec)
    return pd.DataFrame(out)


def md_table1(t: pd.DataFrame) -> str:
    lines = ['| Rank | Policy | Reward (95% CI) | Seed sd | Latency (ms) | Files | Pruning | Seeds | Published |',
             '|---|---|---|---|---|---|---|---|---|']
    for _, r in t.iterrows():
        sd = '—' if pd.isna(r['seed_sd']) else f"{r['seed_sd']:.4f}"
        pub = '—' if pd.isna(r['published_reward']) else f"{r['published_reward']:.4f}"
        lines.append(f"| {r['rank']} | {r['policy']} | {r['reward']:.4f} "
                     f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {sd} | {r['latency_ms']:.1f} | "
                     f"{r['files']:.1f} | {r['pruning']:.3f} | {int(r['n_seeds'])} | {pub} |")
    return '\n'.join(lines)


def md_table2(t: pd.DataFrame) -> str:
    lines = ['| Agent | Reward (95% CI) | Latency ms (95% CI) | Files (95% CI) | Pruning (95% CI) | Seeds |',
             '|---|---|---|---|---|---|']
    for _, r in t.iterrows():
        lines.append(f"| {r['agent']} | {r['reward']:.4f} {r['reward_ci']} | "
                     f"{r['latency_ms']:.1f} {r['latency_ms_ci']} | "
                     f"{r['files']:.1f} {r['files_ci']} | "
                     f"{r['pruning']:.3f} {r['pruning_ci']} | {int(r['n_seeds'])} |")
    return '\n'.join(lines)


def main():
    df = episode_frame()
    md = ['# Rewritten manuscript tables (Phase 1)', '',
          'Means are over independent seeds; intervals are percentile bootstrap 95% CIs',
          '(10,000 resamples) over episodes. "Seed sd" is the standard deviation of',
          'per-seed means — i.e. training-run variability, the quantity the published',
          'tables did not report.', '']

    for protocol, label in [('std1000', '1,000-step v5 standard workload'),
                            ('eval500', '500-step compressed evaluation workload')]:
        t1 = table1(df, protocol)
        t2 = table2(df, protocol)
        if t1.empty:
            continue
        t1.to_csv(OUT / f'table1_rewritten_{protocol}.csv', index=False)
        if not t2.empty:
            t2.to_csv(OUT / f'table2_rewritten_{protocol}.csv', index=False)
        md += [f'## Table 1 (rewritten) — {label}', '', md_table1(t1), '']
        if not t2.empty:
            md += [f'## Table 2 (rewritten) — {label}', '', md_table2(t2), '',
                   'The published Table 2 reported step-level mean ± s.d. pooled over 5,000',
                   'steps. That s.d. describes step-to-step variation within episodes and',
                   'treats autocorrelated steps as independent samples; it is not an',
                   'uncertainty estimate for the mean. The intervals above are computed over',
                   'independent evaluation episodes, and the seed sd column reports',
                   'across-training-run variability.', '']

    (OUT / 'MANUSCRIPT_TABLES.md').write_text('\n'.join(md))
    print('\n'.join(md))
    print(f'\nwrote MANUSCRIPT_TABLES.md + table{{1,2}}_rewritten_*.csv → {OUT}')


if __name__ == '__main__':
    main()
