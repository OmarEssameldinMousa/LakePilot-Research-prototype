"""
Phase 5 — generalization to shifted workloads.

Compares every policy's performance on two held-out workloads against its
in-distribution baseline (eval500 from Phase 1), all at 500 steps:

  drift500  same profiles and order, but transitions are linear interpolations
            of query-type probabilities over 50-step windows. The agents were
            trained only on trajectories with abrupt phase switches.
  mixed500  each phase samples 50/50 from two profiles simultaneously, so no
            single query type dominates. This directly attacks the partition
            specialist's mechanism ("detect the dominant type and partition
            for it"), which has no single right answer here.

Outputs into revision/phase5_generalization/:
    generalization_results.csv   reward/latency/files/pruning per policy per workload
    degradation.csv              % change vs the in-distribution baseline
    figures/degradation.{pdf,png}

Usage:  python3 revision/stats/analyze_phase5.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import bootstrap_ci, paired_tests, correct_family  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
P5 = ROOT / 'revision' / 'phase5_generalization'
P1 = ROOT / 'revision' / 'phase1_stats' / 'raw'
FIGS = P5 / 'figures'

DISPLAY = {
    'MultiAgentRL_ddqn': 'DDQN',
    'MultiAgentRL_attentive_ppoclip': 'AttentivePPO-clip',
    'MultiAgentRL_mlp_ppo': 'MLP-PPO',
    'WorkloadAwareThreshold': 'WorkloadAwareThreshold',
    'AlwaysCompact_C128': 'AlwaysCompact_C128',
}
METRICS = ['avg_global_reward', 'avg_latency_ms', 'avg_file_count',
           'avg_partition_pruning_ratio']


def load(root: Path, policy: str, protocol: str) -> pd.DataFrame | None:
    fs = sorted(glob.glob(str(root / policy / 'seed*' / protocol / 'episode_summary.csv')))
    if not fs:
        return None
    out = []
    for f in fs:
        d = pd.read_csv(f)
        d['seed'] = int(Path(f).parts[-3][4:])
        out.append(d)
    return pd.concat(out, ignore_index=True)


def main():
    rows, per_seed = [], {}
    for policy, name in DISPLAY.items():
        for root, proto, label in [(P1, 'eval500', 'baseline (eval500)'),
                                   (P5 / 'raw', 'drift500', 'drift500'),
                                   (P5 / 'raw', 'mixed500', 'mixed500')]:
            d = load(root, policy, proto)
            if d is None:
                continue
            rec = {'policy': name, 'workload': label, 'n_seeds': d['seed'].nunique(),
                   'n_episodes': len(d)}
            for m in METRICS:
                v, lo, hi = bootstrap_ci(d[m].values)
                rec[m] = v
                rec[f'{m}_ci_low'], rec[f'{m}_ci_high'] = lo, hi
            rows.append(rec)
            per_seed[(name, label)] = d.groupby('seed')['avg_global_reward'].mean()

    t = pd.DataFrame(rows)
    t.to_csv(P5 / 'generalization_results.csv', index=False)

    # ── degradation vs baseline ──
    deg = []
    for name in DISPLAY.values():
        base = t[(t.policy == name) & (t.workload == 'baseline (eval500)')]
        if base.empty:
            continue
        b = base.iloc[0]
        for wl in ['drift500', 'mixed500']:
            cur = t[(t.policy == name) & (t.workload == wl)]
            if cur.empty:
                continue
            c = cur.iloc[0]
            rec = {'policy': name, 'workload': wl,
                   'baseline_reward': b['avg_global_reward'],
                   'reward': c['avg_global_reward'],
                   'delta': c['avg_global_reward'] - b['avg_global_reward'],
                   'pct_change': (c['avg_global_reward'] - b['avg_global_reward'])
                                 / abs(b['avg_global_reward']) * 100,
                   'pruning_baseline': b['avg_partition_pruning_ratio'],
                   'pruning': c['avg_partition_pruning_ratio'],
                   'latency': c['avg_latency_ms'], 'files': c['avg_file_count']}
            # paired test across shared seeds
            a, bb = per_seed.get((name, wl)), per_seed.get((name, 'baseline (eval500)'))
            if a is not None and bb is not None:
                shared = sorted(set(a.index) & set(bb.index))
                if len(shared) >= 3:
                    r = paired_tests(a.loc[shared].values, bb.loc[shared].values, wl, 'baseline')
                    rec['p_raw'] = r.permutation_p
                    rec['cliffs_delta'] = r.cliffs_delta
            deg.append(rec)
    dg = pd.DataFrame(deg)
    if 'p_raw' in dg:
        from stats_utils import holm_bonferroni
        adj, rej = holm_bonferroni(dg['p_raw'].fillna(1.0).tolist())
        dg['p_holm'], dg['significant'] = adj, rej
    dg.to_csv(P5 / 'degradation.csv', index=False)

    pd.set_option('display.width', 200)
    print('── Performance on held-out workloads ──')
    show = ['policy', 'workload', 'avg_global_reward', 'avg_global_reward_ci_low',
            'avg_global_reward_ci_high', 'avg_latency_ms', 'avg_file_count',
            'avg_partition_pruning_ratio', 'n_seeds']
    print(t[show].round(4).to_string(index=False))
    print('\n── Degradation vs in-distribution baseline ──')
    cols = ['policy', 'workload', 'baseline_reward', 'reward', 'delta', 'pct_change',
            'pruning_baseline', 'pruning']
    if 'p_holm' in dg:
        cols += ['p_holm', 'significant']
    print(dg[cols].round(4).to_string(index=False))

    # ── figure ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        FIGS.mkdir(parents=True, exist_ok=True)
        pols = list(DISPLAY.values())
        x = np.arange(len(pols))
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for i, (wl, colour) in enumerate([('baseline (eval500)', '#888780'),
                                          ('drift500', '#185FA5'),
                                          ('mixed500', '#E8593C')]):
            vals, los, his = [], [], []
            for p in pols:
                r = t[(t.policy == p) & (t.workload == wl)]
                if r.empty:
                    vals.append(np.nan); los.append(0); his.append(0); continue
                r = r.iloc[0]
                vals.append(r['avg_global_reward'])
                los.append(r['avg_global_reward'] - r['avg_global_reward_ci_low'])
                his.append(r['avg_global_reward_ci_high'] - r['avg_global_reward'])
            ax.bar(x + (i - 1) * 0.27, vals, 0.26, yerr=[los, his], capsize=3,
                   color=colour, label=wl, error_kw={'elinewidth': 1})
        ax.set_xticks(x); ax.set_xticklabels(pols, rotation=18, ha='right')
        ax.set_ylabel('mean global reward')
        ax.set_title('Generalization to held-out workloads', fontweight='bold', loc='left')
        ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#D3D1C7', lw=0.5, alpha=0.6)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(FIGS / f'degradation.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'  ! figure failed: {e}')

    print(f'\nwrote generalization_results.csv, degradation.csv, figures/ → {P5}')


if __name__ == '__main__':
    main()
