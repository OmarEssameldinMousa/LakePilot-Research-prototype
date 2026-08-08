"""
Phase 3 — sensitivity analysis: meta-controller grid and pruning perturbation.

Consumes the sweeps produced by `LakeGymLite/src/revision_phase3_sweep.py`.
Every configuration was evaluated on the SAME three environment seeds, so
comparisons against the default configuration are paired.

Outputs into revision/phase3_sensitivity/:
    meta_grid_results.csv     reward per configuration, delta vs default, paired test
    pruning_results.csv       reward vs pruning scale for DDQN and the best heuristic,
                              plus the RL-minus-heuristic gap at each scale
    figures/meta_tornado.{pdf,png}
    figures/pruning_gap.{pdf,png}

Usage:  python3 revision/stats/analyze_phase3.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import bootstrap_ci, paired_tests, correct_family  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / 'revision' / 'phase3_sensitivity'
FIGS = P3 / 'figures'

DEFAULTS = {'theta': 0.35, 'boost': 0.15, 'cc': 4, 'pc': 20}
PARAM_LABEL = {'theta': 'θ (urgency threshold)', 'boost': 'utilisation boost',
               'cc': 'compaction cooldown', 'pc': 'partition cooldown'}


def load_config(cfg_dir: Path) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(cfg_dir / 'seed*' / 'eval500' / 'episode_summary.csv')))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return d.sort_values('env_seed')


def parse_meta_label(name: str) -> dict:
    m = re.match(r'meta_theta([\dp]+)_boost([\dp]+)_cc(\d+)_pc(\d+)', name)
    f = lambda s: float(s.replace('p', '.'))
    return {'theta': f(m.group(1)), 'boost': f(m.group(2)),
            'cc': int(m.group(3)), 'pc': int(m.group(4))}


def analyse_meta() -> pd.DataFrame:
    rows, series = [], {}
    for cfg_dir in sorted((P3 / 'meta_grid').glob('meta_*')):
        d = load_config(cfg_dir)
        if d is None:
            continue
        cfg = parse_meta_label(cfg_dir.name)
        r, lo, hi = bootstrap_ci(d['avg_global_reward'].values)
        varied = [k for k in DEFAULTS if cfg[k] != DEFAULTS[k]]
        rows.append({
            'label': cfg_dir.name, **cfg,
            'varied': varied[0] if varied else 'default',
            'value': cfg[varied[0]] if varied else None,
            'reward': r, 'ci_low': lo, 'ci_high': hi,
            'latency_ms': d['avg_latency_ms'].mean(),
            'files': d['avg_file_count'].mean(),
            'pruning': d['avg_partition_pruning_ratio'].mean(),
            'n_episodes': len(d),
        })
        series[cfg_dir.name] = d.set_index('env_seed')['avg_global_reward']

    t = pd.DataFrame(rows)
    if t.empty:
        return t
    dflt = t[t.varied == 'default'].iloc[0]
    t['delta_vs_default'] = t['reward'] - dflt['reward']
    t['pct_vs_default'] = t['delta_vs_default'] / abs(dflt['reward']) * 100

    # Paired tests against the default configuration (same env seeds)
    base = series[dflt['label']]
    results, labels = [], []
    for _, r in t[t.varied != 'default'].iterrows():
        s = series[r['label']]
        shared = sorted(set(base.index) & set(s.index))
        if len(shared) < 3:
            continue
        results.append(paired_tests(s.loc[shared].values, base.loc[shared].values,
                                    r['label'], 'default'))
        labels.append(r['label'])
    if results:
        results = correct_family(results)
        pmap = {lb: (res.permutation_p_corr, res.cliffs_delta, res.significant)
                for lb, res in zip(labels, results)}
        t['p_vs_default'] = t['label'].map(lambda l: pmap.get(l, (np.nan,) * 3)[0])
        t['cliffs_delta'] = t['label'].map(lambda l: pmap.get(l, (np.nan,) * 3)[1])
        t['significant'] = t['label'].map(lambda l: pmap.get(l, (np.nan,) * 3)[2])
    return t.sort_values(['varied', 'value'])


def analyse_pruning() -> pd.DataFrame:
    rows = []
    for cfg_dir in sorted((P3 / 'pruning').glob('prune*')):
        d = load_config(cfg_dir)
        if d is None:
            continue
        m = re.match(r'prune([\dp]+)_(\w+)', cfg_dir.name)
        scale = float(m.group(1).replace('p', '.'))
        who = m.group(2)
        r, lo, hi = bootstrap_ci(d['avg_global_reward'].values)
        rows.append({'scale': scale, 'policy': 'DDQN' if who == 'ddqn' else 'WorkloadAwareThreshold',
                     'reward': r, 'ci_low': lo, 'ci_high': hi,
                     'pruning': d['avg_partition_pruning_ratio'].mean(),
                     'latency_ms': d['avg_latency_ms'].mean(),
                     'files': d['avg_file_count'].mean(), 'n_episodes': len(d)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    # Gap = RL minus heuristic at each scale
    piv = t.pivot(index='scale', columns='policy', values='reward')
    piv['gap'] = piv['DDQN'] - piv['WorkloadAwareThreshold']
    return t, piv.reset_index()


def plot_meta(t: pd.DataFrame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)

    sub = t[t.varied != 'default'].copy()
    sub['name'] = sub.apply(lambda r: f"{PARAM_LABEL[r['varied']]} = {r['value']:g}", axis=1)
    sub = sub.sort_values('delta_vs_default')
    colors = ['#E24B4A' if d < 0 else '#1D9E75' for d in sub['delta_vs_default']]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(sub) + 1.6))
    ax.barh(np.arange(len(sub)), sub['delta_vs_default'], color=colors, height=0.62)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub['name'])
    ax.axvline(0, color='#2C2C2A', lw=1)
    ax.set_xlabel('Δ mean global reward vs default configuration')
    ax.set_title('Meta-controller sensitivity (one-at-a-time)', fontweight='bold', loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#D3D1C7', lw=0.5, alpha=0.6)
    for i, (d, s) in enumerate(zip(sub['delta_vs_default'], sub.get('significant', [False] * len(sub)))):
        if s is True:
            ax.text(d, i, ' *', va='center', ha='left' if d > 0 else 'right', fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'meta_tornado.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_pruning(t: pd.DataFrame, piv: pd.DataFrame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for pol, colour in [('DDQN', '#185FA5'), ('WorkloadAwareThreshold', '#888780')]:
        s = t[t.policy == pol].sort_values('scale')
        ax1.errorbar(s['scale'], s['reward'],
                     yerr=[s['reward'] - s['ci_low'], s['ci_high'] - s['reward']],
                     marker='o', capsize=3, color=colour, label=pol)
    ax1.set_xlabel('pruning-matrix scale'); ax1.set_ylabel('mean global reward')
    ax1.set_title('Reward vs pruning strength', fontweight='bold', loc='left')
    ax1.legend(frameon=False); ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(color='#D3D1C7', lw=0.5, alpha=0.6)

    ax2.axhline(0, color='#E24B4A', lw=1, ls='--')
    ax2.plot(piv['scale'], piv['gap'], marker='s', color='#534AB7')
    ax2.set_xlabel('pruning-matrix scale')
    ax2.set_ylabel('DDQN − WorkloadAwareThreshold')
    ax2.set_title('RL advantage vs pruning strength', fontweight='bold', loc='left')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(color='#D3D1C7', lw=0.5, alpha=0.6)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'pruning_gap.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    meta = analyse_meta()
    if not meta.empty:
        meta.to_csv(P3 / 'meta_grid_results.csv', index=False)
        plot_meta(meta)
        dflt = meta[meta.varied == 'default'].iloc[0]
        print('── Meta-controller sensitivity (default reward '
              f'{dflt["reward"]:.4f}) ──')
        cols = ['varied', 'value', 'reward', 'delta_vs_default', 'pct_vs_default',
                'files', 'pruning']
        if 'p_vs_default' in meta:
            cols += ['p_vs_default', 'significant']
        print(meta[meta.varied != 'default'][cols].round(4).to_string(index=False))
        span = meta['reward'].max() - meta['reward'].min()
        print(f'\n  total spread across all configurations: {span:.4f} '
              f'({span / abs(dflt["reward"]) * 100:.1f}% of default)')

    out = analyse_pruning()
    if out is not None and not (isinstance(out, pd.DataFrame) and out.empty):
        prune, piv = out
        prune.to_csv(P3 / 'pruning_results.csv', index=False)
        piv.to_csv(P3 / 'pruning_gap.csv', index=False)
        plot_pruning(prune, piv)
        print('\n── Pruning-matrix perturbation ──')
        print(prune.pivot(index='scale', columns='policy',
                          values=['reward', 'pruning']).round(4).to_string())
        print('\n  RL advantage (DDQN − WAT) at each scale:')
        for _, r in piv.iterrows():
            print(f'    scale {r["scale"]:.1f}: {r["gap"]:+.4f}')

    print(f'\nwrote meta_grid_results.csv, pruning_results.csv, figures/ → {P3}')


if __name__ == '__main__':
    main()
