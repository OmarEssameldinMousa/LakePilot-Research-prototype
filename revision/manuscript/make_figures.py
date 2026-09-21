#!/usr/bin/env python3
"""
Generate every figure used by the revised manuscript into `charts/`.

Each figure is written as both PDF (vector, for LaTeX) and PNG (300 dpi, for
preview). Figures that already exist as analysis output in the phase directories
are copied rather than regenerated, so the manuscript and the phase summaries
cannot drift apart.

Usage:  python3 revision/manuscript/make_figures.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'charts'
OUT.mkdir(parents=True, exist_ok=True)

# Manuscript colour convention (matches \definecolor in the .tex preamble).
BLUE = '#185FA5'   # agentblue   — AttentivePPO
TEAL = '#1D9E75'   # agentteal   — MLP-PPO
CORAL = '#E8593C'  # agentcoral  — DDQN
GREY = '#8A8A8A'
DARKGREY = '#4A4A4A'

ARCH_COLOR = {
    'DDQN': CORAL,
    'AttentivePPO-clip': BLUE,
    'MLP-PPO': TEAL,
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.7,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
})


def save(fig, name: str):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  wrote {name}.pdf / .png')


def copy_existing(src: Path, name: str):
    """Copy an analysis-generated figure under its manuscript name."""
    n = 0
    for ext in ('pdf', 'png'):
        s = src.with_suffix('.' + ext)
        if s.exists():
            shutil.copy2(s, OUT / f'{name}.{ext}')
            n += 1
    print(f'  copied {name} ({n} formats) <- {src.relative_to(REV)}')


# ────────────────────────────────────────────────────────────────────────────
# Fig 1 — policy ranking with bootstrap 95 % CIs
# ────────────────────────────────────────────────────────────────────────────
def fig_ranking():
    d = pd.read_csv(REV / 'phase1_stats' / 'table1_rewritten_eval500.csv')
    d = d.sort_values('reward')

    labels, colors = [], []
    for p in d['policy']:
        short = p.replace(' (hierarchical RL)', '')
        is_rl = '(hierarchical RL)' in p
        labels.append(short + ('  *' if is_rl else ''))
        colors.append(ARCH_COLOR.get(short, GREY) if is_rl else
                      (DARKGREY if 'ablation' in p.lower() or 'Only' in p else '#B8B8B8'))

    y = np.arange(len(d))
    err = np.vstack([d['reward'] - d['ci_low'], d['ci_high'] - d['reward']])

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.barh(y, d['reward'], color=colors, height=0.68, edgecolor='white', linewidth=0.5)
    ax.errorbar(d['reward'], y, xerr=err, fmt='none', ecolor='#222222',
                elinewidth=0.9, capsize=2.5, capthick=0.9)
    ax.axvline(0, color='black', linewidth=0.7)

    # Reference line: best heuristic.
    best_h = d.loc[~d['policy'].str.contains('hierarchical RL|ablation', case=False,
                                             regex=True), 'reward'].max()
    ax.axvline(best_h, color=CORAL, linestyle=':', linewidth=1.0, alpha=0.8)
    ax.text(best_h, len(d) - 0.35, ' best heuristic', color=CORAL,
            fontsize=7.5, va='center')

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean global reward (95 % bootstrap CI)')
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    save(fig, 'fig_ranking')


# ────────────────────────────────────────────────────────────────────────────
# Fig 2 — per-seed reward: the variance a single checkpoint hides
# ────────────────────────────────────────────────────────────────────────────
def fig_seed_variability():
    policies = [
        ('MultiAgentRL_ddqn', 'DDQN', CORAL),
        ('MultiAgentRL_attentive_ppoclip', 'AttentivePPO-clip', BLUE),
        ('MultiAgentRL_mlp_ppo', 'MLP-PPO', TEAL),
        ('WorkloadAwareThreshold', 'WorkloadAware\nThreshold', GREY),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))

    for i, (dirname, label, color) in enumerate(policies):
        seed_means = []
        for sd in sorted((REV / 'phase1_stats' / 'raw' / dirname).glob('seed*')):
            f = sd / 'eval500' / 'episode_summary.csv'
            if f.exists():
                seed_means.append(pd.read_csv(f)['avg_global_reward'].mean())
        seed_means = np.array(seed_means)
        jitter = np.linspace(-0.13, 0.13, len(seed_means))
        ax.scatter(np.full(len(seed_means), i) + jitter, seed_means,
                   s=26, color=color, zorder=3, edgecolor='white', linewidth=0.6)
        ax.hlines(seed_means.mean(), i - 0.26, i + 0.26,
                  color=color, linewidth=2.0, zorder=2)
        # Annotate the spread, which is the point of the figure.
        ax.annotate(f'sd={seed_means.std(ddof=1):.4f}',
                    (i, seed_means.max()), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=7, color=color)

    best_h = pd.read_csv(REV / 'phase1_stats' / 'table1_rewritten_eval500.csv')
    best_h = best_h[best_h['policy'] == 'WorkloadAwareThreshold']['reward'].iloc[0]
    ax.axhline(best_h, color='#555555', linestyle='--', linewidth=0.8, zorder=1)

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels([p[1] for p in policies])
    ax.set_ylabel('Mean global reward per training seed')
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    save(fig, 'fig_seed_variability')


# ────────────────────────────────────────────────────────────────────────────
# Fig 3 — ablation: reward and pruning side by side
# ────────────────────────────────────────────────────────────────────────────
def fig_ablation():
    d = pd.read_csv(REV / 'phase1_stats' / 'table1_rewritten_eval500.csv')
    want = {
        'AttentivePPO-clip (hierarchical RL)': 'Full\nhierarchical',
        'CompactOnly (ablation)': 'CompactOnly',
        'PartitionOnly (ablation)': 'PartitionOnly',
    }
    sub = d[d['policy'].isin(want)].copy()
    sub['label'] = sub['policy'].map(want)
    sub = sub.set_index('label').loc[list(want.values())]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    x = np.arange(len(sub))
    cols = [BLUE, '#7FA8CF', '#C9C9C9']

    err = np.vstack([sub['reward'] - sub['ci_low'], sub['ci_high'] - sub['reward']])
    axes[0].bar(x, sub['reward'], color=cols, width=0.6, edgecolor='white')
    axes[0].errorbar(x, sub['reward'], yerr=err, fmt='none', ecolor='#222222',
                     elinewidth=0.9, capsize=3)
    axes[0].axhline(0, color='black', linewidth=0.7)
    axes[0].set_ylabel('Mean global reward')
    axes[0].set_title('(a) Reward')

    axes[1].bar(x, sub['pruning'], color=cols, width=0.6, edgecolor='white')
    axes[1].set_ylabel('Mean pruning ratio')
    axes[1].set_title('(b) Pruning')
    for i, v in enumerate(sub['pruning']):
        axes[1].text(i, v + 0.006, f'{v:.3f}', ha='center', fontsize=7.5)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, fontsize=8)
        ax.grid(axis='y')
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, 'fig_ablation')


# ────────────────────────────────────────────────────────────────────────────
# Fig 4 — cost decomposition including the RL infrastructure term
# ────────────────────────────────────────────────────────────────────────────
def fig_cost():
    d = pd.read_csv(REV / 'phase2_compute' / 'cost_model_table11.csv')
    d = d.sort_values('total')
    name = d['variant'].fillna('').where(d['variant'].notna() & (d['variant'] != ''),
                                         d['policy'])
    labels = [n.replace('MultiAgentRL_', '') for n in name]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(d))
    bottom = np.zeros(len(d))
    parts = [
        ('query', '#3C6E9F', 'Query latency'),
        ('compaction', '#E8A33C', 'Compaction ops'),
        ('storage', '#7FB07F', 'Storage'),
    ]
    for col, color, lab in parts:
        ax.bar(x, d[col], bottom=bottom, color=color, width=0.62,
               edgecolor='white', linewidth=0.5, label=lab)
        bottom += d[col].values

    # The RL infrastructure term is real but ~0.003 % of the total, so it is
    # invisible as a stack segment. Mark which policies carry it instead.
    rl = (d['inference'].fillna(0) + d['training_amortised'].fillna(0)).values
    for i, (tot, r) in enumerate(zip(d['total'], rl)):
        ax.text(i, tot + 0.6, f'\\${tot:.2f}' + (r'$^{\dagger}$' if r > 0 else ''),
                ha='center', fontsize=7.5)
    ax.text(0.99, 0.72, r'$\dagger$ includes RL inference + amortised'
                        '\n   training ($<0.004$\\% of total)',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.5,
            color=DARKGREY)

    ax.set_ylim(0, d['total'].max() * 1.14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('Cost per 1,000-step episode (USD)')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    save(fig, 'fig_cost')


# ────────────────────────────────────────────────────────────────────────────
# Fig 5 — online adaptation, per episode, with across-seed spread
# ────────────────────────────────────────────────────────────────────────────
def fig_adaptive():
    """Adaptation against its own controls.

    The round-1 version plotted only the adapting runs, which left the reader to
    compare them against a Table 1 number measured under a different
    action-selection rule. Each adapting curve now carries the frozen control
    (arm A) it should be read against, and MLP-PPO additionally carries arm B
    (argmax, no updates), whose separation is the action-selection effect.
    """
    import json

    def curves(root, label_dir, done_only=True):
        out = []
        for sd in sorted((root / label_dir).glob('seed*')):
            f, man = sd / 'episode_log.csv', sd / 'manifest.json'
            if not f.exists():
                continue
            if done_only and man.exists() and \
                    json.loads(man.read_text()).get('status') != 'done':
                continue
            out.append(pd.read_csv(f).sort_values('episode')['avg_global_reward'].values)
        return out

    ADAPT = REV / 'phase6_ddqn' / 'online_raw'
    FROZ = REV / 'phase10_r2' / 'frozen_raw'
    runs = [
        ('DDQN', CORAL, 'adaptive_ddqn', 'frozen_ddqn', None),
        ('AttentivePPO-clip', BLUE, 'adaptive_attentive_ppoclip',
         'frozen_attentive_ppoclip', None),
        ('MLP-PPO', TEAL, 'adaptive_mlp_ppo', 'frozen_mlp_ppo',
         'frozen_greedy_mlp_ppo'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), sharey=True)

    for ax, (label, color, adapt_dir, froz_dir, greedy_dir) in zip(axes, runs):
        for cs, style, name in [(curves(ADAPT, adapt_dir), '-', 'adaptive'),
                                (curves(FROZ, froz_dir), '--', 'frozen control'),
                                (curves(FROZ, greedy_dir) if greedy_dir else [],
                                 ':', 'frozen, argmax')]:
            if not cs:
                continue
            n = min(len(c) for c in cs)
            M = np.vstack([c[:n] for c in cs])
            ep = np.arange(1, n + 1)
            mean = M.mean(axis=0)
            se = M.std(axis=0, ddof=1) / np.sqrt(M.shape[0])
            ax.plot(ep, mean, style, marker='o', markersize=3.0, color=color,
                    linewidth=1.4, label=f'{name} (n={M.shape[0]})')
            if style != ':':
                ax.fill_between(ep, mean - se, mean + se, color=color,
                                alpha=0.15, linewidth=0)
        ax.set_title(label, fontweight='bold', loc='left', fontsize=9)
        ax.set_xlabel('Adaptation episode')
        ax.set_xticks(range(1, 6))
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc='lower right', fontsize=7)
    axes[0].set_ylabel('Mean global reward')
    save(fig, 'fig_adaptive')


# ────────────────────────────────────────────────────────────────────────────
# Fig 6 — capacity-matched attention vs MLP
# ────────────────────────────────────────────────────────────────────────────
def fig_matched_mlp():
    # Values from phase7_attention/SUMMARY.md (parameter-matched comparison).
    rows = [
        ('AttentivePPO-clip\n170,571 params', 0.1823, 0.1669, 0.1963, BLUE),
        ('Matched MLP\n170,667 params', 0.1774, 0.1664, 0.1885, TEAL),
        ('Published MLP-PPO\n192,203 params', 0.1675, 0.1579, 0.1772, '#9FD0BC'),
    ]
    fig, ax = plt.subplots(figsize=(4.9, 2.9))
    x = np.arange(len(rows))
    vals = [r[1] for r in rows]
    err = np.vstack([[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]])
    ax.bar(x, vals, color=[r[4] for r in rows], width=0.58, edgecolor='white')
    ax.errorbar(x, vals, yerr=err, fmt='none', ecolor='#222222',
                elinewidth=0.9, capsize=3)

    # The non-significant comparison is the point of the figure.
    top = max(r[3] for r in rows[:2]) + 0.006
    ax.plot([0, 0, 1, 1], [top, top + 0.004, top + 0.004, top],
            color='#333333', linewidth=0.8)
    ax.text(0.5, top + 0.006, '$p = 0.54$,  $\\delta = +0.16$ (n.s.)',
            ha='center', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_ylabel('Mean global reward')
    ax.set_ylim(0.14, top + 0.016)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    save(fig, 'fig_matched_mlp')


# ────────────────────────────────────────────────────────────────────────────
# Fig 7 — pruning matrix (assumed, not measured — stated in the caption)
# ────────────────────────────────────────────────────────────────────────────
def fig_pruning_matrix():
    queries = ['Time range', 'Sensor lookup', 'Region filter', 'Type filter', 'Full scan']
    strategies = ['Hour', 'Region', 'Event type', 'None']
    M = np.array([
        [0.92, 0.00, 0.00, 0.00],
        [0.75, 0.00, 0.00, 0.00],
        [0.00, 0.67, 0.00, 0.00],
        [0.00, 0.00, 0.67, 0.00],
        [0.00, 0.00, 0.00, 0.00],
    ])
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(strategies)), strategies, fontsize=8)
    ax.set_yticks(range(len(queries)), queries, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, '—' if v == 0 else f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if v > 0.6 else '#333333')
    ax.set_xlabel('Partition strategy')
    ax.set_ylabel('Query type')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Fraction of files skipped')
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, 'fig_pruning_matrix')


# ────────────────────────────────────────────────────────────────────────────
# Fig 8 — workload: ingestion schedule and query-profile phases
# ────────────────────────────────────────────────────────────────────────────
def fig_workload_ingestion():
    phases = [(1, 100, 40, 'warm-up'), (101, 300, 75, 'normal'),
              (301, 500, 225, 'burst'), (501, 700, 55, 'recovery'),
              (701, 900, 75, 'normal 2'), (901, 1000, 35, 'wind-down')]
    profiles = ['time_heavy', 'region_heavy', 'mixed', 'type_heavy', 'scan_heavy']

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for lo, hi, rate, name in phases:
        ax.fill_between([lo, hi], 0, rate, color=BLUE, alpha=0.22, linewidth=0)
        ax.plot([lo, hi], [rate, rate], color=BLUE, linewidth=1.6)
        ax.text((lo + hi) / 2, rate + 8, f'{rate}', ha='center', fontsize=7,
                color=BLUE)

    # Query-profile phase boundaries (five equal phases of 199 steps).
    for k, prof in enumerate(profiles):
        lo = k * 199 + 1
        ax.axvline(lo, color='#999999', linestyle=':', linewidth=0.8)
        ax.text(lo + 8, 238, prof, fontsize=6.5, rotation=90,
                va='top', color='#666666')

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 260)
    ax.set_xlabel('Step')
    ax.set_ylabel('Ingestion rate (rows/step)')
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    save(fig, 'fig_workload_ingestion')


# ────────────────────────────────────────────────────────────────────────────
def main():
    print('Generating manuscript figures ->', OUT)

    print('\n[regenerated from data]')
    fig_ranking()
    fig_seed_variability()
    fig_ablation()
    fig_cost()
    fig_adaptive()
    fig_matched_mlp()
    fig_pruning_matrix()
    fig_workload_ingestion()

    print('\n[copied from phase analysis output]')
    copy_existing(REV / 'phase1_stats' / 'figures' / 'forest_reward_eval500.pdf',
                  'fig_forest_eval500')
    copy_existing(REV / 'phase1_stats' / 'figures' / 'forest_reward_std1000.pdf',
                  'fig_forest_std1000')
    copy_existing(REV / 'phase3_sensitivity' / 'figures' / 'meta_tornado.pdf',
                  'fig_meta_tornado')
    copy_existing(REV / 'phase3_sensitivity' / 'figures' / 'pruning_gap.pdf',
                  'fig_pruning_gap')
    copy_existing(REV / 'phase5_generalization' / 'figures' / 'degradation.pdf',
                  'fig_generalization')
    copy_existing(REV / 'phase7_attention' / 'figures' / 'attention_profile.pdf',
                  'fig_attention_profile')
    copy_existing(REV / 'phase7_attention' / 'figures' / 'attention_transitions.pdf',
                  'fig_attention_transitions')

    n = len(list(OUT.glob('*.pdf')))
    print(f'\n{n} figures in {OUT}')


if __name__ == '__main__':
    main()
