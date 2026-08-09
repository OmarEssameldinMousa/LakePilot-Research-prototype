"""
Phase 7 Part A analysis — does the attention encoder do what the paper claims?

Consumes the traces from `LakeGymLite/src/revision_phase7_attention.py`
(shape: steps x heads x query x key, captured every step of a frozen 1,000-step
rollout on the standard workload).

Answers three questions:
  (a) Does attention weight recent steps, or is it effectively uniform? A uniform
      profile means the encoder computes approximately what the global average
      pooling layer already provides for free.
  (b) Does it behave differently around the four workload transitions
      (steps 200/400/600/800) than mid-phase?
  (c) Statistic + test: attention mass on the 3 most recent positions,
      transition windows vs mid-phase.

Outputs into revision/phase7_attention/:
    attention_profile.csv        mass per key position, per specialist, per seed
    attention_transitions.csv    recent-mass at transition vs mid-phase + test
    figures/attention_profile.{pdf,png}
    figures/attention_transitions.{pdf,png}

Usage:  python3 revision/stats/analyze_phase7_attention.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
P7 = ROOT / 'revision' / 'phase7_attention'
TRACES = P7 / 'attention_traces'
FIGS = P7 / 'figures'
RECENT_K = 3


def load_traces():
    out = {}
    for f in sorted(TRACES.glob('attention_seed*.npz')):
        seed = int(f.stem.replace('attention_seed', ''))
        out[seed] = np.load(f)
    return out


def main():
    traces = load_traces()
    if not traces:
        sys.exit(f'no traces under {TRACES}')
    print(f'loaded seeds: {sorted(traces)}')

    prof_rows, trans_rows = [], []
    for seed, z in traces.items():
        steps = z['steps']
        transitions, halfwin = z['transitions'], int(z['halfwin'])
        near = np.zeros(len(steps), dtype=bool)
        for t in transitions:
            near |= np.abs(steps - t) <= halfwin

        for spec in ['compaction', 'partition']:
            a = z[spec]                       # (steps, heads, q, k)
            W = a.shape[-1]
            # mass received by each key position, averaged over heads and queries
            per_step = a.mean(axis=(1, 2))    # (steps, k)
            prof = per_step.mean(axis=0)
            for pos, m in enumerate(prof):
                prof_rows.append({'seed': seed, 'specialist': spec, 'window': W,
                                  'position': pos, 'from_end': W - 1 - pos,
                                  'mass': float(m), 'uniform': 1.0 / W})

            recent = per_step[:, -RECENT_K:].sum(axis=1)   # per-step recent mass
            tr, mid = recent[near], recent[~near]
            u, p = sps.mannwhitneyu(tr, mid, alternative='two-sided')
            # rank-biserial effect size
            rb = 1 - 2 * u / (len(tr) * len(mid))
            trans_rows.append({
                'seed': seed, 'specialist': spec, 'window': W,
                'recent_mass_overall': float(recent.mean()),
                'uniform_recent_mass': RECENT_K / W,
                'ratio_vs_uniform': float(recent.mean() / (RECENT_K / W)),
                'recent_mass_transition': float(tr.mean()),
                'recent_mass_midphase': float(mid.mean()),
                'delta': float(tr.mean() - mid.mean()),
                'mannwhitney_p': float(p), 'rank_biserial': float(rb),
                'n_transition': int(near.sum()), 'n_mid': int((~near).sum()),
            })

    prof = pd.DataFrame(prof_rows)
    trans = pd.DataFrame(trans_rows)
    prof.to_csv(P7 / 'attention_profile.csv', index=False)
    trans.to_csv(P7 / 'attention_transitions.csv', index=False)

    pd.set_option('display.width', 200)
    print('\n── (a) Attention mass by position (mean over seeds) ──')
    for spec in ['compaction', 'partition']:
        s = prof[prof.specialist == spec].groupby('from_end')['mass'].mean()
        W = int(prof[prof.specialist == spec]['window'].iloc[0])
        print(f'  {spec} (window {W}, uniform={1/W:.4f}):')
        head = s.sort_index().head(6)
        print('    steps back from newest:', ', '.join(f'{i}:{v:.4f}' for i, v in head.items()))
        recent = s.loc[[0, 1, 2]].sum()
        print(f'    mass on {RECENT_K} most recent = {recent:.4f} '
              f'(uniform {RECENT_K/W:.4f}, ratio {recent/(RECENT_K/W):.2f}x)')

    print('\n── (b,c) Transition windows vs mid-phase ──')
    cols = ['seed', 'specialist', 'recent_mass_transition', 'recent_mass_midphase',
            'delta', 'mannwhitney_p', 'rank_biserial']
    print(trans[cols].round(4).to_string(index=False))
    print('\n  aggregate (mean over seeds):')
    for spec in ['compaction', 'partition']:
        s = trans[trans.specialist == spec]
        print(f"    {spec:11s} transition={s.recent_mass_transition.mean():.4f}  "
              f"mid={s.recent_mass_midphase.mean():.4f}  "
              f"delta={s.delta.mean():+.4f}  "
              f"seeds with p<0.05: {(s.mannwhitney_p < 0.05).sum()}/{len(s)}")

    # ── figures ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        FIGS.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, spec, colour in zip(axes, ['compaction', 'partition'], ['#185FA5', '#1D9E75']):
            s = prof[prof.specialist == spec]
            W = int(s['window'].iloc[0])
            g = s.groupby('from_end')['mass'].agg(['mean', 'std']).sort_index()
            ax.plot(g.index, g['mean'], marker='o', color=colour, label='attention')
            ax.fill_between(g.index, g['mean'] - g['std'].fillna(0),
                            g['mean'] + g['std'].fillna(0), color=colour, alpha=0.2)
            ax.axhline(1.0 / W, ls='--', color='#E24B4A', label=f'uniform (1/{W})')
            ax.set_xlabel('steps back from newest'); ax.set_ylabel('attention mass')
            ax.set_title(f'{spec} (window {W})', fontweight='bold', loc='left')
            ax.invert_xaxis(); ax.legend(frameon=False)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(color='#D3D1C7', lw=0.5, alpha=0.6)
        fig.suptitle('Attention mass by window position', fontweight='bold')
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(FIGS / f'attention_profile.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # transition timeline for seed 1
        z = traces[sorted(traces)[0]]
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for ax, spec, colour in zip(axes, ['compaction', 'partition'], ['#185FA5', '#1D9E75']):
            a = z[spec]; W = a.shape[-1]
            recent = a.mean(axis=(1, 2))[:, -RECENT_K:].sum(axis=1)
            sm = pd.Series(recent).rolling(20, min_periods=1).mean()
            ax.plot(z['steps'], sm, color=colour, lw=1.4)
            ax.axhline(RECENT_K / W, ls='--', color='#E24B4A', lw=1,
                       label=f'uniform ({RECENT_K}/{W})')
            for t in z['transitions']:
                ax.axvline(t, color='#888780', ls=':', lw=1)
            ax.set_ylabel(f'{spec}\nrecent-{RECENT_K} mass')
            ax.legend(frameon=False, loc='upper right')
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(color='#D3D1C7', lw=0.5, alpha=0.6)
        axes[-1].set_xlabel('step (dotted lines = workload phase transitions)')
        fig.suptitle('Attention on recent steps across the workload', fontweight='bold')
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(FIGS / f'attention_transitions.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'  ! figures failed: {e}')

    print(f'\nwrote attention_profile.csv, attention_transitions.csv, figures/ → {P7}')


if __name__ == '__main__':
    main()
