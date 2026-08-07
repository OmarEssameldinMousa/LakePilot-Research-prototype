"""
Phase 2 — training-cost and hyperparameter tables (from Phase 1 run manifests).

Reviewer demand: wall-clock training time, GPU requirements, parameter counts,
training-step counts, and a full hyperparameter appendix.

Every Phase 1 training run writes a manifest, so both tables are derived rather
than hand-transcribed. Inference-latency benchmarking is a separate script
(needs an idle CPU; must not run while an evaluation batch is in flight).

Usage:  python3 revision/phase2_compute/build_training_cost_table.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / 'revision' / 'phase1_stats' / 'phase1_weights' / 'weights'
OUT = ROOT / 'revision' / 'phase2_compute'


def _clean_hw(raw: str) -> str:
    """'Tesla T4\\nTesla T4' -> 'Tesla T4 ×2'; leaves other strings untouched."""
    parts = [p.strip() for p in str(raw).splitlines() if p.strip()]
    if not parts:
        return 'unknown'
    if len(set(parts)) == 1:
        return parts[0] if len(parts) == 1 else f'{parts[0]} ×{len(parts)}'
    return ', '.join(parts)


def load_manifests() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(WEIGHTS / '*_manifest.json'))):
        m = json.load(open(f))
        hp = m.get('hyperparams', {})
        rows.append({
            'label': m.get('label', m['arch']),
            'arch': m['arch'],
            'specialist': m['specialist'],
            'seed': m['seed'],
            'algorithm': m['algorithm'],
            'advantage': m.get('advantage_treatment', ''),
            'balance': m.get('class_balance', ''),
            'epochs': hp.get('epochs'),
            'batch': hp.get('batch'),
            'lr': hp.get('lr', 3e-4),
            'gamma': hp.get('gamma'),
            'window_size': hp.get('window_size'),
            'num_features': hp.get('num_features'),
            'num_actions': hp.get('num_actions'),
            'entropy_coef': hp.get('ent'),
            'clip_ratio': hp.get('clip_ratio'),
            'beta': hp.get('beta'),
            'target_freq': hp.get('target_freq'),
            'n_train_samples': m['n_train_samples'],
            'trainable_params': m['trainable_params'],
            'wall_clock_sec': m['wall_clock_sec'],
            # nvidia-smi lists one line per GPU; collapse duplicates for the table
            'hardware': _clean_hw(m.get('gpu', 'unknown')),
            'tf_version': m.get('tf_version', ''),
        })
    if not rows:
        raise SystemExit(f'no manifests under {WEIGHTS}')
    return pd.DataFrame(rows)


def training_cost_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (label, spec), g in df.groupby(['label', 'specialist']):
        out.append({
            'variant': label,
            'specialist': spec,
            'algorithm': g['algorithm'].iloc[0],
            'n_seeds': len(g),
            'hardware': g['hardware'].iloc[0],
            'trainable_params': int(g['trainable_params'].iloc[0]),
            'epochs': int(g['epochs'].iloc[0]),
            'batch_size': int(g['batch'].iloc[0]),
            'n_train_samples': int(g['n_train_samples'].iloc[0]),
            'wall_clock_mean_sec': round(g['wall_clock_sec'].mean(), 1),
            'wall_clock_sd_sec': round(g['wall_clock_sec'].std(ddof=1), 1) if len(g) > 1 else 0.0,
            'wall_clock_total_sec': round(g['wall_clock_sec'].sum(), 1),
        })
    t = pd.DataFrame(out).sort_values(['variant', 'specialist'])
    return t


def per_variant_totals(cost: pd.DataFrame) -> pd.DataFrame:
    """Total cost of producing one deployable agent (both specialists), and all seeds."""
    rows = []
    for variant, g in cost.groupby('variant'):
        per_seed = (g['wall_clock_mean_sec'].sum())
        rows.append({
            'variant': variant,
            'hardware': g['hardware'].iloc[0],
            'params_total': int(g['trainable_params'].sum()),
            'wall_clock_per_agent_sec': round(per_seed, 1),
            'wall_clock_per_agent_min': round(per_seed / 60, 2),
            'wall_clock_all_seeds_sec': round(g['wall_clock_total_sec'].sum(), 1),
            'wall_clock_all_seeds_min': round(g['wall_clock_total_sec'].sum() / 60, 2),
        })
    return pd.DataFrame(rows).sort_values('variant')


def hyperparameter_appendix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['label', 'specialist', 'algorithm', 'advantage', 'balance', 'epochs',
            'batch', 'lr', 'gamma', 'entropy_coef', 'clip_ratio', 'beta',
            'target_freq', 'window_size', 'num_features', 'num_actions',
            'n_train_samples', 'trainable_params']
    t = df[cols].drop_duplicates(subset=['label', 'specialist']).sort_values(['label', 'specialist'])
    t = t.rename(columns={'label': 'variant', 'batch': 'batch_size',
                          'lr': 'learning_rate', 'entropy_coef': 'entropy_coefficient'})
    # seeds are identical across variants by construction
    t['seeds'] = ', '.join(str(s) for s in sorted(df['seed'].unique()))
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_manifests()

    cost = training_cost_table(df)
    cost.to_csv(OUT / 'training_cost_table.csv', index=False)

    totals = per_variant_totals(cost)
    totals.to_csv(OUT / 'training_cost_per_agent.csv', index=False)

    hp = hyperparameter_appendix(df)
    hp.to_csv(OUT / 'hyperparameter_appendix.csv', index=False)

    pd.set_option('display.width', 200)
    print('── training cost (per specialist, mean ± sd over seeds) ──')
    print(cost.to_string(index=False))
    print()
    print('── cost to produce one deployable agent (both specialists) ──')
    print(totals.to_string(index=False))
    print()
    print(f'wrote training_cost_table.csv, training_cost_per_agent.csv, '
          f'hyperparameter_appendix.csv → {OUT}')
    print()
    print('NOTE: hardware differs across variants (Kaggle T4 vs local CPU) — the table')
    print('reports it per row; do not compare wall-clock across different hardware.')


if __name__ == '__main__':
    main()
