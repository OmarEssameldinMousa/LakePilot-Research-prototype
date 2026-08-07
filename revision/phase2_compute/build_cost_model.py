"""
Phase 2 — cost model extended with the RL-infrastructure term (paper Table 11).

Reviewer demand: "do cost savings survive when RL infrastructure cost is
included?" The published model charged only query time, compaction operations
and storage — i.e. it credited the RL agents with savings while ignoring the cost
of running and training them.

This adds:
  • inference compute: per-step hierarchical decision latency × cloud CPU price
  • amortised training: offline training wall-clock × cloud price / N episodes
    (N is a parameter; the break-even N is reported)

All unit prices are stated explicitly and swept in Phase 3.

Usage:  python3 revision/phase2_compute/build_cost_model.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'revision' / 'phase1_stats' / 'raw'
OUT = ROOT / 'revision' / 'phase2_compute'
PUBLISHED_RESULTS = ROOT / 'LakeGymLite' / 'results'

# ── Published cost model constants (unchanged, so the comparison is like-for-like)
QUERY_COST_PER_MS = 0.00001      # $ per ms of query time
COMPACT_COST_PER_OP = 0.005      # $ per compaction operation
STORAGE_COST_PER_KB = 0.000001   # $ per KB-step

# ── New: cloud compute prices for the RL infrastructure term ──────────────
# Representative on-demand rates (AWS us-east-1, 2026); swept in Phase 3.
CPU_USD_PER_HOUR = 0.096         # m5.large — serving the policy
GPU_USD_PER_HOUR = 0.526         # g4dn.xlarge (T4) — offline training
CPU_USD_PER_MS = CPU_USD_PER_HOUR / 3600 / 1000
GPU_USD_PER_SEC = GPU_USD_PER_HOUR / 3600

# Deployment horizon over which offline training is amortised
DEFAULT_N_EPISODES = 100

# Map evaluation label -> inference-benchmark variant name
VARIANT_OF = {
    'MultiAgentRL_ddqn': 'DDQN',
    'MultiAgentRL_attentive_ppoclip': 'AttentivePPO-clip',
    'MultiAgentRL_mlp_ppo': 'MLP-PPO',
}
# Map evaluation label -> training-manifest label
TRAIN_LABEL_OF = {
    'MultiAgentRL_ddqn': 'ddqn',
    'MultiAgentRL_attentive_ppoclip': 'attentive_ppoclip',
    'MultiAgentRL_mlp_ppo': 'mlp_ppo',
}


def per_episode_operational(policy: str, protocol: str = 'std1000') -> dict | None:
    """Mean per-episode query time, storage and compaction count from Phase 1 runs."""
    files = sorted(glob.glob(str(RAW / policy / 'seed*' / protocol / 'transitions_ep*.csv')))
    if not files:
        return None
    lat, size, ncomp, steps = [], [], [], []
    for f in files:
        d = pd.read_csv(f)
        lat.append(d['latency_ms'].sum())
        size.append(d['total_size_kb'].sum())
        ncomp.append(int(d['action_name'].str.contains('COMPACT', na=False).sum()))
        steps.append(len(d))
    return {'latency_ms_sum': float(np.mean(lat)), 'storage_kb_sum': float(np.mean(size)),
            'n_compactions': float(np.mean(ncomp)), 'steps': float(np.mean(steps)),
            'n_episodes': len(files)}


def published_no_maintenance() -> dict:
    """No-maintenance anchor from the published single-episode run (no CI; documented)."""
    f = sorted(glob.glob(str(PUBLISHED_RESULTS / 'No_Maintenance' / 'transitions_*_ep1.csv')))[0]
    d = pd.read_csv(f)
    return {'latency_ms_sum': float(d['latency_ms'].sum()),
            'storage_kb_sum': float(d['total_size_kb'].sum()),
            'n_compactions': 0.0, 'steps': float(len(d)), 'n_episodes': 1}


def operational_cost(m: dict) -> dict:
    q = m['latency_ms_sum'] * QUERY_COST_PER_MS
    c = m['n_compactions'] * COMPACT_COST_PER_OP
    s = m['storage_kb_sum'] * STORAGE_COST_PER_KB
    return {'query': q, 'compaction': c, 'storage': s, 'operational_total': q + c + s}


def main():
    bench = json.load(open(OUT / 'inference_latency.json'))
    train = pd.read_csv(OUT / 'training_cost_table.csv')

    rows = []
    # No-maintenance anchor (published run)
    nm = published_no_maintenance()
    nm_cost = operational_cost(nm)
    rows.append({'policy': 'No_Maintenance (published run)', 'variant': None,
                 'steps': nm['steps'], **nm_cost, 'inference': 0.0,
                 'training_amortised': 0.0, 'total': nm_cost['operational_total'],
                 'n_episodes_evaluated': nm['n_episodes']})

    # Heuristics (no RL infrastructure cost)
    for policy in ['WorkloadAwareThreshold', 'AlwaysCompact_C128',
                   'Threshold10_C128', 'Threshold10_C64']:
        m = per_episode_operational(policy)
        if not m:
            continue
        c = operational_cost(m)
        rows.append({'policy': policy, 'variant': None, 'steps': m['steps'], **c,
                     'inference': 0.0, 'training_amortised': 0.0,
                     'total': c['operational_total'], 'n_episodes_evaluated': m['n_episodes']})

    # RL agents (operational + inference + amortised training)
    for policy, variant in VARIANT_OF.items():
        m = per_episode_operational(policy)
        if not m:
            continue
        c = operational_cost(m)
        step_ms = bench['variants'][variant]['full_hierarchical_step']['median_ms']
        inference = m['steps'] * step_ms * CPU_USD_PER_MS
        tl = TRAIN_LABEL_OF[policy]
        train_sec = train[train['variant'] == tl]['wall_clock_mean_sec'].sum()
        train_cost = train_sec * GPU_USD_PER_SEC
        amortised = train_cost / DEFAULT_N_EPISODES
        rows.append({'policy': policy, 'variant': variant, 'steps': m['steps'], **c,
                     'inference': inference, 'training_amortised': amortised,
                     'total': c['operational_total'] + inference + amortised,
                     'n_episodes_evaluated': m['n_episodes'],
                     'inference_ms_per_step': step_ms,
                     'training_wall_clock_sec': train_sec,
                     'training_cost_usd': train_cost})

    t = pd.DataFrame(rows)
    base = float(t.loc[t.policy.str.startswith('No_Maintenance'), 'total'].iloc[0])
    t['vs_no_maintenance_pct'] = (base - t['total']) / abs(base) * 100
    t['operational_only_pct'] = (base - t['operational_total']) / abs(base) * 100
    t = t.sort_values('total')
    t.to_csv(OUT / 'cost_model_table11.csv', index=False)

    pd.set_option('display.width', 220)
    show = ['policy', 'query', 'compaction', 'storage', 'inference',
            'training_amortised', 'total', 'vs_no_maintenance_pct', 'operational_only_pct']
    print(f'── Table 11 (regenerated), 1,000-step episode, training amortised over '
          f'{DEFAULT_N_EPISODES} episodes ──')
    print(t[show].round(4).to_string(index=False))

    # Break-even: N at which RL total cost equals the best heuristic's cost
    best_h = t[(t.variant.isna()) & (~t.policy.str.startswith('No_Maintenance'))]
    if not best_h.empty:
        bh = best_h.loc[best_h['total'].idxmin()]
        print(f"\nBest heuristic by cost: {bh['policy']} = ${bh['total']:.4f}/episode")
        print('Break-even deployment horizon (episodes) for each RL agent vs that heuristic:')
        for _, r in t[t.variant.notna()].iterrows():
            per_ep_excl_train = r['operational_total'] + r['inference']
            margin = bh['total'] - per_ep_excl_train
            if margin <= 0:
                print(f"  {r['policy']:34s} never — costs more per episode even before training")
            else:
                n_be = r['training_cost_usd'] / margin
                print(f"  {r['policy']:34s} N = {n_be:.1f} episodes "
                      f"(saves ${margin:.4f}/episode, training ${r['training_cost_usd']:.4f})")

    with open(OUT / 'cost_model_assumptions.json', 'w') as f:
        json.dump({'query_cost_per_ms': QUERY_COST_PER_MS,
                   'compact_cost_per_op': COMPACT_COST_PER_OP,
                   'storage_cost_per_kb': STORAGE_COST_PER_KB,
                   'cpu_usd_per_hour': CPU_USD_PER_HOUR,
                   'gpu_usd_per_hour': GPU_USD_PER_HOUR,
                   'amortisation_episodes': DEFAULT_N_EPISODES,
                   'no_maintenance_source': 'published single-episode run (no CI)'}, f, indent=2)
    print(f'\nwrote cost_model_table11.csv + cost_model_assumptions.json → {OUT}')


if __name__ == '__main__':
    main()
