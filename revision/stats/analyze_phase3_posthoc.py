"""
Phase 3 — post-hoc sensitivity: cost-model prices and reward weights.

Neither analysis needs the environment: both recompute from per-step data already
logged in the Phase 1 transitions CSVs.

1. COST-MODEL SWEEP
   Unit prices (query / compaction / storage) each at {0.5x, 1x, 2x}, and — the
   more consequential variant — a SIZE-PROPORTIONAL compaction cost.

   The published model charges a flat $0.005 per compaction regardless of bytes
   rewritten. A real compaction is a distributed rewrite whose cost scales with
   the data it touches; at production scale one operation can occupy many workers
   for minutes. Flat pricing therefore systematically under-charges aggressive
   strategies. The size-proportional variant charges
       cost = RATE x total_size_kb at the moment of compaction
   calibrated so that the mean compaction under the default workload costs the
   same $0.005 — so the two models differ only in how cost is *distributed*
   across policies, not in overall scale.

   Question: does the cost ranking flip?

2. REWARD-WEIGHT SENSITIVITY
   The global reward is
       R = -0.30*latency - 0.20*files + 0.30*pruning + 0.20*utilisation
   Each weight is perturbed +/-25% and the four renormalised to sum to 1, then the
   global reward is recomputed per step from logged metrics and policies re-ranked.

   Question: is the policy ranking stable under reasonable reward re-weighting?

Usage:  python3 revision/stats/analyze_phase3_posthoc.py
"""
from __future__ import annotations

import glob
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'revision' / 'phase1_stats' / 'raw'
PUB = ROOT / 'LakeGymLite' / 'results'
OUT = ROOT / 'revision' / 'phase3_sensitivity'

# Published cost constants
QUERY, COMPACT, STORAGE = 0.00001, 0.005, 0.000001
# Published reward weights: latency, files, pruning, utilisation
W0 = {'latency': 0.30, 'files': 0.20, 'pruning': 0.30, 'util': 0.20}
LATENCY_MAX, FILE_COUNT_MAX = 15000.0, 2000.0

POLICIES = {
    'MultiAgentRL_ddqn': 'DDQN',
    'MultiAgentRL_attentive_ppoclip': 'AttentivePPO-clip',
    'MultiAgentRL_mlp_ppo': 'MLP-PPO',
    'WorkloadAwareThreshold': 'WorkloadAwareThreshold',
    'AlwaysCompact_C128': 'AlwaysCompact_C128',
    'Threshold10_C128': 'Threshold10_C128',
    'Threshold10_C64': 'Threshold10_C64',
    'CompactOnlyRL_attentive_ppoclip': 'CompactOnly',
}


def load_episodes(protocol='std1000') -> dict[str, list[pd.DataFrame]]:
    out = {}
    for key, name in POLICIES.items():
        files = sorted(glob.glob(str(RAW / key / 'seed*' / protocol / 'transitions_ep*.csv')))
        if files:
            out[name] = [pd.read_csv(f) for f in files]
    nm = sorted(glob.glob(str(PUB / 'No_Maintenance' / 'transitions_*_ep1.csv')))
    if nm:
        out['No_Maintenance'] = [pd.read_csv(nm[0])]
    return out


# ── 1. cost model ────────────────────────────────────────────────────────
def episode_cost(d: pd.DataFrame, q=QUERY, c=COMPACT, s=STORAGE,
                 size_proportional=False, rate=None) -> float:
    query = d['latency_ms'].sum() * q
    storage = d['total_size_kb'].sum() * s
    mask = d['action_name'].str.contains('COMPACT', na=False)
    if size_proportional:
        compaction = (d.loc[mask, 'total_size_kb'] * rate).sum()
    else:
        compaction = int(mask.sum()) * c
    return query + compaction + storage


def calibrate_rate(eps: dict) -> float:
    """Rate such that the mean compaction costs the published $0.005."""
    sizes = []
    for frames in eps.values():
        for d in frames:
            m = d['action_name'].str.contains('COMPACT', na=False)
            sizes.extend(d.loc[m, 'total_size_kb'].tolist())
    return COMPACT / float(np.mean(sizes)) if sizes else 0.0


def cost_sweep(eps: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rate = calibrate_rate(eps)
    rows = []
    for qm, cm, sm in itertools.product([0.5, 1.0, 2.0], repeat=3):
        for name, frames in eps.items():
            v = np.mean([episode_cost(d, QUERY * qm, COMPACT * cm, STORAGE * sm)
                         for d in frames])
            rows.append({'query_mult': qm, 'compact_mult': cm, 'storage_mult': sm,
                         'policy': name, 'cost': v, 'model': 'flat'})
    flat = pd.DataFrame(rows)

    rows2 = []
    for name, frames in eps.items():
        flat_v = np.mean([episode_cost(d) for d in frames])
        prop_v = np.mean([episode_cost(d, size_proportional=True, rate=rate) for d in frames])
        n_comp = np.mean([int(d['action_name'].str.contains('COMPACT', na=False).sum())
                          for d in frames])
        mean_size = np.mean([d.loc[d['action_name'].str.contains('COMPACT', na=False),
                                   'total_size_kb'].mean() for d in frames])
        rows2.append({'policy': name, 'cost_flat': flat_v, 'cost_size_proportional': prop_v,
                      'n_compactions': n_comp, 'mean_table_kb_at_compaction': mean_size})
    prop = pd.DataFrame(rows2).sort_values('cost_flat')
    prop['rank_flat'] = prop['cost_flat'].rank()
    prop['rank_size_proportional'] = prop['cost_size_proportional'].rank()
    prop['rank_change'] = prop['rank_size_proportional'] - prop['rank_flat']
    return flat, prop


# ── 2. reward weights ────────────────────────────────────────────────────
def recompute_reward(d: pd.DataFrame, w: dict) -> float:
    l = np.clip(d['latency_ms'].values / LATENCY_MAX, 0, 1)
    f = np.clip(d['file_count'].values / FILE_COUNT_MAX, 0, 1)
    p = np.clip(d['partition_pruning_ratio'].values, 0, 1)
    u = np.clip(d['block_utilization'].values, 0, 1)
    return float(np.mean(-w['latency'] * l - w['files'] * f + w['pruning'] * p + w['util'] * u))


def weight_sweep(eps: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = {'published': dict(W0)}
    for k in W0:
        for sign, tag in [(1.25, '+25%'), (0.75, '-25%')]:
            w = dict(W0)
            w[k] = W0[k] * sign
            total = sum(w.values())
            variants[f'{k} {tag}'] = {kk: vv / total for kk, vv in w.items()}

    rows = []
    for vname, w in variants.items():
        for name, frames in eps.items():
            r = np.mean([recompute_reward(d, w) for d in frames])
            rows.append({'variant': vname, 'policy': name, 'reward': r})
    t = pd.DataFrame(rows)
    piv = t.pivot(index='policy', columns='variant', values='reward')
    base = piv['published'].rank(ascending=False)
    stab = []
    for v in piv.columns:
        rk = piv[v].rank(ascending=False)
        rho, _ = sps.spearmanr(base, rk)
        top_same = base.idxmin() == rk.idxmin()
        stab.append({'variant': v, 'spearman_vs_published': rho,
                     'top_policy': rk.idxmin(), 'top_unchanged': top_same,
                     'n_rank_changes': int((base != rk).sum())})
    return piv, pd.DataFrame(stab)


def main():
    eps = load_episodes()
    print(f'loaded {len(eps)} policies: {", ".join(eps)}\n')

    flat, prop = cost_sweep(eps)
    flat.to_csv(OUT / 'cost_price_sweep.csv', index=False)
    prop.to_csv(OUT / 'cost_size_proportional.csv', index=False)

    print('── Compaction cost: flat vs size-proportional (calibrated to equal mean) ──')
    print(prop.round(4).to_string(index=False))

    print('\n── Price sweep: does the cheapest policy change? ──')
    winners = flat.loc[flat.groupby(['query_mult', 'compact_mult', 'storage_mult'])['cost'].idxmin()]
    print(winners['policy'].value_counts().to_string())
    rl = {'DDQN', 'AttentivePPO-clip', 'MLP-PPO'}
    nm_rank = []
    for _, g in flat.groupby(['query_mult', 'compact_mult', 'storage_mult']):
        g = g.sort_values('cost')
        order = list(g['policy'])
        nm_rank.append(order.index('No_Maintenance') if 'No_Maintenance' in order else np.nan)
    print(f"\nNo_Maintenance rank across all 27 price combinations: "
          f"always {int(np.nanmin(nm_rank))+1}-{int(np.nanmax(nm_rank))+1} of {len(eps)} "
          f"(1 = cheapest)")

    piv, stab = weight_sweep(eps)
    piv.to_csv(OUT / 'reward_weight_sweep.csv')
    stab.to_csv(OUT / 'reward_weight_stability.csv', index=False)
    print('\n── Reward-weight sensitivity (+/-25% per weight, renormalised) ──')
    print(stab.round(4).to_string(index=False))
    print(f'\nwrote cost_price_sweep.csv, cost_size_proportional.csv, '
          f'reward_weight_sweep.csv, reward_weight_stability.csv → {OUT}')


if __name__ == '__main__':
    main()
