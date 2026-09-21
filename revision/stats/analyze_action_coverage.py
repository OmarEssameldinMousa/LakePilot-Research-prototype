"""
Action coverage of the two offline training pools (round-2 reviewer request).

Reviewer 2 (round 2): "Balanced action sampling addresses class imbalance but not
action coverage. Please report the action-coverage distribution in the two
offline pools and justify the absence of conservative regularisation,
particularly for the partition specialist, which has five actions and only 4,990
transitions."

Coverage — whether every action is *represented*, and in how many distinct
states — is the property that governs extrapolation error in offline value
learning. Class balance, which `balanced_indices` already treats, is a different
property: a pool can be perfectly balanced after resampling and still contain an
action taken in only one region of state space.

Reads the authoritative published pool
(revision/phase1_stats/kaggle_training_data_v2/), applies exactly the splits and
action filters used at training time (revision_train_local.py:44-45, 253-259),
and writes action_coverage.csv + a short verdict.

Usage:  python3 revision/stats/analyze_action_coverage.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / 'revision' / 'phase1_stats' / 'kaggle_training_data_v2'
OUT = ROOT / 'revision' / 'phase1_stats'

# Mirrors revision_train_local.py:44-45
PARTITION_SOURCES = ('WorkloadAwareThreshold', 'PartitionExploration', 'Random_Partition')
COMPACT_ACTIONS = {0: 'NOOP', 1: 'COMPACT_32KB', 2: 'COMPACT_64KB', 3: 'COMPACT_128KB'}
PARTITION_ACTIONS = {0: 'NOOP', 4: 'PARTITION_HOUR', 5: 'PARTITION_REGION',
                     6: 'PARTITION_EVENT_TYPE', 7: 'REMOVE_PARTITION'}

# State features used to judge how widely an action is spread through state space
STATE_COLS = ['file_count', 'block_utilization', 'partition_pruning_ratio',
              'ingestion_rate_rows_per_sec']


def load_pool() -> pd.DataFrame:
    # kaggle_training_data_v2 is the authoritative pool and already contains the
    # RecoveryExploration episodes added in round 1 (directory RecoveryExploration/);
    # revision/phase1_stats/recovery_data/ is the same data before packaging, so
    # globbing both would double-count it.
    files = sorted(glob.glob(str(POOL / '**' / '*.csv'), recursive=True))
    if not files:
        sys.exit(f'no training pool CSVs found under {POOL}')
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d['source_file'] = Path(f).name
        d['source_policy'] = d['policy_name'] if 'policy_name' in d else Path(f).parent.name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def split(df: pd.DataFrame):
    is_part = df['source_file'].str.contains('|'.join(PARTITION_SOURCES)) | \
              df['source_policy'].astype(str).str.contains('|'.join(PARTITION_SOURCES))
    pool_p = df[is_part & df['action'].isin(PARTITION_ACTIONS)]
    pool_c = df[~is_part & df['action'].isin(COMPACT_ACTIONS)]
    return pool_c, pool_p


def coverage(pool: pd.DataFrame, actions: dict, name: str) -> pd.DataFrame:
    rows = []
    n = len(pool)
    for a, label in actions.items():
        g = pool[pool['action'] == a]
        rec = {'pool': name, 'action_id': a, 'action': label,
               'count': len(g), 'share_pct': round(100 * len(g) / n, 2),
               'n_source_policies': g['source_policy'].nunique() if len(g) else 0}
        # how much of the observed state space this action is taken in:
        # coverage of the file-count range, which drives every maintenance decision
        if len(g):
            fc = g['file_count']
            rec['file_count_p5'] = round(float(fc.quantile(0.05)), 1)
            rec['file_count_p95'] = round(float(fc.quantile(0.95)), 1)
            # fraction of the pool-wide file-count deciles in which the action appears
            bins = pd.qcut(pool['file_count'], 10, duplicates='drop')
            rec['state_deciles_covered'] = int(bins[pool['action'] == a].nunique())
            rec['state_deciles_total'] = int(bins.nunique())
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    df = load_pool()
    pool_c, pool_p = split(df)
    cov_c = coverage(pool_c, COMPACT_ACTIONS, 'compaction')
    cov_p = coverage(pool_p, PARTITION_ACTIONS, 'partition')
    out = pd.concat([cov_c, cov_p], ignore_index=True)
    out.to_csv(OUT / 'action_coverage.csv', index=False)

    print(f'compaction pool: {len(pool_c):,} transitions from '
          f'{pool_c["source_policy"].nunique()} behaviour policies')
    print(f'partition pool:  {len(pool_p):,} transitions from '
          f'{pool_p["source_policy"].nunique()} behaviour policies\n')
    print(out.to_string(index=False))

    rare = out[out['count'] < 100]
    missing = out[out['count'] == 0]
    print('\nverdict:')
    print(f'  actions never observed:            {len(missing)}')
    print(f'  actions with fewer than 100 rows:  {len(rare)}')
    print(f'  minimum per-action count:          {out["count"].min():,}')
    print(f'  minimum state-decile coverage:     '
          f'{out["state_deciles_covered"].min()}/{out["state_deciles_total"].max()}')
    print(f'\nwrote {OUT / "action_coverage.csv"}')


if __name__ == '__main__':
    main()
