"""
Revision — checkpoint sanity gate (run BEFORE any long evaluation batch).

Cheap synthetic-state probes that catch the failure modes we have actually hit:

  P1  stale, file-heavy table (files=300, util=0.10, ssc=50)
      → the state a freshly deployed agent falls into. Must NOT be NOOP.
        (v1/v2 AttentivePPO NOOP-locked here and burned a 1000-step episode.)

  P2  compaction target vs ingestion rate (files=60, ssc=8; rows 35/80/250)
      → should compact, and ideally shift target upward as ingestion grows.

  P3  partition strategy vs dominant query type (unpartitioned, pruning=0)
      → time-heavy→HOUR, region-heavy→REGION, type-heavy→EVENT_TYPE.

Usage:
    docker exec -w /app/src lakegym-lite python3 revision_probe_checkpoints.py \
        --weights /app/revision/phase1_stats/phase1_weights/weights

Exit code 1 if any architecture fails P1 on a majority of seeds — i.e. the
evaluation batch should not be launched.
"""
from __future__ import annotations

import argparse
import sys

import policies  # noqa: F401  (import first: avoids agents circular import)
import numpy as np
import tensorflow as tf

from policies.base import Observation
from agents.compaction_agent import CompactionAgent
from agents.partition_agent import PartitionAgent

NAMES_C = ['NOOP', 'C32', 'C64', 'C128']
NAMES_P = ['NOOP', 'HOUR', 'REGION', 'EVENT_TYPE', 'REMOVE']

# (weight-file label, model architecture). Labels may differ from architectures
# when the same encoder is trained under a different objective — e.g.
# 'attentive_ppoclip' is the attentive encoder trained with PPO-clip.
DEFAULT_VARIANTS = [
    ('attentive_ppo', 'attentive_ppo'),
    ('mlp_ppo', 'mlp_ppo'),
    ('ddqn', 'ddqn'),
]

P3_CASES = [
    ('time',   (0.6, 0.1, 0.1, 0.1, 0.1), 'HOUR'),
    ('region', (0.1, 0.6, 0.1, 0.1, 0.1), 'REGION'),
    ('type',   (0.1, 0.1, 0.1, 0.6, 0.1), 'EVENT_TYPE'),
]


def steady_probs(agent, obs):
    """Fill the agent's window with one observation and return action probs."""
    agent.reset()
    for _ in range(agent.WINDOW_SIZE):
        agent.push_observation(obs)
    probs, _ = agent._model(tf.constant(agent.get_window()), training=False)
    return probs.numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='/app/revision/phase1_stats/phase1_weights/weights')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--variants', nargs='+', default=None,
                    help="label:architecture pairs, e.g. attentive_ppoclip:attentive_ppo. "
                         "Defaults to the three published architectures.")
    args = ap.parse_args()
    W = args.weights
    if args.variants:
        VARIANTS = [tuple(v.split(':')) if ':' in v else (v, v) for v in args.variants]
    else:
        VARIANTS = DEFAULT_VARIANTS

    obs_p1 = Observation(rows_ingested=80, ingestion_rate_rows_per_sec=250, latency_ms=2500,
                         file_count=300, block_utilization=0.10, total_size_kb=2500,
                         file_size_skew_kb=3.0, steps_since_compact=50)
    failures = []

    print('══ P1 — stale file-heavy table must trigger compaction ══')
    for label, arch in VARIANTS:
        acts = []
        for seed in args.seeds:
            a = CompactionAgent(weights_path=f'{W}/compaction_{label}_seed{seed}.weights.h5',
                                model_type=arch)
            acts.append(NAMES_C[int(steady_probs(a, obs_p1).argmax())])
        n_noop = sum(1 for x in acts if x == 'NOOP')
        ok = n_noop <= len(acts) // 2
        print(f'  [{"PASS" if ok else "FAIL"}] {label:20s} {acts}  (NOOP {n_noop}/{len(acts)})')
        if not ok:
            failures.append(f'{label}: P1 NOOP on {n_noop}/{len(acts)} seeds')

    print()
    print('══ P2 — compaction target vs ingestion rate ══')
    for label, arch in VARIANTS:
        for rows, rlabel in [(35, 'low'), (80, 'mid'), (250, 'burst')]:
            acts = []
            for seed in args.seeds:
                a = CompactionAgent(weights_path=f'{W}/compaction_{label}_seed{seed}.weights.h5',
                                    model_type=arch)
                obs = Observation(rows_ingested=rows, ingestion_rate_rows_per_sec=rows * 3.2,
                                  latency_ms=600, file_count=60, block_utilization=0.10,
                                  total_size_kb=500, file_size_skew_kb=2.0,
                                  steps_since_compact=8)
                acts.append(NAMES_C[int(steady_probs(a, obs).argmax())])
            print(f'  {label:20s} rows={rows:3d} ({rlabel:5s}): {acts}')
        print()

    print('══ P3 — partition strategy vs dominant query type ══')
    base = dict(latency_ms=800, file_count=40, partition_strategy=0,
                partition_pruning_ratio=0.0, avg_pruning_ratio=0.0,
                steps_since_partition_change=60)
    for label, arch in VARIANTS:
        hits = 0
        for dom, hist, want in P3_CASES:
            acts = []
            for seed in args.seeds:
                a = PartitionAgent(weights_path=f'{W}/partition_{label}_seed{seed}.weights.h5',
                                   model_type=arch)
                obs = Observation(**base, query_hist_time_range=hist[0],
                                  query_hist_region_filter=hist[1],
                                  query_hist_sensor_lookup=hist[2],
                                  query_hist_type_filter=hist[3],
                                  query_hist_full_scan=hist[4])
                acts.append(NAMES_P[int(steady_probs(a, obs).argmax())])
            n_right = sum(1 for x in acts if x == want)
            hits += n_right
            print(f'  {label:20s} {dom:6s}-heavy (want {want:10s}): {acts}  ({n_right}/{len(acts)})')
        total = len(P3_CASES) * len(args.seeds)
        print(f'  → {label}: {hits}/{total} workload-aligned\n')

    print('═' * 60)
    if failures:
        print('GATE FAILED — do not launch the evaluation batch:')
        for f in failures:
            print(f'  • {f}')
        sys.exit(1)
    print('GATE PASSED — checkpoints act on a degraded table; evaluation may proceed.')
    print('(P2/P3 are reported for interpretation, not gated: weak workload alignment is a')
    print(' legitimate finding to report, whereas a NOOP-locked agent is a pipeline defect.)')


if __name__ == '__main__':
    main()
