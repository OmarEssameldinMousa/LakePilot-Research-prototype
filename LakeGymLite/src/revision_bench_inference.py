"""
Phase 2 — per-step inference latency benchmark.

Reviewer demand: per-step inference latency and parameter counts, so the
RL system's runtime overhead can be weighed against its benefit.

Measures, per architecture:
  • MetaController.decide()      — the rule-based urgency computation
  • CompactionAgent.predict()    — feature extraction + forward pass (window 10)
  • PartitionAgent.predict()     — feature extraction + forward pass (window 20)
  • MultiAgentPolicy.get_action()— the full hierarchical decision actually paid per step

Reports median and p95 over N iterations (default 10,000), plus trainable
parameter counts. Run on an otherwise idle machine.

Usage:
    docker exec -w /app/src lakegym-lite python3 revision_bench_inference.py \
        --iterations 10000 --out /app/revision/phase2_compute
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

import policies  # noqa: F401  (import first: avoids agents circular import)
import numpy as np

from policies.base import Observation
from policies.multi_agent_policy import MultiAgentPolicy
from agents.meta_controller import MetaController

W = '/app/revision/phase1_stats/phase1_weights/weights'
VARIANTS = [
    ('DDQN', 'ddqn', 'ddqn'),
    ('AttentivePPO-clip', 'attentive_ppo', 'attentive_ppoclip'),
    ('MLP-PPO', 'mlp_ppo', 'mlp_ppo'),
]


def make_obs(rng) -> Observation:
    """A plausible mid-episode observation with randomised state."""
    return Observation(
        rows_ingested=int(rng.integers(20, 300)),
        ingestion_rate_rows_per_sec=float(rng.uniform(50, 900)),
        latency_ms=float(rng.uniform(100, 3000)),
        file_count=int(rng.integers(5, 400)),
        block_utilization=float(rng.uniform(0.05, 1.2)),
        total_size_kb=float(rng.uniform(200, 20000)),
        file_size_skew_kb=float(rng.uniform(0, 20)),
        partition_strategy=int(rng.integers(0, 4)),
        steps_since_compact=int(rng.integers(0, 60)),
        steps_since_partition_change=int(rng.integers(0, 200)),
        partition_pruning_ratio=float(rng.uniform(0, 0.9)),
        avg_pruning_ratio=float(rng.uniform(0, 0.9)),
        query_hist_time_range=0.2, query_hist_region_filter=0.2,
        query_hist_sensor_lookup=0.2, query_hist_type_filter=0.2,
        query_hist_full_scan=0.2,
    )


def timeit(fn, obs_list, warmup=200):
    for o in obs_list[:warmup]:
        fn(o)
    times = np.empty(len(obs_list))
    for i, o in enumerate(obs_list):
        t0 = time.perf_counter()
        fn(o)
        times[i] = (time.perf_counter() - t0) * 1000.0   # ms
    return {
        'median_ms': round(float(np.median(times)), 4),
        'mean_ms': round(float(np.mean(times)), 4),
        'p95_ms': round(float(np.percentile(times, 95)), 4),
        'p99_ms': round(float(np.percentile(times, 99)), 4),
        'n': len(times),
    }


def hardware() -> dict:
    info = {'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'device': 'CPU'}
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                      text=True).strip()
        if out:
            info['gpu_present'] = out.splitlines()[0]
    except Exception:
        info['gpu_present'] = 'none'
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    info['cpu_model'] = line.split(':', 1)[1].strip()
                    break
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='/app/revision/phase2_compute')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    obs_list = [make_obs(rng) for _ in range(args.iterations)]
    results = {'hardware': hardware(), 'iterations': args.iterations, 'variants': {}}

    # Meta-controller alone (architecture-independent)
    meta = MetaController()
    results['meta_controller'] = timeit(lambda o: meta.decide(o), obs_list)
    print(f"MetaController.decide      median={results['meta_controller']['median_ms']:.4f} ms  "
          f"p95={results['meta_controller']['p95_ms']:.4f} ms", flush=True)

    for name, model_type, label in VARIANTS:
        pol = MultiAgentPolicy(
            compact_weights=f'{W}/compaction_{label}_seed1.weights.h5',
            partition_weights=f'{W}/partition_{label}_seed1.weights.h5',
            stochastic=False, model_type=model_type, name=name,
        )
        ca, pa = pol.compaction_agent, pol.partition_agent
        r = {
            'compaction_forward': timeit(lambda o: ca.get_action(o), obs_list),
            'partition_forward': timeit(lambda o: pa.get_action(o), obs_list),
            'full_hierarchical_step': timeit(lambda o: pol.get_action(o), obs_list),
            'params_compaction': int(sum(np.prod(v.shape) for v in ca._model.trainable_variables)),
            'params_partition': int(sum(np.prod(v.shape) for v in pa._model.trainable_variables)),
        }
        r['params_total'] = r['params_compaction'] + r['params_partition']
        results['variants'][name] = r
        print(f"{name:20s} full-step median={r['full_hierarchical_step']['median_ms']:.4f} ms  "
              f"p95={r['full_hierarchical_step']['p95_ms']:.4f} ms  "
              f"params={r['params_total']:,}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'inference_latency.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # flat CSV for the manuscript table
    import csv
    rows = []
    for name, r in results['variants'].items():
        rows.append({
            'variant': name,
            'meta_controller_median_ms': results['meta_controller']['median_ms'],
            'meta_controller_p95_ms': results['meta_controller']['p95_ms'],
            'compaction_median_ms': r['compaction_forward']['median_ms'],
            'compaction_p95_ms': r['compaction_forward']['p95_ms'],
            'partition_median_ms': r['partition_forward']['median_ms'],
            'partition_p95_ms': r['partition_forward']['p95_ms'],
            'full_step_median_ms': r['full_hierarchical_step']['median_ms'],
            'full_step_p95_ms': r['full_hierarchical_step']['p95_ms'],
            'params_compaction': r['params_compaction'],
            'params_partition': r['params_partition'],
            'params_total': r['params_total'],
            'device': results['hardware']['device'],
            'iterations': args.iterations,
        })
    with open(os.path.join(args.out, 'inference_latency.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'\nwrote inference_latency.{{json,csv}} → {args.out}')


if __name__ == '__main__':
    main()
