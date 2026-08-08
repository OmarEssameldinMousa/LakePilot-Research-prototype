"""
Phase 3 — sensitivity sweeps (evaluation-only, frozen checkpoints).

Two sweeps that require running the environment:

  meta   One-at-a-time variation of the meta-controller constants around their
         published defaults (theta=0.35, util boost=0.15, compaction cooldown=4,
         partition cooldown=20). Reviewer objection: these constants were never
         justified.

  prune  Multiply every non-zero pruning-matrix value by a constant. Reviewer
         objection: the pruning matrix is a hardcoded assumption. Question to
         answer: does the RL-vs-heuristic gap survive a weaker pruning benefit?

Both are driven through environment variables read by `agents.meta_controller`
and `simulation`, whose defaults reproduce the published constants exactly — so
a sweep at the default point is a consistency check against Phase 1.

Each configuration is executed as its own subprocess so the environment variables
take effect at import time and each run gets a fresh SparkSession.

Usage:
    docker exec -w /app/src lakegym-lite python3 revision_phase3_sweep.py \
        --sweep meta --episodes 3
    docker exec -w /app/src lakegym-lite python3 revision_phase3_sweep.py \
        --sweep prune --episodes 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

W = '/app/revision/phase1_stats/phase1_weights/weights'
OUT_ROOT = '/app/revision/phase3_sensitivity'

# Defaults (must match the published constants)
DEFAULTS = {'theta': 0.35, 'boost': 0.15, 'cc': 4, 'pc': 20}
GRID = {
    'theta': [0.25, 0.30, 0.35, 0.40, 0.45],
    'boost': [0.0, 0.15, 0.30],
    'cc': [2, 4, 8],
    'pc': [10, 20, 40],
}
ENV_OF = {'theta': 'LAKEGYM_META_THETA', 'boost': 'LAKEGYM_META_UTIL_BOOST',
          'cc': 'LAKEGYM_META_COMPACT_COOLDOWN', 'pc': 'LAKEGYM_META_PARTITION_COOLDOWN'}

PRUNE_SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]

# The agent used for sensitivity analysis: the best-performing architecture
# from Phase 1. Heuristic comparator for the pruning sweep.
AGENT = ('MultiAgentRL', 'ddqn', 'ddqn')       # (policy, model_type, weight label)
HEURISTIC = 'WorkloadAwareThreshold'


def meta_configs():
    """One-at-a-time around the defaults; the default point appears once."""
    seen, out = set(), []
    key = tuple(DEFAULTS[k] for k in ('theta', 'boost', 'cc', 'pc'))
    seen.add(key)
    out.append(dict(DEFAULTS))
    for param, values in GRID.items():
        for v in values:
            cfg = dict(DEFAULTS)
            cfg[param] = v
            k = tuple(cfg[x] for x in ('theta', 'boost', 'cc', 'pc'))
            if k not in seen:
                seen.add(k)
                out.append(cfg)
    return out


def run_one(label: str, out_dir: str, env_extra: dict, episodes: int,
            policy: str, model_type: str, weight_label: str | None) -> bool:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    # 3 seeds x 1 episode = 3 evaluation episodes per configuration (plan spec).
    # Env seeds are a function of (protocol, seed, episode) only, so every
    # configuration is evaluated on the SAME three episodes -> paired comparison.
    cmd = [sys.executable, 'revision_phase1_eval.py',
           '--policy', policy, '--seeds', '1', '2', '3', '--episodes', '1',
           '--protocols', 'eval500', '--label', label, '--out', out_dir]
    if weight_label:
        cmd += ['--model-type', model_type,
                '--compact-weights', f'{W}/compaction_{weight_label}_seed{{seed}}.weights.h5',
                '--partition-weights', f'{W}/partition_{weight_label}_seed{{seed}}.weights.h5']
    print(f'\n═══ {label}  env={env_extra} ═══', flush=True)
    r = subprocess.run(cmd, env=env)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', required=True, choices=['meta', 'prune'])
    ap.add_argument('--episodes', type=int, default=1,
                    help='episodes per seed slot (3 seeds are used, so 1 -> 3 episodes/config)')
    args = ap.parse_args()

    if args.sweep == 'meta':
        out_dir = f'{OUT_ROOT}/meta_grid'
        cfgs = meta_configs()
        print(f'meta-controller sweep: {len(cfgs)} configurations', flush=True)
        manifest = []
        for cfg in cfgs:
            label = (f"meta_theta{cfg['theta']}_boost{cfg['boost']}"
                     f"_cc{cfg['cc']}_pc{cfg['pc']}").replace('.', 'p')
            env_extra = {ENV_OF[k]: cfg[k] for k in cfg}
            ok = run_one(label, out_dir, env_extra, args.episodes,
                         AGENT[0], AGENT[1], AGENT[2])
            manifest.append({**cfg, 'label': label, 'ok': ok,
                             'is_default': cfg == DEFAULTS})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        json.dump(manifest, open(f'{out_dir}/sweep_manifest.json', 'w'), indent=2)

    else:
        out_dir = f'{OUT_ROOT}/pruning'
        print(f'pruning sweep: {len(PRUNE_SCALES)} scales × 2 policies', flush=True)
        manifest = []
        for scale in PRUNE_SCALES:
            env_extra = {'LAKEGYM_PRUNING_SCALE': scale}
            tag = str(scale).replace('.', 'p')
            for policy, mt, wl in [(AGENT[0], AGENT[1], AGENT[2]), (HEURISTIC, None, None)]:
                name = 'ddqn' if wl else 'wat'
                label = f'prune{tag}_{name}'
                ok = run_one(label, out_dir, env_extra, args.episodes, policy, mt, wl)
                manifest.append({'scale': scale, 'policy': policy, 'label': label, 'ok': ok})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        json.dump(manifest, open(f'{out_dir}/sweep_manifest.json', 'w'), indent=2)

    print('\n🏁 sweep complete', flush=True)


if __name__ == '__main__':
    main()
