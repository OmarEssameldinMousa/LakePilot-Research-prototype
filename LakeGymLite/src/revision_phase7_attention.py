"""
Phase 7 Part A — does the attention encoder demonstrably attend to recent steps
and react at workload transitions?

The paper asserts an attention benefit but never shows the mechanism. This runs a
frozen rollout on the 1,000-step standard workload (whose phase boundaries fall at
steps 200/400/600/800) and captures the multi-head self-attention scores at every
step, then reports:

  (a) attention mass by window position — does it weight recent steps, or is it
      flat? A flat profile would mean the encoder adds nothing a global average
      pool could not do.
  (b) attention around the four phase transitions vs mid-phase.
  (c) a statistic with a test: attention mass on the 3 most recent positions,
      transition windows vs mid-phase.

Attention is captured on an independently maintained feature window updated at
EVERY step, rather than on the agent's own window, because the meta-controller
only delegates on some steps and the agent's window would otherwise be stale.

Usage:
    docker exec -w /app/src lakegym-lite python3 revision_phase7_attention.py \
        --weights-label attentive_ppoclip --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque

import policies  # noqa: F401
import numpy as np
import tensorflow as tf

from simulation import LakeSimulator
from policies.base import Observation
from policies.multi_agent_policy import MultiAgentPolicy

TRANSITIONS = [200, 400, 600, 800]   # workload_plan_v5.json phase boundaries
HALFWIN = 25                          # +/- steps counted as a "transition window"


def attention_scores(model, window: np.ndarray) -> np.ndarray:
    """(heads, query, key) attention scores for one window."""
    x = tf.constant(window[None, ...], dtype=tf.float32)
    e = model.embedding(x) + model.pos_encoding
    _, scores = model.attention(e, e, return_attention_scores=True, training=False)
    return scores.numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights-label', default='attentive_ppoclip')
    ap.add_argument('--weights-dir', default='/app/revision/phase1_stats/phase1_weights/weights')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--out', default='/app/revision/phase7_attention/attention_traces')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    plan_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    plan = json.load(open(os.path.join(plan_dir, 'workload_plan_v5.json')))

    sim = LakeSimulator()
    msg = sim.initialize()
    print(f'   {msg}', flush=True)

    for slot in args.seeds:
        out_npz = os.path.join(args.out, f'attention_seed{slot}.npz')
        if os.path.exists(out_npz):
            print(f'⏭️  seed{slot} exists — skipping', flush=True)
            continue

        pol = MultiAgentPolicy(
            compact_weights=f'{args.weights_dir}/compaction_{args.weights_label}_seed{slot}.weights.h5',
            partition_weights=f'{args.weights_dir}/partition_{args.weights_label}_seed{slot}.weights.h5',
            stochastic=False, model_type='attentive_ppo', name='attn_probe')
        ca, pa = pol.compaction_agent, pol.partition_agent

        es = 11000 + slot          # same env seed as the Phase 1 std1000 protocol
        random.seed(es); np.random.seed(es % (2 ** 32 - 1))
        sim.reset(); sim.scenario_manager.load_plan(plan); pol.reset()

        # independent windows, updated every step regardless of delegation
        win_c = deque(maxlen=ca.window_size)
        win_p = deque(maxlen=pa.window_size)

        def padded(dq, size, nfeat):
            arr = np.zeros((size, nfeat), dtype=np.float32)
            if dq:
                seq = np.array(list(dq), dtype=np.float32)
                arr[size - len(seq):] = seq
            return arr

        sc_c, sc_p, steps_rec = [], [], []
        obs_dict = sim.get_observation()
        t0 = time.time()
        for step in range(1, args.steps + 1):
            obs = Observation.from_dict(obs_dict)
            win_c.append(ca.extract_features(obs))
            win_p.append(pa.extract_features(obs))

            sc_c.append(attention_scores(ca._model, padded(win_c, ca.window_size, ca.num_features)))
            sc_p.append(attention_scores(pa._model, padded(win_p, pa.window_size, pa.num_features)))
            steps_rec.append(step)

            action = pol.get_action(obs)
            sim.run_step(action)
            obs_dict = sim.get_observation()

            if step % 100 == 0:
                print(f'      seed{slot} step {step}/{args.steps} ({time.time()-t0:.0f}s)', flush=True)

        np.savez_compressed(out_npz,
                            compaction=np.array(sc_c, dtype=np.float32),
                            partition=np.array(sc_p, dtype=np.float32),
                            steps=np.array(steps_rec),
                            transitions=np.array(TRANSITIONS), halfwin=HALFWIN)
        print(f'✅ seed{slot}: saved {np.array(sc_c).shape} (steps, heads, q, k) '
              f'in {(time.time()-t0)/60:.0f} min', flush=True)

    print('\n🏁 attention capture complete.', flush=True)


if __name__ == '__main__':
    main()
