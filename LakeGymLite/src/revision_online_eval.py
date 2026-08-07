"""
Revision — headless online (adaptive) learning runner.

Replaces the NiceGUI-driven adaptive path in `app.py` with a seeded, resumable
CLI so the online experiments can be run in batch and reproduced.

Two protocols are supported through the same code path:

  ADAPTIVE   (--compact-weights/--partition-weights given)
      Start from the offline checkpoints and keep learning online. This is the
      published setting (paper Figs 9-11, Table 7) and must be regenerated
      because (a) the original runs were initialised from artifact checkpoints,
      (b) the online DDQN update regressed softmax outputs instead of Q-values,
      and (c) epsilon never decayed.

  SCRATCH    (no weights given)
      Random initialisation, learning purely online. A negative control that
      quantifies how much of the system's performance comes from offline
      pretraining.

Update cadence:
  --update-every 0   update once per episode (the published behaviour)
  --update-every N   update every N specialist transitions (recommended for
                     SCRATCH: once-per-episode gives PPO only ~5 gradient
                     updates in total, which would be a strawman).

Outputs under --out:
    <label>/seed<k>/episode_log.csv      per-episode reward/latency/files/pruning/loss
    <label>/seed<k>/transitions_ep<e>.csv
    <label>/seed<k>/weights/{compact,partition}_final.weights.h5
    <label>/seed<k>/manifest.json        seed, config, git, hardware, wall-clock

A seed whose manifest says status="done" is skipped, so batches are resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from simulation import LakeSimulator
from policies.base import Action, Observation
from metrics import OfflineMetricsCollector
from training.online_training import (
    MultiAgentOnlineTrainer, PPOConfig, DQNConfig, RolloutBuffer, set_global_seeds,
)


def env_seed_for(slot: int, episode: int) -> int:
    """Environment seed: distinct from the Phase 1 evaluation seed space."""
    return 25000 + 100 * slot + episode


def git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:
        return 'unknown'


def hardware_info() -> dict:
    info = {'platform': platform.platform(), 'python': sys.version.split()[0],
            'cpu_count': os.cpu_count()}
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    info['mem_total_kb'] = int(line.split()[1])
                    break
    except Exception:
        pass
    return info


def run_seed(args, slot: int, plan: dict, steps: int) -> None:
    out_dir = os.path.join(args.out, args.label, f'seed{slot}')
    manifest_path = os.path.join(out_dir, 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            if json.load(f).get('status') == 'done':
                print(f'⏭️  {args.label} seed{slot} already done — skipping', flush=True)
                return
    os.makedirs(os.path.join(out_dir, 'weights'), exist_ok=True)

    mode = 'scratch' if not (args.compact_weights or args.partition_weights) else 'adaptive'

    def fill(p):
        return p.replace('{seed}', str(slot)) if p else None

    manifest = {
        'status': 'running', 'label': args.label, 'mode': mode,
        'model_type': args.model_type, 'seed_slot': slot,
        'compact_weights': fill(args.compact_weights),
        'partition_weights': fill(args.partition_weights),
        'episodes': args.episodes, 'steps_per_episode': steps,
        'plan_file': args.plan, 'update_every': args.update_every,
        'lr': args.lr,
        'env_seeds': [env_seed_for(slot, e) for e in range(1, args.episodes + 1)],
        'git_commit': git_commit(), 'hardware': hardware_info(),
        'started_at': datetime.now().isoformat(),
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    set_global_seeds(slot)
    eps_start = (args.epsilon_start if args.epsilon_start is not None
                 else (1.0 if mode == 'scratch' else 0.10))
    ppo_cfg = PPOConfig(actor_lr=args.lr, seed=slot)
    dqn_cfg = DQNConfig(learning_rate=args.lr, seed=slot,
                        epsilon_start=eps_start, epsilon_end=args.epsilon_end,
                        epsilon_decay_steps=args.epsilon_decay_steps)
    manifest['epsilon'] = {'start': eps_start, 'end': args.epsilon_end,
                           'decay_steps': args.epsilon_decay_steps}
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    trainer = MultiAgentOnlineTrainer(
        model_type=args.model_type, config=ppo_cfg, dqn_config=dqn_cfg,
        compact_weights=fill(args.compact_weights),
        partition_weights=fill(args.partition_weights),
        seed=slot,
    )

    sim = LakeSimulator()
    msg = sim.initialize()
    print(f'   {msg}', flush=True)
    if '❌' in msg:
        sys.exit(1)

    rows = []
    t_seed = time.time()
    for ep in range(1, args.episodes + 1):
        es = env_seed_for(slot, ep)
        random.seed(es)
        np.random.seed(es % (2 ** 32 - 1))

        sim.reset()
        sim.scenario_manager.load_plan(plan)
        trainer.reset_episode()

        collector = OfflineMetricsCollector()
        collector.set_policy(f'{args.label}_seed{slot}')
        cbuf, pbuf = RolloutBuffer(), RolloutBuffer()
        rewards, lats, files, prunes, actions = [], [], [], [], []
        losses = {'compaction': 0.0, 'partition': 0.0}
        n_updates = 0

        obs_dict = sim.get_observation()
        t0 = time.time()
        for step in range(1, steps + 1):
            info = trainer.step(sim, obs_dict, greedy=False)   # explore while learning
            result, next_obs = info['result'], info['next_obs_dict']
            done = (step == steps)

            if info['specialist'] == 'compaction' and info['window'] is not None:
                cbuf.add(info['window'], info['action_idx'], info['reward'],
                         info['value'], info['log_prob'], done)
            elif info['specialist'] == 'partition' and info['window'] is not None:
                pbuf.add(info['window'], info['action_idx'], info['reward'],
                         info['value'], info['log_prob'], done)

            collector.record_from_result(obs_dict, Action[result.action_taken],
                                         result, next_obs, done)
            rewards.append(info['global_reward'])
            lats.append(float(result.latency_ms))
            files.append(int(result.file_count))
            prunes.append(float(result.partition_pruning_ratio))
            actions.append(result.action_taken)
            obs_dict = next_obs

            # mid-episode updates
            if args.update_every > 0 and (len(cbuf) + len(pbuf)) >= args.update_every:
                m = trainer.update_models(cbuf, pbuf)
                losses['compaction'] += m['compaction'].get('loss', 0.0)
                losses['partition'] += m['partition'].get('loss', 0.0)
                n_updates += 1
                cbuf, pbuf = RolloutBuffer(), RolloutBuffer()

            if step % 100 == 0:
                print(f'      ep{ep} step {step}/{steps}  lat={result.latency_ms:.0f}ms '
                      f'files={result.file_count} R={result.global_reward:+.3f}', flush=True)

        # end-of-episode update (published cadence, and flushes any remainder)
        if len(cbuf) or len(pbuf):
            m = trainer.update_models(cbuf, pbuf)
            losses['compaction'] += m['compaction'].get('loss', 0.0)
            losses['partition'] += m['partition'].get('loss', 0.0)
            n_updates += 1

        collector.export_transitions(os.path.join(out_dir, f'transitions_ep{ep}.csv'))
        from collections import Counter
        rows.append({
            'episode': ep, 'env_seed': es, 'steps': steps,
            'wall_clock_sec': round(time.time() - t0, 1),
            'avg_global_reward': round(float(np.mean(rewards)), 6),
            'total_global_reward': round(float(np.sum(rewards)), 6),
            'avg_latency_ms': round(float(np.mean(lats)), 4),
            'avg_file_count': round(float(np.mean(files)), 4),
            'avg_partition_pruning_ratio': round(float(np.mean(prunes)), 6),
            'n_updates': n_updates,
            'compaction_loss': round(losses['compaction'] / max(n_updates, 1), 6),
            'partition_loss': round(losses['partition'] / max(n_updates, 1), 6),
            'epsilon': round(getattr(trainer.compact_trainer, '_epsilon', 0.0), 4),
            'action_distribution': json.dumps(dict(Counter(actions))),
        })
        print(f'   ✅ ep{ep}: R={rows[-1]["avg_global_reward"]:+.4f} '
              f'lat={rows[-1]["avg_latency_ms"]:.0f}ms files={rows[-1]["avg_file_count"]:.0f} '
              f'updates={n_updates} ({rows[-1]["wall_clock_sec"]:.0f}s)', flush=True)

        import csv as _csv
        with open(os.path.join(out_dir, 'episode_log.csv'), 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    trainer.save_weights(os.path.join(out_dir, 'weights', 'compact_final.weights.h5'),
                         os.path.join(out_dir, 'weights', 'partition_final.weights.h5'))
    manifest['status'] = 'done'
    manifest['finished_at'] = datetime.now().isoformat()
    manifest['total_wall_clock_sec'] = round(time.time() - t_seed, 1)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    sim.scenario_manager.clear()
    print(f'✅ {args.label} seed{slot} done in {manifest["total_wall_clock_sec"]/60:.0f} min', flush=True)


def main():
    ap = argparse.ArgumentParser(description='Headless online/adaptive learning runner')
    ap.add_argument('--model-type', required=True, choices=['attentive_ppo', 'mlp_ppo', 'ddqn'])
    ap.add_argument('--compact-weights', default=None, help="omit for from-scratch; '{seed}' supported")
    ap.add_argument('--partition-weights', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--plan', default='workload_plan_v5_eval.json')
    ap.add_argument('--update-every', type=int, default=0,
                    help='0 = once per episode (published); N = every N specialist transitions')
    ap.add_argument('--lr', type=float, default=3e-4)
    # Epsilon schedule (DDQN only). The published defaults (1.0 -> 0.05 over 5,000
    # steps) are appropriate for learning from scratch, but for FINE-TUNING a
    # pretrained policy they are actively harmful: decay is counted in specialist
    # actions (~100/episode), so a 5-episode run never leaves epsilon ~0.9 and the
    # agent acts almost uniformly at random throughout, destroying the behaviour it
    # was supposed to adapt. Measured on adaptive_ddqn_publishedcfg seed1:
    # epsilon 0.979 -> 0.896 across all five episodes.
    ap.add_argument('--epsilon-start', type=float, default=None,
                    help='DDQN exploration start (default 0.10 for adaptive, 1.0 for scratch)')
    ap.add_argument('--epsilon-end', type=float, default=0.02)
    ap.add_argument('--epsilon-decay-steps', type=int, default=200,
                    help='decay horizon in specialist transitions')
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', default='/app/revision/phase6_ddqn/online_raw')
    args = ap.parse_args()

    plan_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    with open(os.path.join(plan_dir, args.plan)) as f:
        plan = json.load(f)
    steps = plan.get('total_steps', 500)

    print(f'🧠 {args.label}: {args.model_type} × {len(args.seeds)} seeds × '
          f'{args.episodes} ep × {steps} steps [{args.plan}] '
          f'update_every={args.update_every}', flush=True)

    for slot in args.seeds:
        print(f'\n═══ {args.label} seed{slot} ═══', flush=True)
        run_seed(args, slot, plan, steps)

    print('\n🏁 online runs complete.', flush=True)


if __name__ == '__main__':
    main()
