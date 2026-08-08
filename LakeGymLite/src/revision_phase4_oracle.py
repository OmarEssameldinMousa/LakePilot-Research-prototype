"""
Phase 4 — Oracle meta-controller upper bound (environment rollback).

Reviewer demand: quantify the performance loss attributable to the *rule-based*
meta-controller, i.e. separate "when to act" from "what to do".

METHOD: environment snapshot/rollback (option (a) of the plan).

At each step the environment is advanced once by ingestion, then a snapshot is
taken. Each of the three delegations is executed as a trial branch — NOOP, the
compaction specialist's chosen action, the partition specialist's chosen action —
and its **realised** reward is measured by actually running the query. Between
branches the environment is restored: the Iceberg data snapshot is rolled back,
any partition-spec change is undone, and all Python-side state (simulator state,
pruning/query histories, cumulative reward, and both agents' observation windows)
is restored. The branch with the highest realised global reward is then executed
for real, and that execution is what enters the recorded trajectory.

Why rollback rather than a one-step lookahead on the reward model: the reward
depends on measured query latency, which no model in this codebase predicts. A
lookahead oracle would therefore bound the performance of the *reward model*, not
of the environment. Rollback measures realised reward, giving a genuine upper
bound, at the cost of ~4x the environment work per step.

KNOWN LIMITATION (documented, not hidden): query latency is re-measured on the
final execution, so the reward recorded for the chosen branch differs slightly
from the reward that motivated the choice. The oracle is therefore "select the
delegation that realised the highest reward in a trial rollout", not a
perfectly-informed oracle. Environment stochasticity means it is a tight but
not exact upper bound.

Usage:
    docker exec -w /app/src lakegym-lite python3 revision_phase4_oracle.py \
        --model-type ddqn --weights-label ddqn --seeds 1 --protocols std1000 eval500
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, deque
from datetime import datetime

import numpy as np

import policies  # noqa: F401
from simulation import LakeSimulator, RewardCalculator, PartitionStrategy
from policies.base import Action, Observation
from agents.compaction_agent import CompactionAgent
from agents.partition_agent import PartitionAgent
from agents.meta_controller import MetaController, Delegation
from metrics import OfflineMetricsCollector

PROTOCOLS = {'std1000': ('workload_plan_v5.json', 1, 1000),
             'eval500': ('workload_plan_v5_eval.json', 5, 500)}
BRANCHES = [Delegation.NOOP, Delegation.COMPACTION, Delegation.PARTITION]


def env_seed_for(protocol: str, slot: int, episode: int) -> int:
    # Distinct from Phase 1 and the online runs.
    return 35000 + 100 * slot + episode


def git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=os.path.dirname(__file__), text=True).strip()
    except Exception:
        return 'unknown'


# ── environment snapshot / restore ───────────────────────────────────────


def harden_metadata(sim: LakeSimulator) -> None:
    """Stop Iceberg garbage-collecting metadata under the oracle's commit churn.

    The oracle issues ~3-4 commits per step (trial compaction, rollback, final
    execution), so a 1,000-step episode produces several thousand commits. With
    default retention, manifests behind older snapshots are eventually collected
    and a subsequent rollback fails with
        NotFoundException: Location does not exist: .../metadata/*.avro
    after which every metrics query silently returns zeros. Observed at ~step
    50-100 of the first attempt; 160 steps run clean with retention disabled.

    Must be re-applied after every sim.reset(), which drops and recreates the table.
    """
    try:
        sim._spark.sql(f"""ALTER TABLE {sim.FULL_TABLE_NAME} SET TBLPROPERTIES (
            'write.metadata.delete-after-commit.enabled'='false',
            'write.metadata.previous-versions-max'='2000')""")
    except Exception as e:
        print(f'   ! could not harden metadata retention: {e}', flush=True)


def take_snapshot(sim: LakeSimulator, agents) -> dict:
    snap_id = None
    try:
        rows = sim._spark.sql(
            f'SELECT snapshot_id FROM {sim.FULL_TABLE_NAME}.snapshots '
            f'ORDER BY committed_at DESC LIMIT 1').collect()
        if rows:
            snap_id = rows[0]['snapshot_id']
    except Exception as e:
        print(f'   ! snapshot read failed: {e}', flush=True)
    return {
        'snapshot_id': snap_id,
        'state': copy.deepcopy(sim._state),
        'pruning_history': list(sim._pruning_history),
        'query_history': list(sim._query_history),
        'prev_avg_pruning': sim._prev_avg_pruning,
        'cumulative_global': sim._cumulative_global,
        'windows': [list(a._window) for a in agents],
    }


def restore(sim: LakeSimulator, agents, snap: dict, branch_action: Action) -> None:
    """Undo a trial branch: partition spec, data snapshot, then Python state."""
    # 1. Undo partition-spec evolution (snapshot rollback does NOT revert this).
    before = snap['state'].partition_strategy
    after = sim._state.partition_strategy
    if after != before:
        try:
            if after != PartitionStrategy.UNPARTITIONED:
                sim._drop_current_partition(after)
            if before != PartitionStrategy.UNPARTITIONED:
                if before == PartitionStrategy.HOUR:
                    sim._spark.sql(f'ALTER TABLE {sim.FULL_TABLE_NAME} ADD PARTITION FIELD hours(event_ts)')
                elif before == PartitionStrategy.REGION:
                    sim._spark.sql(f'ALTER TABLE {sim.FULL_TABLE_NAME} ADD PARTITION FIELD identity(region)')
                elif before == PartitionStrategy.EVENT_TYPE:
                    sim._spark.sql(f'ALTER TABLE {sim.FULL_TABLE_NAME} ADD PARTITION FIELD identity(event_type)')
        except Exception as e:
            print(f'   ! partition-spec restore failed: {e}', flush=True)

    # 2. Roll the data back (compaction rewrites files into a new snapshot).
    if snap['snapshot_id'] is not None and branch_action in (
            Action.COMPACT_32KB, Action.COMPACT_64KB, Action.COMPACT_128KB):
        try:
            sim._spark.sql(
                f"CALL {sim.CATALOG}.system.rollback_to_snapshot("
                f"table => '{sim.DATABASE}.{sim.TABLE}', snapshot_id => {snap['snapshot_id']})")
        except Exception as e:
            print(f'   ! snapshot rollback failed: {e}', flush=True)
    sim._spark.catalog.clearCache()

    # 3. Python-side state.
    sim._state = copy.deepcopy(snap['state'])
    sim._pruning_history = deque(snap['pruning_history'], maxlen=sim.QUERY_HISTORY_SIZE)
    sim._query_history = deque(snap['query_history'], maxlen=sim.QUERY_HISTORY_SIZE)
    sim._prev_avg_pruning = snap['prev_avg_pruning']
    sim._cumulative_global = snap['cumulative_global']
    for a, w in zip(agents, snap['windows']):
        a._window = deque(w, maxlen=a.window_size)


def execute_branch(sim: LakeSimulator, action: Action) -> dict:
    """Action -> metrics -> query -> reward, mirroring LakeSimulator.run_step
    steps 2-5 (ingestion is done once, before branching)."""
    action_cost, _, ok = sim.execute_action(action)
    fc, size_kb, _ = sim.collect_metrics()
    adv = sim.collect_advanced_metrics()
    latency_ms, qtype, _ = sim.measure_performance()
    pruning = sim._compute_pruning_ratio(qtype)
    sim._pruning_history.append(pruning)
    avg_now = float(np.mean(sim._pruning_history)) if sim._pruning_history else 0.0
    r_global = RewardCalculator.global_reward(latency_ms, fc, pruning, adv['block_utilization'])
    return {'action': action, 'action_success': ok, 'action_cost': action_cost,
            'file_count': fc, 'total_size_kb': size_kb, 'latency_ms': latency_ms,
            'pruning': pruning, 'avg_pruning_now': avg_now, 'qtype': qtype,
            'block_utilization': adv['block_utilization'],
            'file_size_skew_kb': adv['file_size_skew_kb'],
            'partition_skew': adv['partition_skew'], 'global_reward': r_global}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-type', default='ddqn', choices=['attentive_ppo', 'mlp_ppo', 'ddqn'])
    ap.add_argument('--weights-label', default='ddqn')
    ap.add_argument('--weights-dir', default='/app/revision/phase1_stats/phase1_weights/weights')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1])
    ap.add_argument('--protocols', nargs='+', default=['std1000', 'eval500'])
    ap.add_argument('--out', default='/app/revision/phase4_oracle/raw')
    args = ap.parse_args()

    label = f'Oracle_{args.weights_label}'
    plan_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    print(f'🔮 Oracle meta-controller — {args.model_type} ({args.weights_label})', flush=True)
    sim = LakeSimulator()
    msg = sim.initialize()
    print(f'   {msg}', flush=True)
    if '❌' in msg:
        sys.exit(1)
    harden_metadata(sim)

    for slot in args.seeds:
        for proto in args.protocols:
            plan_file, n_eps, steps = PROTOCOLS[proto]
            out_dir = os.path.join(args.out, label, f'seed{slot}', proto)
            man_path = os.path.join(out_dir, 'manifest.json')
            if os.path.exists(man_path) and json.load(open(man_path)).get('status') == 'done':
                print(f'⏭️  {label} seed{slot} {proto} done — skipping', flush=True)
                continue
            os.makedirs(out_dir, exist_ok=True)
            plan = json.load(open(os.path.join(plan_dir, plan_file)))

            ca = CompactionAgent(
                weights_path=f'{args.weights_dir}/compaction_{args.weights_label}_seed{slot}.weights.h5',
                model_type=args.model_type)
            pa = PartitionAgent(
                weights_path=f'{args.weights_dir}/partition_{args.weights_label}_seed{slot}.weights.h5',
                model_type=args.model_type)
            agents = [ca, pa]
            meta = MetaController()

            manifest = {'status': 'running', 'label': label, 'method': 'environment_rollback',
                        'model_type': args.model_type, 'weights_label': args.weights_label,
                        'seed_slot': slot, 'protocol': proto, 'plan_file': plan_file,
                        'episodes': n_eps, 'steps_per_episode': steps,
                        'env_seeds': [env_seed_for(proto, slot, e) for e in range(1, n_eps + 1)],
                        'git_commit': git_commit(), 'platform': platform.platform(),
                        'started_at': datetime.now().isoformat()}
            json.dump(manifest, open(man_path, 'w'), indent=2)

            summaries, confusion, t_proto = [], Counter(), time.time()
            for ep in range(1, n_eps + 1):
                es = env_seed_for(proto, slot, ep)
                random.seed(es); np.random.seed(es % (2 ** 32 - 1))
                sim.reset(); harden_metadata(sim); sim.scenario_manager.load_plan(plan)
                ca.reset(); pa.reset(); meta.reset()

                coll = OfflineMetricsCollector(); coll.set_policy(label)
                rows, t0 = [], time.time()
                obs_dict = sim.get_observation()

                for step in range(1, steps + 1):
                    obs = Observation.from_dict(obs_dict)

                    # What the RULE-BASED controller would have chosen (not executed).
                    rule_dec = meta.decide(obs).delegation

                    # Advance the world once, then branch.
                    sim._state.step_count += 1
                    sim.ingest_micro_batch()
                    snap = take_snapshot(sim, agents)

                    # Candidate action per delegation (specialists are deterministic).
                    cand = {Delegation.NOOP: Action.NOOP,
                            Delegation.COMPACTION: ca.get_action(obs)[0],
                            Delegation.PARTITION: pa.get_action(obs)[0]}
                    for a in agents:      # undo the window pushes from get_action
                        a._window = deque(snap['windows'][agents.index(a)], maxlen=a.window_size)

                    trials = {}
                    for br in BRANCHES:
                        trials[br] = execute_branch(sim, cand[br])
                        restore(sim, agents, snap, cand[br])

                    best = max(BRANCHES, key=lambda b: trials[b]['global_reward'])
                    confusion[(rule_dec, best)] += 1

                    # Execute the winner for real.
                    res = execute_branch(sim, cand[best])
                    sim._prev_avg_pruning = res['avg_pruning_now']
                    sim._cumulative_global += res['global_reward']
                    sim._state.last_action = cand[best]
                    if cand[best] in (Action.COMPACT_32KB, Action.COMPACT_64KB, Action.COMPACT_128KB):
                        sim._state.last_compact_step = sim._state.step_count
                    elif cand[best] != Action.NOOP:
                        sim._state.last_partition_step = sim._state.step_count
                    meta.notify_action(cand[best])

                    if res['file_count'] == 0 or res['latency_ms'] <= 0:
                        raise RuntimeError(
                            f'environment health check FAILED at step {step} '
                            f'(files={res["file_count"]}, latency={res["latency_ms"]}): '
                            f'the Iceberg metadata is inconsistent; this episode is not valid data.')

                    next_obs = sim.get_observation()
                    coll.record_step(obs_dict, cand[best], 0.0, 0.0,
                                     res['global_reward'], next_obs, step == steps)
                    rows.append({'step': step, 'rule_delegation': rule_dec,
                                 'oracle_delegation': best, 'action': cand[best].name,
                                 'agree': int(rule_dec == best),
                                 **{f'r_{b}': trials[b]['global_reward'] for b in BRANCHES},
                                 'realised_reward': res['global_reward'],
                                 'latency_ms': res['latency_ms'], 'file_count': res['file_count'],
                                 'pruning': res['pruning']})
                    obs_dict = next_obs

                    if step % 50 == 0:
                        agree = np.mean([r['agree'] for r in rows])
                        print(f'      ep{ep} step {step}/{steps} R={res["global_reward"]:+.3f} '
                              f'files={res["file_count"]} agreement={agree:.1%}', flush=True)

                import pandas as pd
                pd.DataFrame(rows).to_csv(os.path.join(out_dir, f'oracle_trace_ep{ep}.csv'), index=False)
                coll.export_transitions(os.path.join(out_dir, f'transitions_ep{ep}.csv'))
                summaries.append({
                    'episode': ep, 'env_seed': es, 'steps': steps,
                    'wall_clock_sec': round(time.time() - t0, 1),
                    'avg_global_reward': round(float(np.mean([r['realised_reward'] for r in rows])), 6),
                    'avg_latency_ms': round(float(np.mean([r['latency_ms'] for r in rows])), 4),
                    'avg_file_count': round(float(np.mean([r['file_count'] for r in rows])), 4),
                    'avg_partition_pruning_ratio': round(float(np.mean([r['pruning'] for r in rows])), 6),
                    'agreement_rate': round(float(np.mean([r['agree'] for r in rows])), 4),
                })
                print(f'   ✅ ep{ep}: R={summaries[-1]["avg_global_reward"]:+.4f} '
                      f'agreement={summaries[-1]["agreement_rate"]:.1%} '
                      f'({summaries[-1]["wall_clock_sec"]/60:.0f} min)', flush=True)
                rows = []

            import csv
            with open(os.path.join(out_dir, 'episode_summary.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
                w.writeheader(); w.writerows(summaries)
            json.dump({f'{k[0]}->{k[1]}': v for k, v in confusion.items()},
                      open(os.path.join(out_dir, 'confusion.json'), 'w'), indent=2)
            manifest.update(status='done', finished_at=datetime.now().isoformat(),
                            total_wall_clock_sec=round(time.time() - t_proto, 1))
            json.dump(manifest, open(man_path, 'w'), indent=2)
            print(f'✅ {label} seed{slot} {proto} done', flush=True)

    sim.scenario_manager.clear()
    print('\n🏁 oracle runs complete.', flush=True)


if __name__ == '__main__':
    main()
