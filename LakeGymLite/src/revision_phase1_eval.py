"""
Revision Phase 1 — Seeded headless evaluation runner.

Runs the multi-seed evaluation protocol demanded by the Scientific Reports
reviewers, with full reproducibility metadata. Designed to run INSIDE the
lakegym-lite container:

    docker exec lakegym-lite python /app/src/revision_phase1_eval.py \
        --policy WorkloadAwareThreshold --seeds 1 2 3 4 5

Protocol per seed slot k:
  • std1000 : 1 × 1000-step episode on workload_plan_v5.json     (env seed 11000+k)
  • eval500 : 5 × 500-step episodes on workload_plan_v5_eval.json (env seeds 15000+100k+e)

Env seeds are a fixed function of (protocol, slot, episode) and NOT of the
policy, so episodes are matched across policies → paired statistics.

RL policies: pass --model-type and --compact-weights/--partition-weights with
an optional "{seed}" placeholder that is filled with the seed slot number, so
each slot evaluates the checkpoint trained with that seed.

Frozen RL evaluation is DETERMINISTIC (argmax): MultiAgentPolicy is
constructed with stochastic=False, which routes through agent.get_action()
(greedy argmax; for DDQN argmax(softmax(Q)) == argmax(Q)).

Outputs under --out (default /app/revision/phase1_stats/raw):
    <policy>/seed<k>/<protocol>/transitions_ep<e>.csv   per-step transitions
    <policy>/seed<k>/<protocol>/episode_summary.csv     per-episode summary
    <policy>/seed<k>/<protocol>/manifest.json           seed/config/git/hardware/time

A run whose manifest has status="done" is skipped → the batch is resumable.
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
from agents.model_registry import MODEL_REGISTRY
from policies import ALL_POLICIES
from policies.base import Observation
from policies.multi_agent_policy import MultiAgentPolicy, CompactOnlyPolicy, PartitionOnlyPolicy
from metrics import OfflineMetricsCollector

PROTOCOLS = {
    # name: (plan_file, episodes_per_seed, steps_per_episode)
    # Plan paths may be absolute; os.path.join then ignores the base directory.
    # The Phase 5 generalization plans live under the mounted revision/ tree
    # because the built-in plans are baked into the container image.
    "std1000": ("workload_plan_v5.json", 1, 1000),
    "eval500": ("workload_plan_v5_eval.json", 5, 500),
    "drift500": ("/app/revision/phase5_generalization/workload_plan_v5_drift.json", 1, 500),
    "mixed500": ("/app/revision/phase5_generalization/workload_plan_v5_mixed.json", 1, 500),
}


def env_seed_for(protocol: str, slot: int, episode: int) -> int:
    """Deterministic env seed shared by every policy for the same slot/episode."""
    if protocol == "std1000":
        return 11000 + slot
    if protocol == "drift500":
        return 45000 + 100 * slot + episode
    if protocol == "mixed500":
        return 55000 + 100 * slot + episode
    return 15000 + 100 * slot + episode


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:
        return "unknown"


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["mem_total_kb"] = int(line.split()[1])
                    break
    except Exception:
        pass
    return info


def build_policy(args, slot: int):
    """Instantiate the policy for a given seed slot."""
    def fill(p):
        return p.replace("{seed}", str(slot)) if p else None

    if args.policy == "MultiAgentRL":
        return MultiAgentPolicy(
            compact_weights=fill(args.compact_weights),
            partition_weights=fill(args.partition_weights),
            stochastic=False,
            name=f"MultiAgentRL_{args.model_type}",
            model_type=args.model_type,
        )
    if args.policy == "CompactOnlyRL":
        return CompactOnlyPolicy(
            compact_weights=fill(args.compact_weights),
            stochastic=False, model_type=args.model_type,
        )
    if args.policy == "PartitionOnlyRL":
        return PartitionOnlyPolicy(
            partition_weights=fill(args.partition_weights),
            stochastic=False, model_type=args.model_type,
        )
    if args.policy in ALL_POLICIES:
        factory = ALL_POLICIES[args.policy]
        return factory() if callable(factory) else factory
    raise SystemExit(f"Unknown policy: {args.policy}. "
                     f"Available: MultiAgentRL, CompactOnlyRL, PartitionOnlyRL, "
                     f"{', '.join(sorted(ALL_POLICIES))}")


def run_episode(sim, policy, steps: int, env_seed: int, policy_label: str):
    """Run one seeded episode; returns (collector, summary_dict)."""
    random.seed(env_seed)
    np.random.seed(env_seed % (2**32 - 1))

    sim.reset()
    policy.reset()

    collector = OfflineMetricsCollector()
    collector.set_policy(policy_label)

    prev_obs = sim.get_observation()
    t0 = time.time()
    for step in range(1, steps + 1):
        obs = Observation.from_dict(prev_obs)
        action = policy.get_action(obs)
        result = sim.run_step(action)
        next_obs = sim.get_observation()
        collector.record_from_result(prev_obs, action, result, next_obs, step == steps)
        prev_obs = next_obs
        if step % 100 == 0:
            print(f"      step {step}/{steps}  lat={result.latency_ms:.0f}ms "
                  f"files={result.file_count} R={result.global_reward:+.3f}", flush=True)
    wall = time.time() - t0

    # ── Environment-health guard ──────────────────────────────────────────
    # A Spark driver crash does not raise here: metrics collection returns 0 and
    # the query returns latency -1, which the reward formula maps to exactly 0.0.
    # Without this check the runner keeps "stepping" against a dead JVM and then
    # writes a status="done" manifest over meaningless data (observed 2026-08-07
    # on PartitionOnly seeds 2-3). Fail loudly instead.
    rows_chk = collector._transitions
    mean_lat = float(np.mean([r["latency_ms"] for r in rows_chk]))
    mean_fc = float(np.mean([r["file_count"] for r in rows_chk]))
    if mean_lat <= 0 or mean_fc == 0 or mean_lat > 60_000:
        raise RuntimeError(
            f"environment health check FAILED (env_seed={env_seed}): "
            f"mean_latency={mean_lat:.1f}ms mean_file_count={mean_fc:.1f} — "
            f"the Spark driver most likely died; this episode is not valid data."
        )

    summary = {
        "env_seed": env_seed,
        "steps": steps,
        "wall_clock_sec": round(wall, 1),
        **{k: v for k, v in collector.get_summary().items() if k != "action_distribution"},
        "action_distribution": json.dumps(collector.get_summary().get("action_distribution", {})),
    }
    # Episode-level metric means from the transitions
    rows = collector._transitions
    for metric in ("latency_ms", "file_count", "partition_pruning_ratio", "block_utilization"):
        summary[f"avg_{metric}"] = round(float(np.mean([r[metric] for r in rows])), 6)
    return collector, summary


def main():
    ap = argparse.ArgumentParser(description="Phase 1 seeded evaluation runner")
    ap.add_argument("--policy", required=True,
                    help="Registry policy name, or MultiAgentRL / CompactOnlyRL / PartitionOnlyRL")
    # Choices come from the registry so a newly registered model cannot be
    # rejected here (the Phase 7 parameter-matched MLP hit exactly that).
    ap.add_argument("--model-type", default="attentive_ppo",
                    choices=sorted(MODEL_REGISTRY.keys()))
    ap.add_argument("--compact-weights", default=None,
                    help="Weights path; '{seed}' is replaced with the seed slot")
    ap.add_argument("--partition-weights", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--protocols", nargs="+", default=["std1000", "eval500"],
                    choices=list(PROTOCOLS.keys()))
    ap.add_argument("--out", default="/app/revision/phase1_stats/raw")
    ap.add_argument("--label", default=None, help="Output folder name (default: policy name)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="Override the protocol's episode count (Phase 3 sweeps use fewer "
                         "episodes per configuration). Env seeds are unchanged, so runs "
                         "remain paired with the full-length protocol.")
    args = ap.parse_args()

    label = args.label or args.policy
    plan_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    print(f"🚀 Initializing simulator for {label} ...", flush=True)
    sim = LakeSimulator()
    msg = sim.initialize()
    print(f"   {msg}", flush=True)
    if "❌" in msg:
        sys.exit(1)

    for slot in args.seeds:
        for proto in args.protocols:
            plan_file, n_eps, steps = PROTOCOLS[proto]
            if args.episodes is not None:
                n_eps = args.episodes
            out_dir = os.path.join(args.out, label, f"seed{slot}", proto)
            manifest_path = os.path.join(out_dir, "manifest.json")

            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    if json.load(f).get("status") == "done":
                        print(f"⏭️  {label} seed{slot} {proto} already done — skipping", flush=True)
                        continue
            os.makedirs(out_dir, exist_ok=True)

            with open(os.path.join(plan_dir, plan_file)) as f:
                plan = json.load(f)

            policy = build_policy(args, slot)
            manifest = {
                "status": "running",
                "policy": label,
                "policy_class": type(policy).__name__,
                "model_type": args.model_type if "RL" in args.policy else None,
                "compact_weights": (args.compact_weights or "").replace("{seed}", str(slot)) or None,
                "partition_weights": (args.partition_weights or "").replace("{seed}", str(slot)) or None,
                "seed_slot": slot,
                "protocol": proto,
                "plan_file": plan_file,
                "episodes": n_eps,
                "steps_per_episode": steps,
                "env_seeds": [env_seed_for(proto, slot, e) for e in range(1, n_eps + 1)],
                "action_selection": "greedy_argmax" if "RL" in args.policy else "deterministic_rule",
                "git_commit": git_commit(),
                "hardware": hardware_info(),
                "started_at": datetime.now().isoformat(),
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            print(f"\n═══ {label} seed{slot} {proto}: {n_eps} × {steps} steps ═══", flush=True)
            sim.scenario_manager.load_plan(plan)

            summaries = []
            t0 = time.time()
            for e in range(1, n_eps + 1):
                es = env_seed_for(proto, slot, e)
                print(f"   ▶ episode {e}/{n_eps} (env_seed={es})", flush=True)
                collector, summary = run_episode(sim, policy, steps, es, label)
                summary["episode"] = e
                summaries.append(summary)
                collector.export_transitions(os.path.join(out_dir, f"transitions_ep{e}.csv"))
                # reload plan (sim.reset in next episode clears nothing, but keep robust)
                sim.scenario_manager.load_plan(plan)

            import csv as _csv
            keys = list(summaries[0].keys())
            with open(os.path.join(out_dir, "episode_summary.csv"), "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(summaries)

            manifest["status"] = "done"
            manifest["finished_at"] = datetime.now().isoformat()
            manifest["total_wall_clock_sec"] = round(time.time() - t0, 1)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"✅ {label} seed{slot} {proto} done in {manifest['total_wall_clock_sec']:.0f}s", flush=True)

    sim.scenario_manager.clear()
    print("\n🏁 All requested runs complete.", flush=True)


if __name__ == "__main__":
    main()
