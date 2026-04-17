"""
LakeGym v5 — CLI Experiment Runner

Run experiments from the command line:
    python run_experiment.py --policies No_Maintenance Periodic5 Threshold10_C64 --episodes 3 --steps 500
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import List

from simulation import LakeSimulator
from experiment import ExperimentManager
from policies import ALL_POLICIES
from policies.base import BasePolicy


def get_policy(name: str) -> BasePolicy:
    """Look up a policy by name from the registry."""
    if name not in ALL_POLICIES:
        print(f"❌ Unknown policy: '{name}'")
        print(f"   Available: {', '.join(sorted(ALL_POLICIES.keys()))}")
        sys.exit(1)
    factory = ALL_POLICIES[name]
    return factory() if callable(factory) else factory


def main():
    parser = argparse.ArgumentParser(
        description="LakeGym v5 — Run policy comparison experiments",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--policies", nargs="+", required=True,
        help="Policy names to evaluate.\n"
             f"Available: {', '.join(sorted(ALL_POLICIES.keys()))}",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per policy")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument(
        "--plan", default=None,
        help="Path to workload plan JSON file",
    )
    parser.add_argument("--list", action="store_true", help="List available policies and exit")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable policies:")
        for name in sorted(ALL_POLICIES.keys()):
            print(f"  • {name}")
        return

    # Load workload plan
    plan = None
    if args.plan:
        with open(args.plan) as f:
            plan = json.load(f)
        print(f"📋 Loaded workload plan: {plan.get('name', 'unnamed')}")

    # Build policies
    policies: List[BasePolicy] = [get_policy(name) for name in args.policies]

    # Initialize simulator
    print("🚀 Initializing simulator...")
    sim = LakeSimulator()
    msg = sim.initialize()
    print(f"   {msg}")

    if "❌" in msg:
        sys.exit(1)

    # Run experiments
    em = ExperimentManager(sim)
    results = em.run_batch(
        policies=policies,
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        base_dir=args.output,
        workload_plan=plan,
    )

    # Summary
    print(f"\n{'='*60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for r in results:
        eps = r.get('episodes', [])
        avg_reward = sum(e.get('avg_global_reward', 0) for e in eps) / max(len(eps), 1)
        avg_latency = sum(e.get('avg_latency', 0) for e in eps) / max(len(eps), 1)
        print(f"  {r['policy']:30s} — avg_reward: {avg_reward:+.4f}, avg_latency: {avg_latency:.0f}ms")

    print(f"\n📁 Results saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
