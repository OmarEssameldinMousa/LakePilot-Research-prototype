"""
LakeGym v5 — Reward Recomputing / Preprocessing Script

Recomputes reward columns in transition CSVs or results CSVs when reward weights,
action costs, or reward logic are changed.

Presets:
  • current         — matches the simulator's current reward design
  • descriptive_v2  — phase-aware reward preset proposed in REPORT.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from policies.base import Action


@dataclass(frozen=True)
class RewardPreset:
    name: str
    latency_max: float
    file_count_max: float
    compact_cost_max: float
    partition_cost_max: float
    skew_max: float
    cw_files: float
    cw_util: float
    cw_cost: float
    cw_target: float
    pw_pruning: float
    pw_dpruning: float
    pw_skew: float
    pw_cost: float
    gw_latency: float
    gw_files: float
    gw_pruning: float
    gw_util: float
    action_costs: Dict[int, float]
    use_target_bonus: bool = False


CURRENT_PRESET = RewardPreset(
    name="current",
    latency_max=15000.0,
    file_count_max=2000.0,
    compact_cost_max=1.1,
    partition_cost_max=2.0,
    skew_max=2.0,
    cw_files=0.25,
    cw_util=0.20,
    cw_cost=0.10,
    cw_target=0.45,
    pw_pruning=0.55,
    pw_dpruning=0.25,
    pw_skew=0.10,
    pw_cost=0.10,
    gw_latency=0.30,
    gw_files=0.20,
    gw_pruning=0.30,
    gw_util=0.20,
    action_costs={
        int(Action.NOOP): 0.0,
        int(Action.COMPACT_32KB): 0.9,
        int(Action.COMPACT_64KB): 1.0,
        int(Action.COMPACT_128KB): 1.1,
        int(Action.PARTITION_HOUR): 2.0,
        int(Action.PARTITION_REGION): 2.0,
        int(Action.PARTITION_EVENT_TYPE): 2.0,
        int(Action.REMOVE_PARTITION): 0.7,
    },
    use_target_bonus=True,
)

DESCRIPTIVE_V2_PRESET = RewardPreset(
    name="descriptive_v2",
    latency_max=15000.0,
    file_count_max=2000.0,
    compact_cost_max=1.1,
    partition_cost_max=2.0,
    skew_max=2.0,
    cw_files=0.25,
    cw_util=0.20,
    cw_cost=0.10,
    cw_target=0.45,
    pw_pruning=0.55,
    pw_dpruning=0.25,
    pw_skew=0.10,
    pw_cost=0.10,
    gw_latency=0.30,
    gw_files=0.20,
    gw_pruning=0.30,
    gw_util=0.20,
    action_costs={
        int(Action.NOOP): 0.0,
        int(Action.COMPACT_32KB): 0.9,
        int(Action.COMPACT_64KB): 1.0,
        int(Action.COMPACT_128KB): 1.1,
        int(Action.PARTITION_HOUR): 2.0,
        int(Action.PARTITION_REGION): 2.0,
        int(Action.PARTITION_EVENT_TYPE): 2.0,
        int(Action.REMOVE_PARTITION): 0.7,
    },
    use_target_bonus=True,
)

LEGACY_V1_PRESET = RewardPreset(
    name="legacy_v1",
    latency_max=15000.0,
    file_count_max=2000.0,
    compact_cost_max=2.0,
    partition_cost_max=3.0,
    skew_max=2.0,
    cw_files=0.40,
    cw_util=0.35,
    cw_cost=0.25,
    cw_target=0.0,
    pw_pruning=0.50,
    pw_dpruning=0.20,
    pw_skew=0.15,
    pw_cost=0.15,
    gw_latency=0.35,
    gw_files=0.25,
    gw_pruning=0.20,
    gw_util=0.20,
    action_costs={
        int(Action.NOOP): 0.0,
        int(Action.COMPACT_32KB): 1.0,
        int(Action.COMPACT_64KB): 1.5,
        int(Action.COMPACT_128KB): 2.0,
        int(Action.PARTITION_HOUR): 3.0,
        int(Action.PARTITION_REGION): 3.0,
        int(Action.PARTITION_EVENT_TYPE): 3.0,
        int(Action.REMOVE_PARTITION): 1.0,
    },
    use_target_bonus=False,
)

PRESETS = {
    CURRENT_PRESET.name: CURRENT_PRESET,
    DESCRIPTIVE_V2_PRESET.name: DESCRIPTIVE_V2_PRESET,
    LEGACY_V1_PRESET.name: LEGACY_V1_PRESET,
}


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def get_after_value(row: pd.Series, base_name: str) -> float:
    next_name = f"next_{base_name}"
    if next_name in row and pd.notna(row[next_name]):
        return float(row[next_name])
    if base_name in row and pd.notna(row[base_name]):
        return float(row[base_name])
    return 0.0


def get_value(row: pd.Series, name: str, default: float = 0.0) -> float:
    if name in row and pd.notna(row[name]):
        return float(row[name])
    return default


def action_id_from_row(row: pd.Series) -> int:
    if "action" in row and pd.notna(row["action"]):
        return int(row["action"])
    if "action_name" in row and pd.notna(row["action_name"]):
        return int(Action[str(row["action_name"])])
    if "action_taken" in row and pd.notna(row["action_taken"]):
        return int(Action[str(row["action_taken"])])
    return int(Action.NOOP)


def compaction_phase_bonus(action_id: int, rows_ingested: float) -> float:
    if action_id not in {
        int(Action.COMPACT_32KB),
        int(Action.COMPACT_64KB),
        int(Action.COMPACT_128KB),
    }:
        return 0.0

    if rows_ingested <= 50:
        table = {
            int(Action.COMPACT_32KB): 1.00,
            int(Action.COMPACT_64KB): 0.25,
            int(Action.COMPACT_128KB): -0.50,
        }
    elif rows_ingested >= 150:
        table = {
            int(Action.COMPACT_32KB): -0.60,
            int(Action.COMPACT_64KB): 0.40,
            int(Action.COMPACT_128KB): 1.00,
        }
    else:
        table = {
            int(Action.COMPACT_32KB): 0.15,
            int(Action.COMPACT_64KB): 1.00,
            int(Action.COMPACT_128KB): 0.15,
        }
    return float(table.get(action_id, 0.0))


def recompute_compact_reward(row: pd.Series, preset: RewardPreset) -> float:
    action_id = action_id_from_row(row)
    cost = preset.action_costs.get(action_id, 0.0)
    file_count = get_after_value(row, "file_count")
    block_util = get_after_value(row, "block_utilization")
    rows_ingested = get_value(row, "rows_ingested", get_after_value(row, "rows_ingested"))

    f_norm = clamp(file_count / preset.file_count_max)
    u_norm = clamp(block_util)
    c_norm = cost / preset.compact_cost_max if preset.compact_cost_max > 0 else 0.0

    reward = -preset.cw_files * f_norm + preset.cw_util * u_norm - preset.cw_cost * c_norm
    if preset.use_target_bonus:
        reward += preset.cw_target * compaction_phase_bonus(action_id, rows_ingested)
    return float(reward)


def recompute_partition_reward(row: pd.Series, preset: RewardPreset) -> float:
    action_id = action_id_from_row(row)
    cost = preset.action_costs.get(action_id, 0.0)

    pruning_ratio = get_after_value(row, "partition_pruning_ratio")
    avg_pruning_prev = get_value(row, "avg_pruning_ratio")
    avg_pruning_now = get_after_value(row, "avg_pruning_ratio")
    partition_skew = get_after_value(row, "partition_skew")

    p_norm = clamp(pruning_ratio)
    dp = max(-1.0, min(1.0, avg_pruning_now - avg_pruning_prev))
    s_norm = clamp(partition_skew / preset.skew_max) if preset.skew_max > 0 else 0.0
    c_norm = cost / preset.partition_cost_max if preset.partition_cost_max > 0 else 0.0

    reward = (
        preset.pw_pruning * p_norm
        + preset.pw_dpruning * dp
        - preset.pw_skew * s_norm
        - preset.pw_cost * c_norm
    )
    return float(reward)


def recompute_global_reward(row: pd.Series, preset: RewardPreset) -> float:
    latency_ms = get_after_value(row, "latency_ms")
    file_count = get_after_value(row, "file_count")
    pruning_ratio = get_after_value(row, "partition_pruning_ratio")
    block_util = get_after_value(row, "block_utilization")

    l_norm = clamp(latency_ms / preset.latency_max) if preset.latency_max > 0 else 0.0
    f_norm = clamp(file_count / preset.file_count_max) if preset.file_count_max > 0 else 0.0
    p_norm = clamp(pruning_ratio)
    u_norm = clamp(block_util)

    reward = (
        -preset.gw_latency * l_norm
        -preset.gw_files * f_norm
        + preset.gw_pruning * p_norm
        + preset.gw_util * u_norm
    )
    return float(reward)


def process_file(path: Path, output_dir: Path, preset: RewardPreset, overwrite: bool = False) -> Path:
    df = pd.read_csv(path)

    for col in ("compact_reward", "partition_reward", "global_reward"):
        if col in df.columns and f"original_{col}" not in df.columns:
            df[f"original_{col}"] = df[col]

    df["compact_reward"] = df.apply(lambda row: recompute_compact_reward(row, preset), axis=1)
    df["partition_reward"] = df.apply(lambda row: recompute_partition_reward(row, preset), axis=1)
    df["global_reward"] = df.apply(lambda row: recompute_global_reward(row, preset), axis=1)
    df["reward_preset"] = preset.name

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / path.name
    if overwrite:
        out_path = path
    df.to_csv(out_path, index=False)
    return out_path


def iter_input_files(inputs: Iterable[str]) -> Iterable[Path]:
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            yield from sorted(p.glob("*.csv"))
        elif any(ch in raw for ch in "*?[]"):
            yield from sorted(Path().glob(raw))
        else:
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute reward columns in CSV files")
    parser.add_argument("inputs", nargs="+", help="CSV files, directories, or glob patterns")
    parser.add_argument("--preset", default="current", choices=sorted(PRESETS.keys()))
    parser.add_argument("--output-dir", default="processed_rewards")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite source files in place")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    output_dir = Path(args.output_dir)

    files = list(iter_input_files(args.inputs))
    if not files:
        raise FileNotFoundError("No input CSV files found")

    print(f"Using preset: {preset.name}")
    for path in files:
        out_path = process_file(path, output_dir=output_dir, preset=preset, overwrite=args.overwrite)
        print(f"✅ {path} -> {out_path}")


if __name__ == "__main__":
    main()
