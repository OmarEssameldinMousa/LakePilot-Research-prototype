#!/usr/bin/env bash
# Round-2 experiment batch, in dependency order. Every runner skips seeds whose
# manifest says status="done", so this script is safe to re-run after an
# interruption.
#
# Operational rules from REPRODUCE.md that this script relies on:
#   • nothing else CPU-heavy may run concurrently (query latency is in the reward)
#   • one seed per process for long batches (the runners already fork per seed)
set -u

W=/app/revision/phase1_stats/phase1_weights/weights
EX="docker exec -w /app/src lakegym-lite python3"
LOG=/app/revision/phase10_r2/logs
mkdir -p "$(dirname "$0")/phase10_r2/logs"

run () {                       # run <name> <command...>
  local name=$1; shift
  echo "════ $(date '+%F %T')  START  $name"
  if "$@" > "$(dirname "$0")/phase10_r2/logs/${name}.log" 2>&1; then
    echo "════ $(date '+%F %T')  DONE   $name"
  else
    echo "════ $(date '+%F %T')  FAILED $name (exit $?) — continuing"
  fi
}

# ── 1. Frozen control under the ADAPTIVE protocol ────────────────────────────
# Same env-seed block (25101-25505), same checkpoints, same episode structure as
# the adaptive runs; weight updates disabled. Arm A keeps the adaptive protocol's
# stochastic action selection, so the only difference from the adaptive runs is
# learning itself. MLP-PPO first: it is the architecture whose adaptive mean
# (0.2093) most exceeds its Phase 1 value (0.1675).
for ARCH in mlp_ppo ddqn attentive_ppoclip; do
  case $ARCH in
    attentive_ppoclip) MT=attentive_ppo ;;
    *)                 MT=$ARCH ;;
  esac
  run "frozenA_${ARCH}" $EX revision_online_eval.py \
      --model-type "$MT" --frozen \
      --compact-weights  "$W/compaction_${ARCH}_seed{seed}.weights.h5" \
      --partition-weights "$W/partition_${ARCH}_seed{seed}.weights.h5" \
      --seeds 1 2 3 4 5 --episodes 5 \
      --label "frozen_${ARCH}" --out /app/revision/phase10_r2/frozen_raw
done

# ── 2. Arm B: argmax + no updates, MLP-PPO only ──────────────────────────────
# Separates "different environment-seed block" from "argmax vs sampling".
run "frozenB_mlp_ppo_greedy" $EX revision_online_eval.py \
    --model-type mlp_ppo --frozen --greedy \
    --compact-weights  "$W/compaction_mlp_ppo_seed{seed}.weights.h5" \
    --partition-weights "$W/partition_mlp_ppo_seed{seed}.weights.h5" \
    --seeds 1 2 3 4 5 --episodes 5 \
    --label frozen_greedy_mlp_ppo --out /app/revision/phase10_r2/frozen_raw

# ── 3. Ablation replicated on DDQN ───────────────────────────────────────────
run "ablation_compactonly_ddqn" $EX revision_phase1_eval.py \
    --policy CompactOnlyRL --model-type ddqn \
    --compact-weights "$W/compaction_ddqn_seed{seed}.weights.h5" \
    --seeds 1 2 3 4 5 --protocols eval500 \
    --label CompactOnlyRL_ddqn --out /app/revision/phase10_r2/ablation_raw

# PartitionOnly is the slow one: ~790 files and ~3.6 s queries make one episode
# cost 36 min against 7 for an ordinary policy. It runs at 1 episode per seed
# (n=3 paired with episode 1 of full-agent seeds 1-3, env seeds 15101/201/301).
# That is enough because the result is forced by the environment rather than by
# the agent: pruning benefit is scaled by a coverage factor that is 1.0 only when
# compacted_after_partition is true (simulation.py:915-938), and declaring a
# partition spec sets that flag false (simulation.py:1107). An agent with no
# compaction action cannot reach full coverage however well it partitions, so the
# run confirms a mechanism rather than estimating a quantity.
run "ablation_partitiononly_ddqn" $EX revision_phase1_eval.py \
    --policy PartitionOnlyRL --model-type ddqn --episodes 1 \
    --partition-weights "$W/partition_ddqn_seed{seed}.weights.h5" \
    --seeds 1 2 3 --protocols eval500 \
    --label PartitionOnlyRL_ddqn --out /app/revision/phase10_r2/ablation_raw

# ── 4. Paired comparator for the oracle ──────────────────────────────────────
# The oracle ran on env_seed 35101; the rule-based comparator in Table 7 came
# from the std1000 block (11001-11005), so the published comparison was unpaired.
run "oracle_paired_comparator" $EX revision_phase1_eval.py \
    --policy MultiAgentRL --model-type ddqn \
    --compact-weights  "$W/compaction_ddqn_seed{seed}.weights.h5" \
    --partition-weights "$W/partition_ddqn_seed{seed}.weights.h5" \
    --seeds 1 --protocols std1000 --env-seed 35101 \
    --label MultiAgentRL_ddqn_oracleseed --out /app/revision/phase10_r2/oracle_raw

echo "════ $(date '+%F %T')  BATCH COMPLETE"
