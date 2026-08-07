# How to resume the revision work

Everything is resumable. A run is skipped when its `manifest.json` has
`status: "done"`; anything else is re-run from scratch, so an interrupted run
never produces partial data in the analysis (per-episode CSVs are only counted
via `episode_summary.csv`, which is written after a protocol completes).

Stopped: 2026-08-02, laptop shutdown.

## 1. Bring the environment back up

```bash
cd /home/omar/Level_4_semester_1/Graduation_project
docker compose up -d spark-master minio mc rest lakegym-lite
docker ps          # wait until minio / rest / spark-master are healthy
```

## 2. Resume the evaluation batch

Remaining work, in priority order. Each command is resumable — completed seeds
are skipped automatically.

```bash
W=/app/revision/phase1_stats/phase1_weights/weights
LOG=$PWD/revision/phase1_stats/rl_batch_final.log

# (a) finish attention+PPO-clip — only seed 5 eval500 is missing (~40 min)
docker exec -w /app/src lakegym-lite python3 revision_phase1_eval.py \
  --policy MultiAgentRL --model-type attentive_ppo \
  --compact-weights  $W/compaction_attentive_ppoclip_seed{seed}.weights.h5 \
  --partition-weights $W/partition_attentive_ppoclip_seed{seed}.weights.h5 \
  --seeds 1 2 3 4 5 --label MultiAgentRL_attentive_ppoclip >> $LOG 2>&1

# (b) DDQN and MLP-PPO — 4 std1000 seeds + all 5 eval500 blocks each (~5.5 h each)
for A in ddqn mlp_ppo; do
docker exec -w /app/src lakegym-lite python3 revision_phase1_eval.py \
  --policy MultiAgentRL --model-type $A \
  --compact-weights  $W/compaction_${A}_seed{seed}.weights.h5 \
  --partition-weights $W/partition_${A}_seed{seed}.weights.h5 \
  --seeds 1 2 3 4 5 --label MultiAgentRL_$A >> $LOG 2>&1
done

# (c) CompactOnly ablation — 5 seeds, both protocols (~5.5 h)
docker exec -w /app/src lakegym-lite python3 revision_phase1_eval.py \
  --policy CompactOnlyRL --model-type attentive_ppo \
  --compact-weights $W/compaction_attentive_ppoclip_seed{seed}.weights.h5 \
  --seeds 1 2 3 4 5 --label CompactOnlyRL_attentive_ppoclip >> $LOG 2>&1

# (d) PartitionOnly ablation — 3 seeds, eval500 only (~5 h; never compacts, so slow)
docker exec -w /app/src lakegym-lite python3 revision_phase1_eval.py \
  --policy PartitionOnlyRL --model-type attentive_ppo \
  --partition-weights $W/partition_attentive_ppoclip_seed{seed}.weights.h5 \
  --seeds 1 2 3 --protocols eval500 --label PartitionOnlyRL_attentive_ppoclip >> $LOG 2>&1
```

Total remaining ≈ 22 h. **Do not run anything else CPU-heavy while these run** —
query latency is the reward signal, so contention corrupts the measurements.

## 3. Analyse (safe to run at any time, including partway through)

```bash
python3 revision/stats/analyze_phase1.py
```

Writes `results_table.csv`, `significance_matrix.csv`, `figures/forest_reward_*.{pdf,png}`
and `SUMMARY.md` into `revision/phase1_stats/`.

## 4. Then

- **STOP for review** of Phase 1 before any manuscript text is written — the
  outcome decides how the paper is reframed.
- Phase 2 remainder: inference-latency benchmark (needs an idle CPU) + cost model
  with the amortised RL-infrastructure term.
- Phases 3–7 as per the plan. Note Phase 5 (drift/mixed workloads) requires
  changes to `simulation.py`; do **not** edit it while an evaluation batch is
  running, because each variant launches a fresh process that would pick up the
  edit mid-batch.

## State at shutdown

| item | status |
|---|---|
| Heuristics (WAT, Threshold10_C128, AlwaysCompact_C128, Threshold10_C64) | ✅ 5 seeds, both protocols |
| attention+PPO-clip | ✅ 5/5 std1000, 4/5 eval500 (seed 5 pending) |
| DDQN | seed 1 std1000 only |
| MLP-PPO | seed 1 std1000 only |
| attention+AWR (published recipe) | seed 1 std1000 only — **intentionally not extended** (defect documented) |
| CompactOnly / PartitionOnly | not started |
| v3 checkpoints (30 + 10 ppoclip) | ✅ trained, gated, archived |
| Phase 2 training-cost + hyperparameter tables | ✅ built |
| Phase 8 code verification + Table 7 guard | ✅ done |

## Standing note for the final report (user, 2026-08-07)

The cost model's flat $0.005-per-compaction term does not scale. A real
compaction is a distributed rewrite whose cost grows with bytes rewritten; at
production scale a single operation can occupy many workers for minutes. Flat
per-operation pricing therefore under-charges aggressive strategies such as
Threshold10_C128 and AlwaysCompact_C128, and any "heuristic X is cheaper than
RL" conclusion drawn from it is an artifact. Phase 3 adds a size-proportional
compaction cost and reports whether the ranking flips.

The three compaction targets (C32/C64/C128) are a conceptual low/medium/high
spectrum, not a hyperparameter search over three values. Each carries a distinct
trade-off across compaction cost, block utilisation, query latency and storage,
and the manuscript should present them that way.

Both points are recorded in `revision/phase2_compute/SUMMARY.md` §4 and must
appear in `revision/MASTER_SUMMARY.md`.
