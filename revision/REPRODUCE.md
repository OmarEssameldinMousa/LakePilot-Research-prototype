# Reproducing the Scientific Reports revision

Every result in the revision was produced by the scripts below, on the
environment pinned in `LakeGymLite/requirements.txt` (container) and
`requirements-analysis.txt` (host analysis). Each run writes a
`manifest.json` recording its seed, configuration, git commit, hardware and
wall-clock time, so any figure can be traced back to the exact run that produced
it.

## 0. Environment

```bash
docker compose up -d spark-master minio mc rest lakegym-lite
docker ps            # wait until minio / iceberg-rest / spark-master are healthy
```

Container: Python 3.10.12, Spark 3.5.7, Apache Iceberg via REST catalog + MinIO.
Host analysis: Python 3, `pip install -r requirements-analysis.txt`.

**Two operational rules that affect correctness:**

1. **Do not run other CPU-heavy work while an evaluation is in flight.** Query
   latency is part of the reward, so contention changes the measurements.
2. **Run one seed per process** for long batches. Sharing a SparkSession across
   many seeds caused cumulative JVM degradation and silently corrupted data
   (see `phase1_stats/PIPELINE_HISTORY.md`).

## 1. Training (Kaggle GPU, ~1 h each)

| notebook | dataset to attach | output → |
|---|---|---|
| `phase1_stats/kaggle_train_phase1.ipynb` | `phase1_stats/kaggle_training_data_v2.zip` | `phase1_stats/phase1_weights/` |
| `phase7_attention/kaggle_train_phase7.ipynb` | same dataset | `phase7_attention/phase7_weights/` |

The data zip contains exactly the published training pools (14 baseline episodes →
compaction, 6 partition-source episodes → partition) plus the
`RecoveryExploration` episodes added in the revision.

## 2. Gate the checkpoints before spending evaluation time

```bash
docker exec -w /app/src lakegym-lite python3 revision_probe_checkpoints.py
```

Exits non-zero if any architecture NOOP-locks on a stale, file-heavy table. This
has already prevented one 28-hour wasted batch.

## 3. Evaluation

All runs are resumable: a run whose manifest says `status: "done"` is skipped.

```bash
W=/app/revision/phase1_stats/phase1_weights/weights

# Phase 1 — heuristics
docker exec -w /app/src lakegym-lite python3 revision_phase1_eval.py \
    --policy WorkloadAwareThreshold --seeds 1 2 3 4 5

# Phase 1 — RL agents (one seed per process for long batches)
docker exec -w /app/src lakegym-lite python3 revision_phase1_eval.py \
    --policy MultiAgentRL --model-type ddqn \
    --compact-weights  $W/compaction_ddqn_seed{seed}.weights.h5 \
    --partition-weights $W/partition_ddqn_seed{seed}.weights.h5 \
    --seeds 1 2 3 4 5 --label MultiAgentRL_ddqn

# Phase 3 — sensitivity sweeps
docker exec -w /app/src lakegym-lite python3 revision_phase3_sweep.py --sweep meta
docker exec -w /app/src lakegym-lite python3 revision_phase3_sweep.py --sweep prune

# Phase 4 — oracle meta-controller (environment rollback)
docker exec -w /app/src lakegym-lite python3 revision_phase4_oracle.py \
    --model-type ddqn --weights-label ddqn --seeds 1 --protocols std1000

# Phase 5 — held-out workloads
docker exec -w /app/src lakegym-lite python3 revision_phase1_eval.py \
    --policy MultiAgentRL --model-type ddqn \
    --compact-weights $W/compaction_ddqn_seed{seed}.weights.h5 \
    --partition-weights $W/partition_ddqn_seed{seed}.weights.h5 \
    --seeds 1 2 3 4 5 --protocols drift500 mixed500 \
    --label MultiAgentRL_ddqn --out /app/revision/phase5_generalization/raw

# Phase 6 — online adaptation
docker exec -w /app/src lakegym-lite python3 revision_online_eval.py \
    --model-type ddqn \
    --compact-weights  $W/compaction_ddqn_seed{seed}.weights.h5 \
    --partition-weights $W/partition_ddqn_seed{seed}.weights.h5 \
    --seeds 1 2 3 4 5 --episodes 5 --epsilon-start 0.10 --epsilon-decay-steps 200 \
    --label adaptive_ddqn --out /app/revision/phase6_ddqn/online_raw

# Phase 7 — attention traces
docker exec -w /app/src lakegym-lite python3 revision_phase7_attention.py \
    --weights-label attentive_ppoclip --seeds 1 2 3
```

## 4. Validate, then analyse

**Always validate first.** A Spark driver crash does not raise inside the
simulator — metrics return 0 and latency −1, which the reward maps to exactly
0.0, so a dead run can look like data:

```bash
python3 revision/stats/validate_runs.py          # add --fix to delete corrupted seeds
```

```bash
python3 revision/stats/analyze_phase1.py             # tables, significance matrix, forest plots
python3 revision/stats/build_manuscript_tables.py    # rewritten Tables 1 and 2
python3 revision/stats/analyze_phase3.py             # meta-controller + pruning sensitivity
python3 revision/stats/analyze_phase3_posthoc.py     # cost sweep + reward weights
python3 revision/stats/analyze_phase5.py             # generalization / degradation
python3 revision/stats/analyze_phase7_attention.py   # attention profile + transitions
python3 revision/phase2_compute/build_training_cost_table.py
python3 revision/phase2_compute/build_cost_model.py
```

Inference benchmark (needs an idle machine):

```bash
docker exec -w /app/src lakegym-lite python3 revision_bench_inference.py --iterations 10000
```

## Environment variables that alter behaviour

All default to the published values, so unset behaviour is bit-identical to the
original system.

| variable | default | used by |
|---|---|---|
| `LAKEGYM_META_THETA` | 0.35 | Phase 3 meta-controller sweep |
| `LAKEGYM_META_UTIL_BOOST` | 0.15 | ” |
| `LAKEGYM_META_COMPACT_COOLDOWN` | 4 | ” |
| `LAKEGYM_META_PARTITION_COOLDOWN` | 20 | ” |
| `LAKEGYM_PRUNING_SCALE` | 1.0 | Phase 3 pruning perturbation |
| `LAKEGYM_COMPACT_WINDOW` | 10 | Phase 7 tuning study |
| `LAKEGYM_PARTITION_WINDOW` | 20 | ” |
| `LAKEGYM_EMBED_DIM` | 64 | ” |
| `LAKEGYM_NUM_HEADS` | 4 | ” |

## Directory map

```
revision/
├── stats/                    statistics utilities and all analysis scripts
├── phase1_stats/             multi-seed validation  + PIPELINE_HISTORY.md
├── phase2_compute/           training cost, inference latency, cost model
├── phase3_sensitivity/       meta-controller, pruning, prices, reward weights
├── phase4_oracle/            oracle meta-controller (rollback)
├── phase5_generalization/    drift / mixed held-out workloads
├── phase6_ddqn/              online adaptation  + ONLINE_DESIGN.md
├── phase7_attention/         attention analysis + parameter-matched MLP
└── phase8_manuscript/        code verification, Zenodo checklist
```

Every phase directory contains a `SUMMARY.md` with its numbers and verdicts.
`revision/MASTER_SUMMARY.md` aggregates all of them.
