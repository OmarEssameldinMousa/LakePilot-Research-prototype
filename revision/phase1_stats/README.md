# Phase 1 — Multi-seed training and statistical validation

Reviewer demand: independent training runs + CIs/non-parametric tests, replacing
n=5-episodes-from-one-checkpoint claims.

## Pipeline

### 1. Training (Kaggle GPU) — YOUR ACTION REQUIRED
Data provenance (verified against the saved outputs of the original notebooks):
`kaggle_training_data.zip` = exactly the union of the two published training pools —
14 baseline episodes → compaction (14,000 transitions) and 6 partition-source episodes
(2 runs each of WorkloadAwareThreshold / PartitionExploration / Random_Partition) →
partition (4,990 transitions). The notebook reproduces both counts exactly.

1. Upload `kaggle_training_data.zip` (in this folder) as a Kaggle dataset.
2. Open `kaggle_train_phase1.ipynb` on Kaggle, attach the dataset, enable GPU, Run All (~2–4 h).
3. Download `phase1_weights.zip`, unpack **into this folder** so you get
   `revision/phase1_stats/phase1_weights/weights/{spec}_{arch}_seed{k}.weights.h5`.

### 2. Evaluation (local docker stack)
Heuristic batch (no weights needed) — **already running** via:
```bash
docker exec -w /app/src lakegym-lite python3 revision_phase1_eval.py \
    --policy WorkloadAwareThreshold --seeds 1 2 3 4 5
# then Threshold10_C128, AlwaysCompact_C128, No_Maintenance
```
Progress: `tail -f revision/phase1_stats/heuristic_batch.log`. Runs are resumable
(a run with manifest `status: done` is skipped).

RL batch — after weights arrive:
```bash
for ARCH in attentive_ppo mlp_ppo ddqn; do
docker exec -w /app/src lakegym-lite python3 revision_phase1_eval.py \
    --policy MultiAgentRL --model-type $ARCH \
    --compact-weights  /app/revision/phase1_stats/phase1_weights/weights/compaction_${ARCH}_seed{seed}.weights.h5 \
    --partition-weights /app/revision/phase1_stats/phase1_weights/weights/partition_${ARCH}_seed{seed}.weights.h5 \
    --seeds 1 2 3 4 5 --label MultiAgentRL_${ARCH}
done
# Ablations (AttentivePPO weights):
# --policy CompactOnlyRL  --compact-weights ...   --label CompactOnlyRL_attentive_ppo
# --policy PartitionOnlyRL --partition-weights ... --label PartitionOnlyRL_attentive_ppo
```

### 3. Statistics
`revision/stats/stats_utils.py` — bootstrap 95% CIs (10k resamples), Wilcoxon +
paired permutation tests, Cliff's delta / rank-biserial, Holm–Bonferroni.
Analysis script (results_table.csv, significance matrix, forest plot) is generated
after the eval batches finish.

## Protocol (fixed — do not change between policies)
- Per seed slot k ∈ {1..5}:
  - `std1000`: 1 × 1000-step episode, `workload_plan_v5.json`, env seed `11000+k`
  - `eval500`: 5 × 500-step episodes, `workload_plan_v5_eval.json`, env seeds `15000+100k+e`
- Env seeds are functions of (protocol, slot, episode) only → episodes are **matched
  across policies** for paired tests.
- Frozen RL evaluation is **deterministic argmax** (`stochastic=False`); the published
  frozen evals sampled stochastically and, for DDQN, acted with ε=1.0 (random) — this is
  a documented protocol correction.
- One episode ≈ 25–35 min (1000-step) / 12–18 min (500-step) on the local docker stack.
  Full heuristic batch ≈ 30–40 h; RL batch ≈ 25–30 h.

## Outputs
- `raw/<policy>/seed<k>/<protocol>/transitions_ep<e>.csv` — per-step transitions
- `raw/.../episode_summary.csv` — per-episode metrics (reward, latency, files, pruning)
- `raw/.../manifest.json` — seeds, config, git commit, wall-clock, hardware
- `phase1_weights/weights/*_manifest.json` — training cost/hyperparams (feeds Phase 2)
