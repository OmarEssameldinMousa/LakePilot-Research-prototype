# Phase 1 — offline training pipeline history and evidence

Audit trail of why the training pipeline changed during the revision. Each claim
below is backed by a committed artifact so it can be defended to reviewers.

## v1 — faithful reproduction of the published pipeline (+ normalization fix)

Trained 3 archs × 2 specialists × 5 seeds on the exact published pools
(14,000 compaction / 4,990 partition transitions, provenance verified against the
original notebooks' saved outputs).

**Result: collapse.** The first live evaluation episode
(`raw/MultiAgentRL_attentive_ppo/seed1/std1000/`) produced 999 NOOPs and 1
partition action → 1,879 files, 8.8 s queries, mean global reward −0.34.

**Diagnosis — two interacting defects:**

1. *Trajectory-identity leakage.* `steps_since_compact` fingerprints which
   heuristic generated a trajectory (AlwaysCompact ≈ 0; No_Maintenance → ∞).
   Advantage-weighted regression, being weighted behavior cloning, learned the
   fingerprint instead of the table state. Measured at a **fixed** state
   (files = 30, util = 0.10): P(NOOP) rises 0.00 → 0.94 as `steps_since_compact`
   goes 0 → 30. A freshly deployed agent starts stale, so it NOOPs, which makes it
   staler — a self-reinforcing lock.
2. *The published checkpoints only escaped this by accident.* Training z-scored
   `rows_ingested`/`ingestion_rate` per dataset while deployment used constants
   (5, 3)/(10, 8), pushing live features 10–50× out of distribution and saturating
   the network into a **constant COMPACT_128KB** policy (verified: original
   checkpoint selects C128 on 100 % of training windows under deployed scaling;
   under corrected scaling it selects C128 on 67 %). The published frozen
   AttentivePPO results therefore measure an accidentally-good constant policy,
   not learned workload adaptivity.

Artifacts: `phase1_weights_v1_artifact/`, `raw/MultiAgentRL_attentive_ppo/seed1/std1000/`.

## v2 — pipeline correction

1. Staleness feature caps applied identically in training and deployment:
   compaction `exp(−min(ssc,10)/10)`, partition `exp(−min(ssp,60)/30)`.
2. `RecoveryExploration` counterfactual episodes (3 × 1000 steps, real
   environment): randomized build-up/compact cycles supplying the
   "stale, file-heavy table → compact → outcome" evidence absent from every
   heuristic log. 97 compactions, ~35 at file counts > 100.
3. Compaction credit horizon γ = 0.95 (was 0.99).
4. Training budgets standardized across architectures (compaction 150 epochs,
   partition 200, batch 64 everywhere), removing the published recipes' 150/200
   epoch and 64/128 batch confound.

**Result: partial.** Gate output in `v2_probe_gate.log`:

| architecture | P1 stale table | P3 workload alignment |
|---|---|---|
| AttentivePPO | ✗ NOOP 5/5 seeds | 5/15 |
| MLP-PPO | ✓ C64 5/5 | 8/15 |
| DDQN | ✓ compacts 5/5 | 1/15 |

The one architecture still failing was the one using plain AWR, on a compaction
pool that is **66.5 % NOOP** (partition pool: 71.5 %). MLP-PPO (standardized
advantages) and DDQN (no cloning term) were unaffected — i.e. residual
majority-class collapse of the AWR objective, not a feature problem.

## v3 — uniform class-imbalance treatment (current)

A local 3-way ablation on AttentivePPO-compaction (1 seed, 100 epochs,
`v3_trial.log`) isolated the remedy:

| variant | P1 (stale table) | P2 low/mid/burst |
|---|---|---|
| A: plain AWR (= v2) | ✗ NOOP at every checkpoint | NOOP / NOOP / NOOP |
| B: balanced sampling + global z-scored advantages | ✓ from epoch 40 (p = 0.55) | NOOP / C32 / NOOP |
| **C: balanced sampling + per-action z-scored advantages** | **✓ from epoch 20 (p = 0.59)** | NOOP / C32 / C32 |

v3 adopts **C for every architecture and both specialists**, so the remedy cannot
itself become an architecture confound. This is also the treatment the published
*partition* recipe already used — v3 applies it consistently rather than
introducing a new technique.

Per-architecture *algorithms* remain as published (AWR / offline PPO-clip /
Double-DQN); the fully algorithm-matched comparison is Phase 7.

### v3 outcome (`v3_probe_gate.log`)

| architecture | P1 stale table | P2 target vs ingestion | P3 workload alignment |
|---|---|---|---|
| AttentivePPO (attention + AWR) | acts on 3/5 seeds | weak (NOOP / C32 only) | 5/15 |
| MLP-PPO (MLP + PPO-clip) | 5/5 (C64) | clean C32 → C64/C128 | 9/15 |
| DDQN | 4/5 | clean C32 → C64/C128 | 3/15 |

Gate passed, but AttentivePPO stayed marginal, and the live sanity episode for its
seed 1 — one of the two NOOP seeds — collapsed exactly as the probe predicted
(999 NOOPs, 1,878 files, mean global reward −0.094). The probe is therefore a
validated predictor of live behaviour and should gate every future checkpoint set.

### Why AttentivePPO alone remains fragile

AWR is advantage-weighted *behaviour cloning*: it can only raise probability on
actions present in the data, so on a 66.5 %-NOOP pool it drifts back to the
majority class. PPO-clip additionally *lowers* probability on negative-advantage
actions, which is why the identical class-imbalance remedy rescued MLP-PPO 5/5
and DDQN 4/5 but only lifted AttentivePPO to 3/5.

Because "AttentivePPO" = attention + AWR while "MLP-PPO" = MLP + PPO-clip, the
published architecture comparison confounds encoder with objective. The revision
therefore trains **attention + PPO-clip** under the identical v3 recipe
(`LakeGymLite/src/revision_train_local.py`), isolating the encoder. This is a
confound removal, not a per-architecture tuning step: no architecture receives a
treatment the others do not.

### Known limitation of the per-action advantage treatment

Per-action z-scoring fixes the NOOP collapse but normalises advantages *within*
each action class, which removes cross-action ranking at a given state. Symptom:
MLP-PPO chose C32 for 99 of 103 compactions, although measured outcomes rank
targets C128 > C64 > C32 on both global and compaction reward:

| policy | global reward | compaction reward | files | utilisation |
|---|---|---|---|---|
| Threshold10_C32 | 0.028 | 0.087 | 182 | 0.40 |
| Threshold10_C64 | 0.120 | 0.219 | 88 | 0.74 |
| Threshold10_C128 | 0.171 | 0.287 | 47 | 1.34 |

The reward's hand-designed `target_bonus` compounds this by awarding C32 its
largest bonus at low ingestion, contradicting the measured ranking — direct
evidence for the reviewers' "reward weights are unjustified magic numbers"
comment, to be quantified in Phase 3. If the encoder-decoupled variant is also
C32-heavy, the next candidate treatment is balanced sampling with critic-baseline
advantages (A = G − V(s)), which preserves state-relative cross-action ranking;
it would likewise be applied uniformly to all architectures.

## Gate before evaluation

`LakeGymLite/src/revision_probe_checkpoints.py` — synthetic-state probes, exit 1
if any architecture NOOP-locks on a stale table. Run it on every new checkpoint
set **before** launching an evaluation batch; it costs seconds and has already
prevented one 28-hour wasted run.

P2/P3 are reported but not gated: weak target-size differentiation or workload
alignment is a legitimate finding to report honestly, whereas a NOOP-locked agent
is a pipeline defect.

## Evaluation scope decisions (2026-08-02)

- **attention + AWR (the published recipe) is NOT evaluated at full scale.** Its
  defect is already established by the 5-seed probe gate (NOOP on 2/5 seeds) and
  by a full 1000-step sanity episode on seed 1, which collapsed to 999 NOOPs,
  1,878 files and a mean global reward of −0.094 (kept in
  `raw/MultiAgentRL_attentive_ppo/seed1/std1000/`). Spending ~9 h to characterise
  a recipe that is being replaced adds no evidence; the sanity episode plus the
  probe table is what the response letter needs.
- **PartitionOnly ablation reduced to 3 seeds × 500-step episodes.** Because it
  never compacts, its file count runs away exactly as No_Maintenance does
  (published run: 2,176 files, 5.3 s mean latency), which would have cost ~19 h.
  Its role is the synergy claim — showing that removing compaction hurts — and it
  loses by a wide margin at every seed, so it needs enough runs for a confidence
  interval rather than high precision.
- The same reasoning previously retired the 5-seed No_Maintenance evaluation; the
  published single-episode run remains the no-maintenance anchor for the cost
  table, reported without a CI and flagged as such.

## Scope of regeneration

Contaminated by artifact checkpoints, must be regenerated from v3: frozen RL
evaluations (all architectures), adaptive runs (initialized from offline
weights), CompactOnly/PartitionOnly ablations, the RL rows of the cost table,
Table 7, and the Phase 7 attention analysis.

Unaffected: the environment, all heuristic results (published and the new 5-seed
CIs in `heuristics_preliminary_table.csv`), and the heuristic training logs.
