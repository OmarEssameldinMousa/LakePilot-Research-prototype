# Online / adaptive learning experiments — audit and design

Covers regeneration of the published adaptive results (paper Figs 9–11, Table 7),
the Phase 6 DDQN-remedy question, and a proposed from-scratch control.

Runner: `LakeGymLite/src/revision_online_eval.py` (headless, seeded, resumable).
It replaces the NiceGUI-driven adaptive path in `app.py`.

## 1. Conflicts found in the existing online logic

| # | Issue | Impact | Status |
|---|---|---|---|
| 1 | `OnlineSpecialistTrainer` always seeded from `PPOConfig.seed`; `MultiAgentOnlineTrainer` never passed `config` for DDQN | Every "independent" DDQN online run used seed 42 — multi-seed DDQN runs would have been **identical** | **fixed** |
| 2 | Both specialists seeded with the same value | Partition model initialised from the same RNG state as the compaction model | **fixed** (partition offset by 10,000) |
| 3 | Online DDQN update regressed `softmax(Q)` instead of Q-values | TD targets on a [0,1] probability scale — a likely direct cause of the published "DDQN adaptation collapse" | fixed earlier; original kept as `_dqn_step_legacy_softmax` |
| 4 | ε never decayed (decremented once per *update call*, not per step) | "Adaptive" DDQN explored at ε≈1.0 throughout | fixed earlier (linear decay in env steps) |
| 5 | Frozen evaluation sampled stochastically | Frozen ≠ deterministic | fixed earlier (`greedy` flag) |
| 6 | Updates only once per episode; `PPOConfig.rollout_steps` unused | ~5 gradient updates for an entire run — acceptable for fine-tuning, a strawman for from-scratch | `--update-every N` added |

### Known limitation, documented rather than "fixed"

Each specialist's buffer contains only the steps at which **that specialist was
delegated**, but GAE/TD bootstrapping treats consecutive buffer entries as
adjacent in time. If the compaction agent acts at env steps 5, 12 and 30, the
discount γ is applied per *decision*, not per *elapsed env step*. This is a
semi-MDP (options) formulation, and it is what the published system does. Making
γ time-aware (γ^Δt) would change the algorithm, so we keep the published
semantics and state the assumption explicitly in the manuscript.

## 2. Protocol A — adaptive from offline checkpoints (must be regenerated)

The published adaptive results are invalid for three independent reasons: they
were initialised from artifact checkpoints, the DDQN update was defective, and ε
never decayed. Re-running with v3 checkpoints and the corrected update is
required regardless of any new experiment.

**Expected scientific payoff.** The paper's headline adaptive finding is that
DDQN *collapses* during adaptation. Issues 3 and 4 above are each sufficient to
produce exactly that. If the corrected DDQN adapts normally, Phase 6 changes from
"do PER / target-update tweaks rescue DDQN?" to "the reported collapse was an
implementation defect; no remedy is required" — a stronger result, and one that
matters because DDQN is now the best-performing architecture (Phase 1).

### Configuration change, with the evidence that motivated it

A first run under the **published** configuration (`adaptive_ddqn_publishedcfg`,
seed 1, retained as evidence) showed the experiment measures almost nothing:

| episode | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| reward | 0.1795 | 0.1850 | 0.2143 | 0.1951 | 0.1787 |
| ε | 0.979 | 0.959 | 0.939 | 0.917 | 0.896 |
| gradient updates | 1 | 1 | 1 | 1 | 1 |

Two defects in the published setup, independent of the ones already fixed:

1. **ε never leaves ~0.9.** `epsilon_decay_steps=5000` is counted in specialist
   transitions (~100 per episode), so a five-episode run completes ~10 % of the
   decay. The agent therefore acts almost uniformly at random throughout.
2. **ε_start = 1.0 is wrong for fine-tuning.** Starting from a *pretrained*
   policy, maximal exploration destroys the behaviour the experiment is meant to
   adapt. ε_start = 1.0 is appropriate only for the from-scratch protocol.

Combined with one gradient update per episode (5 in total), the published
protocol cannot express adaptation in either direction — which also means the
published "DDQN degrades −22.2 %" and "AttentivePPO improves +20.3 %" figures
largely reflect exploration noise rather than learning.

**Corrected configuration** (documented as a deliberate deviation):
ε: 0.10 → 0.02 over 200 specialist transitions; `--update-every 128`
(≈ 4–8 updates per episode, ~25 per run instead of 5). PPO agents are unaffected
by ε — they explore by sampling from their own policy.

Configuration: 3 architectures × 5 seeds × 5 episodes × 500 steps,
eval workload. ≈ 12 h.

### Preliminary observation (published config, seed 1)

Even at ε ≈ 0.9, corrected DDQN showed **no collapse**: −0.5 % from episode 1 to
5, versus the published −22.2 %. This is consistent with the hypothesis that the
published collapse was caused by the softmax-Q update defect rather than by any
architectural property of off-policy learning.

## 3. Protocol B — from scratch (proposed control)

### Should we run it? Yes — but framed correctly.

It cannot be a fair test of "can online RL learn this task", and we should not
present it as one. PPO typically needs 10⁵–10⁶ environment steps; here a step is
a real Iceberg operation costing ~1–2 s, so 10⁵ steps ≈ **55 h per seed per
architecture**. Any budget we can afford is far short of convergence.

What it *can* do — and what a reviewer will actually ask — is quantify **what the
offline pretraining phase buys**. Framed as a negative control it answers
"why offline pretraining?" with data instead of assertion:

> Starting from random initialisation under an identical (indeed more generous)
> online budget, the hierarchical system reached only X after N episodes, versus
> Y for the offline-pretrained agent, and remained below the best heuristic
> throughout. Offline pretraining is therefore not a convenience but a
> requirement at realistic environment cost.

### Design

- **Budget parity**: same episode count and length as Protocol A, so the curves
  can be plotted on shared axes.
- **Deliberately favourable to from-scratch**: `--update-every 128`, giving it
  roughly 20× more gradient updates than the published once-per-episode cadence.
  Stating that we tilted the comparison in its favour makes the negative result
  much harder to dismiss.
- **All three architectures.** DDQN is the interesting case: being off-policy
  with replay, it is far more sample-efficient than PPO, so it is the most
  likely to show non-trivial learning. A finding of "only the off-policy
  architecture makes progress from scratch" would be genuinely publishable.
- **3 seeds** — enough to show a trend; we do not need tight CIs for a control.
- **Reference lines** on the figure: No_Maintenance, best heuristic
  (WorkloadAwareThreshold), and the offline-pretrained frozen ceiling.

Configuration: 3 architectures × 3 seeds × 10 episodes × 500 steps,
`--update-every 128`. ≈ 20–25 h (slower than Protocol A because a poor policy
accumulates files, which makes every query slower).

Trimmed option: 2 seeds × 8 episodes ≈ 12 h.

## 4. Which workload — 1000 or 500?

**Use the 500-step eval workload for both protocols.** Reasons:

1. **Comparability** — the published adaptive results (Figs 9–11) used 5 × 500,
   so regenerating on the same protocol keeps the figures directly comparable.
2. **What is being measured is change *across* episodes.** The dependent variable
   is the learning curve, so the number of episodes matters more than the length
   of each. For a fixed wall-clock budget, 500-step episodes give twice as many
   learning cycles.
3. **No loss of workload variety** — the 500-step plan contains all five query
   profiles and the full ingestion schedule, simply at half duration.
4. Phase 1 showed policy rankings agree across the two lengths at Spearman
   ρ = 0.964, so nothing is hidden by the shorter episodes.

**Where 1000 steps is worth it:** a supplementary long-horizon check for
within-episode degradation (catastrophic forgetting during a long deployment),
which short episodes cannot reveal because each `sim.reset()` wipes the table.
Recommended as 1 seed × 3 episodes × 1000 steps per architecture (≈ 4 h), run
only after Protocol A shows the adaptation dynamics are stable.

## 5. Commands (run when the CPU is free)

```bash
W=/app/revision/phase1_stats/phase1_weights/weights
OUT=/app/revision/phase6_ddqn/online_raw

# Protocol A — adaptive from v3 checkpoints (label maps arch -> weight label)
for SPEC in "ddqn ddqn" "mlp_ppo mlp_ppo" "attentive_ppo attentive_ppoclip"; do
  set -- $SPEC; ARCH=$1; LBL=$2
  docker exec -w /app/src lakegym-lite python3 revision_online_eval.py \
    --model-type $ARCH \
    --compact-weights  $W/compaction_${LBL}_seed{seed}.weights.h5 \
    --partition-weights $W/partition_${LBL}_seed{seed}.weights.h5 \
    --seeds 1 2 3 4 5 --episodes 5 --update-every 0 \
    --label adaptive_${LBL} --out $OUT
done

# Protocol B — from scratch (no weights), favourable update cadence
for ARCH in ddqn mlp_ppo attentive_ppo; do
  docker exec -w /app/src lakegym-lite python3 revision_online_eval.py \
    --model-type $ARCH --seeds 1 2 3 --episodes 10 --update-every 128 \
    --label scratch_${ARCH} --out $OUT
done
```

Both are resumable: a seed whose manifest says `status: "done"` is skipped.
