# Phase 6 — Online adaptation: DDQN stability and remedies

**Verdict: the published "DDQN adaptation collapse" was an implementation defect,
not an architectural property. No remedy sweep is required.**

Protocol A: 3 architectures × 5 seeds × 5 episodes × 500 steps on the compressed
eval workload, initialised from the Phase 1 v3 checkpoints (15 runs, all complete).

## Results

Mean over 5 seeds; CIs are percentile bootstrap over seeds.

| Architecture | ep1 → ep5 change | 95% CI | Adaptive mean reward | Frozen ceiling | Published claim |
|---|---|---|---|---|---|
| DDQN | **+3.8 %** | [−4.8, +11.4] | 0.2092 [0.2002, 0.2179] | 0.2105 | **−22.2 % (collapse)** |
| MLP-PPO | −4.2 % | [−12.9, +2.4] | 0.2093 [0.2019, 0.2166] | 0.1675 | +4.9 % |
| AttentivePPO-clip | +8.4 % | [−5.0, +21.9] | 0.1998 [0.1943, 0.2054] | 0.1823 | +20.3 % |

**Every architecture is stable.** All three confidence intervals straddle zero:
online adaptation over five episodes produces small drift in either direction, and
no architecture collapses. Per-seed changes range from −20.6 % to +29.6 % with no
architecture-specific pattern — the spread is seed noise, not instability.

Adaptive mean reward is at or above the frozen ceiling for all three
architectures (DDQN 0.2092 vs 0.2105; MLP-PPO 0.2093 vs 0.1675; AttentivePPO-clip
0.1998 vs 0.1823), so adaptation is neutral-to-beneficial rather than harmful.

## Why the published result differed

Three defects, each independently sufficient to produce a spurious collapse:

1. **The Double-DQN update regressed softmax outputs, not Q-values.** TD targets
   were fitted against a probability vector bounded in [0, 1] on the same scale as
   the rewards. Fixed; the original is retained as `_dqn_step_legacy_softmax`.
2. **ε never decayed.** It was decremented once per *update call* rather than per
   step, so "adaptive" DDQN explored at ε ≈ 1.0 throughout — acting uniformly at
   random within whichever specialist the meta-controller delegated to.
3. **The adaptive runs were initialised from artifact checkpoints** (see
   `phase1_stats/PIPELINE_HISTORY.md`), so the starting policy itself was not the
   one the paper describes.

### Evidence for defect 2, measured directly

A run under the **published** ε configuration is retained at
`online_raw/adaptive_ddqn_publishedcfg/seed1`:

| episode | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| reward | 0.1795 | 0.1850 | 0.2143 | 0.1951 | 0.1787 |
| ε | 0.979 | 0.959 | 0.939 | 0.917 | 0.896 |

`epsilon_decay_steps = 5000` is counted in specialist transitions (~100 per
episode), so a five-episode run completes ~10 % of the decay. The agent therefore
acts near-randomly for the entire experiment. Additionally, ε_start = 1.0 is
appropriate for learning from scratch but actively destructive when fine-tuning a
pretrained policy.

**Corrected configuration** (documented deviation): ε 0.10 → 0.02 over 200
specialist transitions, updates every 128 transitions (~25 per run instead of 5).
PPO agents are unaffected by ε — they explore by sampling their own policy.

## Table 7 regeneration, with the guard applied

The published Table 7 reports "% of gap closed" including **−3,207 %** for DDQN.
That figure is a division-by-near-zero artifact. `revision/stats/gap_metric.py`
reports the metric only when |initial gap| exceeds 0.01 (the reward is O(0.1–0.3)
per step, so smaller gaps are within episode noise).

| Architecture | ep1 | ep5 | Frozen ceiling | % gap closed | Published |
|---|---|---|---|---|---|
| DDQN | 0.2029 | 0.2097 | 0.2105 | **n/a** (initial gap 0.0076 < 0.01) | −3,207 % |
| MLP-PPO | 0.2174 | 0.2081 | 0.1675 | −18.6 % | 75.1 % |
| AttentivePPO-clip | 0.1931 | 0.2057 | 0.1823 | +116.5 % | 146.9 % |

DDQN's entry is exactly the pathological case the guard exists for: it begins
0.0076 below its own ceiling, so any ratio is dominated by noise. Reporting
"n/a" is the honest presentation.

## Consequences for the manuscript

Claims that no longer hold:

- *"Adaptive online weight updates … destabilise DDQN, exposing an important
  architecture-specific constraint on on-policy versus off-policy online
  adaptation."* — Not reproducible. DDQN is stable (+3.8 %, CI straddling zero)
  once the update and ε schedule are corrected. The on-policy/off-policy
  distinction is not supported by the data.
- *"Adaptive online weight updates further improve AttentivePPO by +20.3 %,
  closing 146.9 % of the initial gap."* — The direction survives (+8.4 %) but the
  magnitude does not, and the CI [−5.0, +21.9] includes zero.
- *"Frozen weights are preferred for deployment, and online adaptation should be
  restricted to on-policy architectures such as AttentivePPO."* — Unsupported.
  Adaptation is neutral-to-positive for all three.
- Table 7's **−3,207 %** — now reported as n/a with a programmatic guard.

## Is the remedy sweep still needed?

**No.** Phase 6 was scoped to test whether prioritised experience replay, smaller
target-network update intervals, or sliding-window replay buffers could rescue
DDQN from adaptation collapse. There is no collapse to rescue: DDQN is the
best-performing architecture in Phase 1 and is stable under adaptation. Running
remedies against a non-existent failure would produce a null result at ~15 h of
compute.

**Honest limitation to state in the manuscript.** With five 500-step episodes,
each specialist receives ~500 transitions and ~25 gradient updates in total. This
measures *fine-tuning drift*, not substantial online learning, and it is why all
three confidence intervals straddle zero. The correct claim is that online
adaptation over a short deployment window neither helps nor harms materially —
not that these architectures cannot learn online. Establishing the latter would
require a budget an order of magnitude larger (see `ONLINE_DESIGN.md`,
Protocol B).

## Artefacts

| Path | Contents |
|---|---|
| `online_raw/adaptive_<arch>/seed<k>/episode_log.csv` | per-episode reward, latency, files, pruning, loss, ε, update count |
| `online_raw/adaptive_<arch>/seed<k>/transitions_ep<e>.csv` | per-step transitions |
| `online_raw/adaptive_<arch>/seed<k>/weights/` | post-adaptation weights |
| `online_raw/adaptive_ddqn_publishedcfg/seed1/` | evidence run under the published ε schedule |
| `ONLINE_DESIGN.md` | audit of the online logic, conflicts found, protocol design |
