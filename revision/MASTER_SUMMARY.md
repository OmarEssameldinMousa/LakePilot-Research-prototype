# LakePilot — Scientific Reports revision: MASTER SUMMARY

All phases complete. **216 seeded evaluation runs validated clean, 0 corrupted**
(`revision/stats/validate_runs.py`), plus 16 online-adaptation runs which use a
separate per-episode log format. Every run carries a manifest recording its seed,
configuration, git commit, hardware and wall-clock time.

> **Note on reviewer numbering.** The sections below are organised by *demand*,
> not by reviewer-comment label, because the R1.1–R1.7 / R2 numbering is in the
> reviewer letter rather than in this repository. Map each block to its comment
> number when assembling the response letter — the content is one-to-one with the
> demands in the revision plan.

---

# PART A — Numbers for the manuscript

## A1. Headline result (replaces Table 1 / Table 2)

500-step evaluation workload, 25 matched episodes per policy, 5 seeds.
Intervals are percentile bootstrap 95 % CIs; comparisons are paired by
environment seed.

| Policy | Reward (95 % CI) | Seed sd | Latency ms | Files | Pruning |
|---|---|---|---|---|---|
| **DDQN (hierarchical RL)** | **0.2105 [0.2050, 0.2155]** | 0.0104 | 220 | 37.4 | 0.204 |
| WorkloadAwareThreshold | 0.1943 [0.1923, 0.1964] | 0.0020 | 324 | 49.9 | 0.248 |
| AlwaysCompact_C128 | 0.1903 [0.1901, 0.1904] | 0.0002 | 198 | 22.1 | 0.000 |
| AttentivePPO-clip | 0.1823 [0.1669, 0.1963] | 0.0403 | 331 | 51.6 | 0.225 |
| Threshold10_C128 | 0.1710 [0.1708, 0.1712] | 0.0003 | 203 | 26.1 | 0.000 |
| MLP-PPO | 0.1675 [0.1579, 0.1772] | 0.0254 | 317 | 61.6 | 0.221 |
| Threshold10_C64 | 0.1273 [0.1267, 0.1280] | 0.0011 | 321 | 46.8 | 0.000 |
| CompactOnly (ablation) | 0.1202 [0.1086, 0.1315] | 0.0428 | 314 | 48.4 | 0.000 |
| PartitionOnly (ablation) | −0.1099 [−0.1195, −0.1011] | 0.0116 | 3643 | 793 | 0.080 |

1,000-step ranking agrees at Spearman ρ = 0.964 (p < 0.001).

## A2. Significance (Holm-corrected paired permutation tests)

**DDQN beats every policy**, all p ≤ 0.0018:

| comparison | Δ | p | Cliff's δ |
|---|---|---|---|
| vs WorkloadAwareThreshold | +0.0162 | 0.00105 | +0.71 large |
| vs AlwaysCompact_C128 | +0.0202 | 0.00075 | +0.84 large |
| vs AttentivePPO-clip | +0.0282 | 0.0018 | +0.44 medium |
| vs Threshold10_C128 | +0.0395 | 0.00075 | +1.00 large |
| vs Threshold10_C64 | +0.0832 | 0.00075 | +1.00 large |

All five DDQN seeds individually beat the best heuristic (0.198–0.224).

**Not significant:** AttentivePPO-clip vs MLP-PPO (p = 0.31), and
AttentivePPO-clip vs the best heuristic (p = 0.47). MLP-PPO is significantly
*worse* than the best heuristic (−0.0268, p = 0.0018).

## A3. Ablation — both specialists required, interaction non-additive

| configuration | reward | Δ vs full | pruning |
|---|---|---|---|
| Full hierarchical agent | 0.1823 | — | 0.225 |
| CompactOnly | 0.1202 | −0.0621 | 0.000 |
| PartitionOnly | −0.1099 | −0.2922 | 0.080 |

All Holm p = 0.0022, |δ| ≥ 0.97. Full and CompactOnly are physically almost
identical (51.6 vs 48.4 files; 331 vs 314 ms) — the entire gap is pruning.
PartitionOnly's pruning **collapses to 0.080 despite being the partition agent**,
because without compaction data is never rewritten into the new layout.

## A4. Compute and cost (Table 11 regenerated)

Per-step inference, 10,000 iterations, CPU:

| component | median | p95 | params |
|---|---|---|---|
| Meta-controller (rule-based) | **0.0019 ms** | 0.003 ms | — |
| MLP-PPO full step | 3.48 ms | 4.06 ms | 192,203 |
| DDQN full step | 4.80 ms | 6.28 ms | 193,739 |
| AttentivePPO-clip full step | 10.73 ms | 12.57 ms | 170,571 |

Attention is **3× slower than MLP despite fewer parameters** (attention is
quadratic in window length).

Cost per 1,000-step episode, with the RL-infrastructure term added:

| policy | operational | inference | training (amortised /100 ep) | total | vs no-maint |
|---|---|---|---|---|---|
| Threshold10_C128 | 8.37 | — | — | **8.37** | −83.2 % |
| **DDQN** | 8.79 | 0.0001 | 0.0002 | **8.79** | **−82.4 %** |
| AttentivePPO-clip | 10.37 | 0.0003 | 0.0003 | 10.37 | −79.2 % |
| No_Maintenance | 49.86 | — | — | 49.86 | 0 % |

**The −82.9 % saving survives at −82.4 %.** RL infrastructure is 0.003 % of total
cost. **But RL is not the cheapest policy** — `Threshold10_C128` is, because it
compacts less often (124 vs 173 operations). RL buys reward, not minimum cost.

Training cost per deployable agent: 1.7–3.3 min (Tesla T4).

## A5. Sensitivity (Phase 3)

| analysis | result |
|---|---|
| Meta-controller constants | reward varies 0.0295 (14.5 %) across the whole grid; defaults mildly conservative — compaction cooldown 2 would give +8.2 % |
| Pruning matrix ×{0.8…1.2} | RL advantage positive at **every** scale (+0.0034 to +0.0106) |
| Cost prices ×{0.5, 1, 2} (27 combos) | ranking never flips; also unchanged under a size-proportional compaction cost |
| Reward weights ±25 % | Spearman ≥ 0.95, DDQN top under **all nine** weightings |

## A6. Generalization (Phase 5)

| policy | baseline | drift500 | mixed500 |
|---|---|---|---|
| **DDQN** | 0.2105 | 0.2009 (−4.5 %) | 0.1994 (−5.3 %) |
| WorkloadAwareThreshold | 0.1943 | 0.1869 (−3.8 %) | 0.1834 (−5.6 %) |
| AlwaysCompact_C128 | 0.1903 | 0.1902 (−0.05 %) | 0.1898 (−0.2 %) |
| AttentivePPO-clip | 0.1823 | 0.1788 (−1.9 %) | 0.1638 (−10.2 %) |
| MLP-PPO | 0.1675 | 0.1609 (−4.0 %) | 0.1469 (−12.3 %) |

DDQN's margin over the best heuristic is essentially unchanged (+0.0162 in
distribution, +0.0140 drift, +0.0160 mixed). Degradation tracks pruning loss;
`AlwaysCompact_C128` — which never partitions — is the only policy that does not
degrade, confirming the effect is partition-alignment specific.

## A7. Online adaptation (Phase 6) — replaces Table 7

| architecture | ep1 → ep5 | 95 % CI | adaptive mean | frozen ceiling |
|---|---|---|---|---|
| DDQN | **+3.8 %** | [−4.8, +11.4] | 0.2092 | 0.2105 |
| MLP-PPO | −4.2 % | [−12.9, +2.4] | 0.2093 | 0.1675 |
| AttentivePPO-clip | +8.4 % | [−5.0, +21.9] | 0.1998 | 0.1823 |

All CIs straddle zero: adaptation is neutral over a short deployment window, and
**no architecture collapses**. Table 7's "% gap closed" for DDQN is now **n/a**
(initial gap 0.0076 < 0.01 threshold) rather than −3,207 %.

## A8. Attention analysis (Phase 7)

Capacity matched to within 0.06 %:

| model | reward (95 % CI) | params |
|---|---|---|
| AttentivePPO-clip | 0.1823 [0.1669, 0.1963] | 170,571 |
| Parameter-matched MLP | 0.1774 [0.1664, 0.1885] | 170,667 |

**p = 0.54, δ = +0.16** — the +3.5 % attention advantage does not survive.

Mechanism (3 seeds, frozen 1,000-step rollouts):

| steps back | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| compaction (uniform 0.100) | 0.150 | 0.125 | 0.113 | 0.102 |
| partition (uniform 0.050) | **0.517** | 0.105 | 0.051 | 0.032 |

Attention is a **static recency gate** — 52 % of partition attention on the single
newest step, compaction only 1.29× uniform — and shows **no systematic response
at workload transitions** (Δ = −0.0066 / +0.0031, direction inconsistent across
seeds).

Tuning study: only longer training helps (+0.0058, p = 0.11 — not significant),
reaching heuristic parity (0.1949 vs 0.1943) but staying below DDQN. **7×
parameters changes nothing; 2× context hurts.**

---

# PART B — Numbers for the point-by-point response letter

## B1. "Provide confidence intervals / non-parametric tests across independent training runs"

**Done, and it changed the conclusions.** 3 architectures × 5 independent seeds,
evaluated on matched environment seeds; bootstrap CIs (10,000 resamples),
Wilcoxon signed-rank *and* paired permutation tests, Cliff's delta,
Holm–Bonferroni correction. See A1–A2.

We additionally identified a deeper statistical problem the comment implies: the
published Table 2 reported **step-level mean ± s.d. pooled over 5,000 steps**.
That s.d. measures within-episode variation and treats autocorrelated steps as
independent samples; it is not an uncertainty estimate for the mean. All
intervals are now computed over independent episodes, with across-seed sd
reported separately.

**Key admission:** seed variance is large for the attention architecture
(sd 0.0403 — three of five seeds beat the best heuristic, two fail badly), so a
single-checkpoint evaluation could have reported almost any ranking.

## B2. "Wall-clock training time, GPU requirements, per-step inference latency; does the saving survive RL infrastructure cost?"

**Yes — the saving survives at −82.4 %**, with RL infrastructure at 0.003 % of
total cost. Full tables in A4, plus a complete hyperparameter appendix
(`phase2_compute/hyperparameter_appendix.csv`).

**Caveat we state ourselves:** the cost model charges a flat $0.005 per
compaction regardless of bytes rewritten, which under-charges aggressive
strategies. We add a size-proportional variant; the ranking is unchanged at this
simulation's scale, but we flag that production-scale costs would diverge.

## B3. "No sensitivity analysis for meta-controller constants; pruning matrix and prices are hardcoded assumptions; reward weights are unjustified"

All four addressed — see A5. Every conclusion is stable. We report honestly that
the meta-controller sweep is **powered for estimation, not hypothesis testing**
(3 episodes per configuration; an exact permutation test on n = 3 cannot reach
p < 0.25).

## B4. "Quantify the loss attributable to the rule-based meta-controller"

A greedy one-step oracle (environment rollback, realised rewards) **does not
upper-bound** the rule-based controller — it scores 0.0546 vs 0.2306. The reason
is measurable: the per-step reward difference between delegations has median
0.0069, while query-latency noise alone is σ = 0.0120, and trial-to-realised
correlation is only 0.156. Maintenance payoff is delayed; one-step realised
reward cannot see it.

**The practical bound instead comes from Phase 3:** the best rule-based
configuration scores 0.2200 vs the default 0.2033, so the loss attributable to
suboptimal constants is **≈ 8 %**.

## B5. "The compressed eval workload is near in-distribution; test drift / mixed profiles"

Two genuinely held-out workloads built and evaluated — see A6. The RL advantage
persists. We note the design is powered for estimation (one episode per seed;
n = 5 gives a permutation-test floor of p = 0.0625).

## B6. "Test prioritised replay, smaller target-update frequencies, buffers that discard old transitions"

**The remedy sweep is unnecessary because the collapse does not reproduce.** With
the Double-DQN update corrected to regress raw Q-values (it regressed softmax
outputs) and a working ε schedule (ε never decayed — "adaptive" DDQN explored at
ε ≈ 1.0), DDQN is stable at +3.8 %. Three published claims do not survive:
DDQN's instability, the on-policy/off-policy distinction drawn from it, and the
recommendation to restrict adaptation to on-policy architectures.

## B7. "The attention benefit is asserted, never demonstrated; +3.5 % is confounded with capacity"

Both confirmed as problems — see A8. Capacity-matched, the advantage is not
significant (p = 0.54). The encoder is a static recency gate that does not react
to workload transitions. We also found the published MLP was **mis-allocated**
(17 % fewer parameters on compaction, 42 % more on partition); correcting the
allocation improved it (0.1774 vs 0.1675) with *fewer* total parameters.

## B8. Manuscript corrections

| item | status |
|---|---|
| Dueling aggregation `Q = V + (A − mean(A))` | **verified correct** — no change needed |
| Double-DQN target (online argmax, target eval) | correct in structure; the *update* consumed softmax outputs — fixed and documented |
| Table 7 "−3,207 % gap closed" | now **n/a** behind a programmatic guard |
| Data Availability URL | **truncated and 404** — `…-prot` should be `…-prototype` (see `phase8_manuscript/ZENODO_CHECKLIST.md`) |

## B9. Self-reported: a pipeline defect we found and corrected

Not raised by reviewers; we report it because it changes the published numbers.

The published frozen agents were affected by a **train/serve normalization
skew**: training z-scored `rows_ingested`/`ingestion_rate` per dataset while
deployment used fixed constants (5, 3)/(10, 8), pushing inference features 10–50×
out of distribution and saturating the network into an effectively **constant
COMPACT_128KB policy** — adaptivity by artifact rather than by learning.

Correcting it exposed a second defect: **trajectory-identity leakage**. The
staleness features encode *which heuristic generated a trajectory*
(AlwaysCompact ≈ 0, No_Maintenance → ∞), so advantage-weighted regression cloned
the generating policy rather than reading table state. At a fixed table state,
P(NOOP) rises 0.00 → 0.94 as `steps_since_compact` goes 0 → 30.

Both are fixed (feature capping, unified normalization, counterfactual recovery
data, balanced action sampling). **The environment is unchanged and reproduces
the published heuristic baselines within 0.003** (AlwaysCompact_C128:
0.1869 → 0.1870), so the differences are attributable to the pipeline
corrections rather than to environment drift.

---

# PART C — Recommended reframing

The evidence supports a stronger and more defensible thesis than the original:

> Hierarchical RL beats hand-tuned heuristics for lakehouse maintenance — but
> which architecture wins is determined by the **training objective**, not the
> encoder, and naive offline pipelines fail in ways invisible to
> single-checkpoint evaluation.

Claims to retire: "all three RL agents outperform all baselines" (MLP-PPO ranks
below three heuristics); "AttentivePPO achieves the highest reward" (DDQN does);
"+3.5 % attention advantage" (not significant); "adaptation destabilises DDQN"
(an implementation defect).

Claims that strengthen: hierarchical RL beats the best heuristic — now with
p = 0.001, δ = +0.71, and consistency across all five seeds; joint compaction and
partitioning beats either alone with a **measured** non-additive interaction;
the advantage persists under held-out workload shift; and DDQN's reward
*improved* over its published value (0.2196 → 0.2306) once the pipeline was
corrected.

Two findings generalize beyond lakehouses and are arguably the paper's most
transferable contributions: **trajectory-identity leakage** when training offline
RL on logs from multiple policies, and the **signal-below-noise** explanation for
why greedy meta-control fails where cooldown-based pacing succeeds.

---

## Phase index

| phase | topic | summary |
|---|---|---|
| 1 | Multi-seed statistical validation | `phase1_stats/SUMMARY.md`, `PIPELINE_HISTORY.md` |
| 2 | Compute, overhead, cost model | `phase2_compute/SUMMARY.md` |
| 3 | Sensitivity analyses | `phase3_sensitivity/SUMMARY.md` |
| 4 | Oracle meta-controller | `phase4_oracle/SUMMARY.md` |
| 5 | Generalization | `phase5_generalization/SUMMARY.md` |
| 6 | Online adaptation / DDQN remedies | `phase6_ddqn/SUMMARY.md`, `ONLINE_DESIGN.md` |
| 7 | Attention + parameter-matched MLP | `phase7_attention/SUMMARY.md` |
| 8 | Manuscript & repo | `phase8_manuscript/CODE_VERIFICATION.md`, `ZENODO_CHECKLIST.md` |
| — | Reproduction | `REPRODUCE.md` |
