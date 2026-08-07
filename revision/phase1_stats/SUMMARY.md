# Phase 1 — Multi-seed training and statistical validation: SUMMARY

**Status: complete.** 84 evaluation runs, all validated clean
(`revision/stats/validate_runs.py`). Every policy was evaluated on two protocols
with environment seeds that are a function of (protocol, seed slot, episode)
only — never of the policy — so all comparisons below are **paired**.

- Protocol `std1000`: 1 × 1,000-step episode per seed, v5 standard workload
- Protocol `eval500`: 5 × 500-step episodes per seed, compressed eval workload
- Confidence intervals: percentile bootstrap, 10,000 resamples, over episodes
- p-values: paired permutation tests, Holm–Bonferroni corrected within each
  (protocol, metric) family; Wilcoxon signed-rank reported alongside
- Effect sizes: Cliff's delta

---

## Headline results (eval500, 25 matched episodes per policy)

| Policy | Reward (95% CI) | Seed sd | Latency (ms) | Files | Pruning |
|---|---|---|---|---|---|
| **DDQN (hierarchical RL)** | **0.2105 [0.2050, 0.2155]** | 0.0104 | 220 | 37.4 | 0.204 |
| WorkloadAwareThreshold | 0.1943 [0.1923, 0.1964] | 0.0020 | 324 | 49.9 | 0.248 |
| AlwaysCompact_C128 | 0.1903 [0.1901, 0.1904] | 0.0002 | 198 | 22.1 | 0.000 |
| AttentivePPO-clip (hierarchical RL) | 0.1823 [0.1669, 0.1963] | 0.0403 | 331 | 51.6 | 0.225 |
| Threshold10_C128 | 0.1710 [0.1708, 0.1712] | 0.0003 | 203 | 26.1 | 0.000 |
| MLP-PPO (hierarchical RL) | 0.1675 [0.1579, 0.1772] | 0.0254 | 317 | 61.6 | 0.221 |
| Threshold10_C64 | 0.1273 [0.1267, 0.1280] | 0.0011 | 321 | 46.8 | 0.000 |
| CompactOnly (ablation) | 0.1202 [0.1086, 0.1315] | 0.0428 | 314 | 48.4 | 0.000 |
| PartitionOnly (ablation) | −0.1099 [−0.1195, −0.1011] | 0.0116 | 3643 | 793 | 0.080 |

The 1,000-step ranking agrees (Spearman ρ = 0.964, p < 0.001); see
`MANUSCRIPT_TABLES.md` for both protocols with published values alongside.

---

## (a) Is RL significantly better than the best heuristic?

**Not as a class — it depends entirely on the architecture.** The honest claim is
*"the best RL configuration significantly outperforms the best heuristic"*, not
*"RL outperforms heuristics"*.

Against `WorkloadAwareThreshold` (best heuristic, 0.1943), eval500:

| Agent | Δ reward | Holm p | Cliff's δ | Verdict |
|---|---|---|---|---|
| DDQN | **+0.0162** | **0.00105** | +0.71 (large) | **significantly better** |
| AttentivePPO-clip | −0.0120 | 0.4656 | +0.10 (negligible) | indistinguishable |
| MLP-PPO | −0.0268 | 0.0018 | −0.71 (large) | significantly **worse** |

DDQN beats **every** policy in the study, all Holm-corrected p ≤ 0.0018:
AlwaysCompact_C128 (+0.0202), AttentivePPO-clip (+0.0282), Threshold10_C128
(+0.0395), Threshold10_C64 (+0.0832), WorkloadAwareThreshold (+0.0162).

**All five DDQN seeds** individually beat the best heuristic (0.198–0.224), so
the result is not carried by one lucky run.

**Mechanism.** DDQN wins with *less* pruning than the best heuristic
(0.204 vs 0.248) but markedly better compaction: 37.4 files vs 49.9 and 220 ms
vs 324 ms. Its advantage comes from state-dependent compaction-target selection
(C64/C128/C32 mixed by ingestion rate), not from partitioning more aggressively.

## (b) Are the three architectures distinguishable?

**Partly.** eval500, RL-vs-RL, Holm-corrected:

| Comparison | Δ | p | δ | Verdict |
|---|---|---|---|---|
| DDQN vs AttentivePPO-clip | +0.0282 | 0.0018 | +0.44 (medium) | significant |
| DDQN vs MLP-PPO | +0.0430 | 0.00105 | +0.84 (large) | significant |
| AttentivePPO-clip vs MLP-PPO | +0.0148 | 0.3107 | +0.33 (medium) | **not significant** |

**The published "+3.5% attention advantage over MLP-PPO" does not survive
multi-seed testing.** Attention and MLP-PPO are statistically indistinguishable
on reward; DDQN is significantly ahead of both.

**Stability separates the architectures as sharply as mean performance.**
Across-seed sd of mean reward: DDQN 0.0104, MLP-PPO 0.0254, AttentivePPO-clip
0.0403. Attention is ~4× more variable than DDQN: three of its five seeds beat
the best heuristic (0.199–0.220) while two fail badly (0.124, 0.158),
accumulating 2.5× more files. A single-checkpoint evaluation — the published
protocol — could therefore have reported anything from "clearly best" to
"clearly worse" purely by seed luck.

## Ablation: both specialists are required, and the interaction is non-additive

eval500, all Holm-corrected p = 0.0022, all |δ| ≥ 0.97:

| Configuration | Reward | Δ vs full | Latency | Files | Pruning |
|---|---|---|---|---|---|
| Full hierarchical agent | 0.1823 | — | 331 ms | 51.6 | 0.225 |
| CompactOnly (partition specialist removed) | 0.1202 | −0.0621 | 314 ms | 48.4 | 0.000 |
| PartitionOnly (compaction specialist removed) | −0.1099 | −0.2922 | 3643 ms | 793 | 0.080 |

Two points worth making in the manuscript:

1. **Removing partitioning isolates cleanly.** Full and CompactOnly are
   physically almost identical (51.6 vs 48.4 files; 331 vs 314 ms). The entire
   0.062 difference is pruning: 0.225 vs exactly 0.000.
2. **The interaction is non-additive.** PartitionOnly's pruning collapses to
   0.080 despite being the partition-only agent — without compaction, data is
   never rewritten into the new layout, so partitioning cannot deliver its
   benefit. Neither specialist's contribution is independent of the other.

---

## Methodological corrections carried by this phase

1. **Statistics.** The published Table 2 reported step-level mean ± s.d. pooled
   over 5,000 steps. That s.d. measures step-to-step variation within episodes
   and treats autocorrelated steps as independent; it is not an uncertainty
   estimate for the mean. All intervals here are computed over independent
   episodes, with across-seed sd reported separately.
2. **Training-pipeline defects** (full audit in `PIPELINE_HISTORY.md`): a
   trajectory-identity leak through the staleness features, a train/serve
   normalization skew that made the published frozen agents behave as a constant
   COMPACT_128KB policy via out-of-distribution saturation, majority-class
   collapse of the AWR objective, a Double-DQN update that regressed softmax
   outputs instead of Q-values, and an ε schedule that never decayed.
3. **Evaluation protocol.** Frozen evaluation is now deterministic (argmax).
   Previously it sampled stochastically, and frozen DDQN acted with ε = 1.0 —
   i.e. uniformly at random within the delegated specialist.
4. **Environment-health guard.** A Spark driver crash does not raise inside the
   simulator; it silently yields latency −1 and file_count 0, which the reward
   maps to exactly 0.0. Runs now abort instead of writing `status: done` over
   meaningless data, and `validate_runs.py` audits every batch.

## Environment reproducibility

The heuristics reproduce the published figures within 0.003 on the 1,000-step
workload (AlwaysCompact_C128 0.1869 → 0.1870; Threshold10_C128 0.1734 → 0.1759;
WorkloadAwareThreshold 0.2111 → 0.2139). The environment is therefore stable,
and the changes in the RL numbers are attributable to the training-pipeline
corrections rather than environment drift.

## Consequences for the manuscript

Claims that no longer hold as written:

- *"All three hierarchical RL agents outperform 17 heuristic and workload-aware
  baselines"* — MLP-PPO now ranks below three heuristics; only DDQN beats the
  best heuristic.
- *"AttentivePPO achieves the highest mean global reward (0.2201)"* — DDQN is
  now highest (0.2306 on std1000, 0.2105 on eval500); AttentivePPO-clip is third.
- *"+3.5% improvement over MLP-PPO"* — not statistically significant.
- *"+4.3% improvement over the best heuristic"* — becomes +7.8% (std1000) /
  +8.3% (eval500), for DDQN rather than AttentivePPO.

Claims that survive or strengthen:

- Hierarchical RL beats the best heuristic — now with a significance test,
  a large effect size, and consistency across all five seeds.
- Joint compaction + partitioning beats either specialist alone, with the
  non-additive interaction now measured rather than asserted.
- DDQN's reward *improved* over its published value (0.2196 → 0.2306) once the
  pipeline was corrected.

## Artefacts

| File | Contents |
|---|---|
| `results_table.csv` | mean ± bootstrap 95% CI, per policy × protocol × metric |
| `significance_matrix.csv` | every pairwise paired test, Holm-corrected |
| `MANUSCRIPT_TABLES.md` | rewritten Tables 1 and 2 with published values alongside |
| `table{1,2}_rewritten_*.csv` | machine-readable versions |
| `figures/forest_reward_*.{pdf,png}` | forest plots, 300 dpi |
| `PIPELINE_HISTORY.md` | training-pipeline audit trail and scope decisions |
| `raw/<policy>/seed<k>/<protocol>/` | per-step transitions, episode summaries, manifests |
