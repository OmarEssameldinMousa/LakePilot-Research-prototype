# Phase 7 — Attention analysis and parameter-matched MLP: SUMMARY

Reviewer objections addressed: *"the attention benefit is asserted, never
demonstrated"* and *"the +3.5 % edge is confounded with capacity."*

**Scope note.** The three-architecture comparison from Phase 1 is unchanged and is
not affected by anything here: those models were trained under identical
configuration (same pools, 150/200 epochs, batch 64, balanced sampling, 5 seeds)
and evaluated on matched environment seeds. Phase 7 adds (1) a capacity control
for the attention-vs-MLP question and (2) a *separate, exploratory* tuning study
asking whether attention improves with a configuration suited to it. The tuning
study is deliberately **not** entered into the three-architecture comparison,
because that comparison is defined by matched configuration.

---

## 1. Capacity control — the attention advantage does not survive

The published models differ in both encoder *and* size, and **in opposite
directions per specialist** — a detail the original paper does not note:

| specialist | AttentivePPO | MLP-PPO | |
|---|---|---|---|
| compaction | 84,741 | 70,469 | MLP has 17 % **fewer** |
| partition | 85,830 | 121,734 | MLP has 42 % **more** |

Matched widths (compaction h1=128/h2=288, partition h1=128/h2=192) bring the MLP
to within **0.06 %** of AttentivePPO's parameter count. Evaluated on 25 matched
episodes, 5 seeds:

| model | reward (95 % CI) | params |
|---|---|---|
| AttentivePPO-clip | 0.1823 [0.1669, 0.1963] | 170,571 |
| **Parameter-matched MLP** | **0.1774 [0.1664, 0.1885]** | 170,667 |
| MLP-PPO (published widths) | 0.1675 [0.1579, 0.1772] | 192,203 |

**Attention vs parameter-matched MLP: +0.0049, permutation p = 0.54,
Wilcoxon p = 0.65, Cliff's δ = +0.16 (small) — not significant.**

Combined with Phase 1 (attention vs published MLP: p = 0.31), the published
+3.5 % attention advantage does not survive either an unmatched or a matched
comparison.

**Incidental finding worth reporting:** the matched MLP *outperforms* the
published MLP (0.1774 vs 0.1675) while using **fewer total parameters**.
Matching redistributed capacity — compaction 70k → 85k, partition 122k → 86k —
so the published MLP was simply mis-allocated, starving the specialist that
matters more. Architecture sizing, not architecture family, accounted for part of
the published gap.

## 2. Tuning study — attention was undertrained, not under-capacity

Exploratory screen, 3 episodes per variant, **all variants reported**:

| variant | change | reward | sd | params |
|---|---|---|---|---|
| `attn_A_base` | baseline (GPU reference) | 0.1841 | 0.0263 | 170,571 |
| **`attn_B_longer`** | **epochs ×2 (300/400)** | **0.1994** | **0.0100** | 170,571 |
| `attn_C_wider` | embed 128, heads 8 | 0.1830 | 0.0528 | 1,196,171 (**7×**) |
| `attn_D_context` | window 20/40 | 0.1765 | 0.0179 | 172,491 |
| `attn_E_combo` | all three | 0.1830 | 0.0494 | 1,200,011 (**7×**) |

`attn_A_base` reproduces Phase 1's attention (0.1841 vs 0.1823) despite CPU→GPU
training, which validates comparing across the two.

Only **longer training** helps. **Seven times the parameters changes nothing**
(0.1830, and seed variance doubles), and **longer context actively hurts**
(0.1765).

### Confirmatory run — and it corrected the screen

The screen suggested +8.3 % for `attn_B_longer`. Re-run on the **full eval500
protocol** (3 seeds × 5 episodes = 15 episodes, paired against the Phase 1
attention runs on the same environment seeds):

| | reward (95 % CI) | per-seed sd |
|---|---|---|
| `attn_B_longer` | **0.1949 [0.1866, 0.2033]** | 0.0187 |
| baseline attention | 0.1823 [0.1669, 0.1963] | 0.0403 |

**Paired on 15 matched episodes: +0.0058, permutation p = 0.11,
Wilcoxon p = 0.17, Cliff's δ = +0.08 (negligible) — not significant.**

The screen overstated the effect roughly threefold, because its 3-episode samples
happened to favour B. **The defensible claim is that doubling the training budget
brings attention to approximate parity with the best heuristic (0.1949 vs
0.1943) but leaves it below DDQN (0.2105), and the improvement is not
statistically significant.**

**Fairness statement required in the manuscript:** DDQN was *not* given an
equivalent hyperparameter sweep. "Tuned attention approaches DDQN" means
*attention can be brought closer with tuning*, not *attention is as good*.

## 3. Mechanism — attention is a static recency gate, not workload-aware

Multi-head attention scores captured at every step of frozen 1,000-step rollouts,
3 seeds. Attention was recorded on an independently maintained feature window
updated at every step, because the meta-controller delegates on only ~20 % of
steps and the agent's own window would otherwise be stale.

### (a) Attention mass by window position

| steps back from newest | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| compaction (uniform 0.100) | 0.150 | 0.125 | 0.113 | 0.102 | 0.098 | 0.090 |
| partition (uniform 0.050) | **0.517** | 0.105 | 0.051 | 0.032 | 0.024 | 0.024 |

- **Partition**: 52 % of all attention mass on the *single newest step*
  (recent-3 = 0.673, **4.49× uniform**).
- **Compaction**: recent-3 = 0.388, only **1.29× uniform** — close enough to flat
  that the global average pooling which follows performs a similar function.

Attention concentrated on one position is a **recency gate**, not sequence
modelling: it selects the latest frame rather than relating steps to one another,
and a feed-forward network over a flattened window expresses a fixed positional
weighting trivially.

### (b, c) Response to workload transitions

Attention mass on the 3 most recent positions, within ±25 steps of the four phase
boundaries (200/400/600/800) versus mid-phase:

| specialist | transition | mid-phase | Δ | seeds with p<0.05 |
|---|---|---|---|---|
| compaction | 0.3822 | 0.3888 | −0.0066 | 1/3 |
| partition | 0.6757 | 0.6727 | +0.0031 | 2/3 |

Both effects are negligible, and the partition direction is **inconsistent across
seeds** (+0.001, −0.012, +0.020; seed 3's rank-biserial has the opposite sign to
seeds 1 and 2). **There is no systematic increase in recency focus when the
workload shifts** — which is exactly what a "workload-aware" attention mechanism
would require.

## 4. Synthesis — the three results are mutually consistent

Attention learns a **static positional weighting** rather than dynamic context
modelling. Everything else in this phase follows from that:

- it is statistically indistinguishable from a parameter-matched MLP (p = 0.54),
  because an MLP can express a fixed positional weighting;
- 7× capacity changes nothing, because the mechanism is not capacity-limited;
- a longer window *hurts*, because extending a static profile only adds positions
  that dilute it;
- and it does not react at transitions, because the weighting is static by
  construction.

## 5. Consequences for the manuscript

Does not hold as written:

- *"a +3.5 % improvement over MLP-PPO"* — not significant unmatched (p = 0.31) or
  parameter-matched (p = 0.54).
- Any claim that the attention encoder demonstrably attends to workload
  transitions — it does not.

Survives or is newly supported:

- Attention **does** weight recent steps, strongly so for the partition
  specialist (4.49× uniform) — the mechanism exists, it is simply static.
- The two specialists use it very differently (0.517 vs 0.150 on the newest
  step), which is itself a reportable observation about what each task needs.
- Attention's weakness is **training budget and capacity allocation**, not the
  encoder: doubling epochs recovers roughly parity with the best heuristic, and
  correctly re-allocating MLP capacity improved that model too.

## Artefacts

| File | Contents |
|---|---|
| `phase7_weights/` | 40 checkpoints (5 attention variants × 3 seeds + matched MLP × 5 seeds) + per-run manifests |
| `raw/MLP_matched/` | capacity-control evaluation, 5 seeds × 5 episodes |
| `raw/attn_*/` | tuning screen, 3 seeds × 1 episode per variant |
| `raw/attn_B_longer_full/` | confirmatory run, 3 seeds × 5 episodes |
| `attention_traces/attention_seed*.npz` | raw scores (1000 steps × 4 heads × q × k) |
| `attention_profile.csv`, `attention_transitions.csv` | position profile and transition statistics |
| `figures/attention_profile.{pdf,png}`, `figures/attention_transitions.{pdf,png}` | 300 dpi |
