# Phase 2 — Compute, overhead, and cost-model integration: SUMMARY

Reviewer demands addressed: wall-clock training time, GPU requirements, per-step
inference latency, parameter counts, a full hyperparameter appendix, and whether
the cost saving survives once RL infrastructure cost is included.

## 1. Training cost

Per deployable agent (both specialists), 5 seeds each. Hardware differs by
variant and is reported per row — wall-clock is **not** comparable across
hardware.

| Variant | Hardware | Params | Per agent | All 5 seeds |
|---|---|---|---|---|
| DDQN | Tesla T4 | 193,739 | 2.26 min | 11.3 min |
| MLP-PPO | Tesla T4 | 192,203 | 1.73 min | 8.7 min |
| AttentivePPO-clip | local CPU | 170,571 | 3.30 min | 16.5 min |
| AttentivePPO-AWR (published recipe) | Tesla T4 | 170,571 | 2.52 min | 12.6 min |

Full per-specialist detail in `training_cost_table.csv`; all hyperparameters in
`hyperparameter_appendix.csv`.

## 2. Inference latency

10,000 randomised observations per measurement, CPU only (Intel i7-1165G7,
no GPU present).

| Component | Median | p95 | Params |
|---|---|---|---|
| MetaController (rule-based urgency) | **0.0019 ms** | 0.003 ms | — |
| MLP-PPO — full hierarchical step | 3.48 ms | 4.06 ms | 192,203 |
| DDQN — full hierarchical step | 4.80 ms | 6.28 ms | 193,739 |
| AttentivePPO-clip — full hierarchical step | 10.73 ms | 12.57 ms | 170,571 |

The rule-based meta-controller is effectively free (~2 µs). **AttentivePPO-clip
is 3× slower than MLP-PPO despite having fewer parameters** — self-attention is
quadratic in window length, so parameter count understates its serving cost.
This is an additional deployment argument for DDQN beyond its reward advantage.

Against ~344 ms of mean query time per step, a ~5 ms policy decision is ~1.4 %
of a single query's latency.

## 3. Regenerated Table 11 (cost model with RL infrastructure)

1,000-step episode. Operational terms use the published constants unchanged
(query $1e-5/ms, compaction $0.005/op, storage $1e-6/KB-step) so the comparison
is like-for-like. New terms: inference at $0.096/h CPU, offline training at
$0.526/h GPU amortised over 100 deployment episodes.

| Policy | Query | Compact | Storage | Inference | Training | **Total** | vs no-maint |
|---|---|---|---|---|---|---|---|
| Threshold10_C128 | 3.31 | 0.62 | 4.45 | — | — | **8.37** | −83.2 % |
| **DDQN** | 3.44 | 0.87 | 4.49 | 0.0001 | 0.0002 | **8.79** | **−82.4 %** |
| AttentivePPO-clip | 4.98 | 0.82 | 4.57 | 0.0003 | 0.0003 | 10.37 | −79.2 % |
| Threshold10_C64 | 5.71 | 0.62 | 4.57 | — | — | 10.90 | −78.1 % |
| WorkloadAwareThreshold | 5.73 | 1.05 | 4.57 | — | — | 11.35 | −77.2 % |
| MLP-PPO | 6.37 | 0.58 | 4.67 | 0.0001 | 0.0002 | 11.62 | −76.7 % |
| AlwaysCompact_C128 | 3.48 | 5.00 | 4.43 | — | — | 12.92 | −74.1 % |
| No_Maintenance (published run) | 42.98 | 0.00 | 6.88 | — | — | 49.86 | 0 % |

**Verdict: the −82.9 % claim survives, at −82.4 % for DDQN.** The RL
infrastructure term is $0.0003 per episode against an $8.79 total — **0.003 %**.
Including it changes the saving from 82.364 % to 82.363 %.

## 4. ⚠ Load-bearing caveat: the compaction cost term does not scale

The model charges a **flat $0.005 per compaction operation, independent of table
size, bytes rewritten, or cluster time**. That is unrealistic and it biases the
ranking in a specific direction:

- A real compaction is a distributed rewrite job. At production scale a single
  operation can occupy many workers for minutes, and its cost scales with the
  **bytes rewritten**, not with a per-call constant.
- Aggressive compaction policies are therefore charged far too little here.
  `Threshold10_C128` — which appears cheapest at $8.37 — is among the most
  aggressive strategies in the suite. Under a size-proportional cost term its
  advantage would shrink or invert.
- Under the flat model no RL agent ever repays its training cost relative to
  `Threshold10_C128`, because it costs more per episode before training is even
  counted. **That conclusion is an artifact of the flat term and must not be
  reported without this caveat.**

Phase 3's cost-model sweep therefore adds a **size-proportional compaction cost**
(cost ∝ `total_size_kb` at the moment of compaction) alongside the flat model,
and reports whether the ranking flips. The per-step data needed for this is
already logged in every Phase 1 transitions CSV.

### On the three compaction targets

C32 / C64 / C128 are not three arbitrary settings; they instantiate **low,
medium and high** compaction targets, each with a distinct trade-off across
compaction cost, block utilisation, query latency and storage:

| Target | Compaction ops | Utilisation | Query latency | Storage |
|---|---|---|---|---|
| C32 (low) | many small rewrites | poor (~0.40) | worst (file proliferation) | highest file count |
| C64 (medium) | balanced | ~0.74 | intermediate | intermediate |
| C128 (high) | fewer, larger rewrites | best (~1.34) | best | fewest files |

Under the **flat** per-operation cost, larger targets look strictly better
because they trigger fewer operations. Under a **size-proportional** cost they
would be charged for rewriting more data per operation, which is what makes the
trade-off real. The manuscript should present the three targets as a conceptual
low/medium/high spectrum rather than as a search over three hyperparameter
values, and should state the cost-model limitation explicitly.

## 5. Honest framing for the manuscript

- "−82.9 % versus no maintenance, with RL infrastructure adding under 0.01 % of
  total cost" — supported.
- "RL is the cheapest policy" — **not** supported under this cost model. RL buys
  *reward* (pruning, adaptivity, lower latency), not minimum dollar cost.
- Any statement about RL versus heuristic cost must be accompanied by the
  flat-compaction-cost caveat above, or deferred to the size-proportional model
  in Phase 3.

## Artefacts

| File | Contents |
|---|---|
| `training_cost_table.csv` | per specialist: wall-clock mean ± sd, hardware, params, epochs |
| `training_cost_per_agent.csv` | cost to produce one deployable agent and all 5 seeds |
| `hyperparameter_appendix.csv` | full hyperparameter table for the appendix |
| `inference_latency.{json,csv}` | median/p95/p99 per component, per architecture |
| `cost_model_table11.csv` | regenerated Table 11 with inference + amortised training |
| `cost_model_assumptions.json` | every unit price and the amortisation horizon |
