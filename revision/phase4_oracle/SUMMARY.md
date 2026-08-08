# Phase 4 — Oracle meta-controller: SUMMARY

Reviewer demand: quantify the performance loss attributable to the rule-based
meta-controller, i.e. separate the "when to act" decision from "what to do".

**Headline: a greedy one-step oracle does not upper-bound the rule-based
controller — it underperforms it four-fold. The reason is measurable, and it is
itself the finding: the per-step reward signal that distinguishes delegations is
smaller than the environment's own measurement noise, and maintenance payoff is
delayed. This is evidence that the rule-based controller's value lies in
*temporal pacing*, which no one-step optimisation can discover.**

## What was built

Environment snapshot/rollback (option (a) of the plan). At each step the world is
advanced once by ingestion, a snapshot is taken, and each of the three
delegations — NOOP, the compaction specialist's action, the partition
specialist's action — is executed as a trial branch with its **realised** reward
measured by actually running the query. Between branches the Iceberg data
snapshot is rolled back, any partition-spec change undone, and all Python state
restored (simulator state, pruning/query histories, both agents' observation
windows). The best branch is then executed for real.

Rollback was chosen over a model-based lookahead because the reward depends on
measured query latency, which nothing in this codebase predicts; we fitted the
alternative and a latency model reaches only R² = 0.67, so a lookahead oracle
would have bounded the *model* rather than the environment.

### Implementation note: Iceberg metadata retention

The first attempt failed at ~step 50–100 with
`NotFoundException: .../metadata/*.avro`, after which every metrics query
silently returned zeros. The oracle issues 3–4 commits per step, so a 1,000-step
episode produces several thousand commits and Iceberg garbage-collected
manifests behind snapshots that rollback later needed. Four isolated tests
(compaction+rollback unpartitioned; partition-spec churn; the full three-branch
loop; compaction+rollback while partitioned) each ran clean, confirming a
scale effect rather than a logic error. Setting
`write.metadata.delete-after-commit.enabled=false` and
`write.metadata.previous-versions-max=2000` — re-applied after every
`sim.reset()` — fixed it; 160 steps then ran clean with the partition spec
active, and the full 1,000-step episode completed.

An environment-health guard now aborts the episode if `file_count == 0` or
latency ≤ 0, so this class of silent corruption cannot reach the results again.

## Result (1,000-step v5 standard workload, DDQN specialists, seed 1)

| Metric | Oracle | Rule-based DDQN (Phase 1) |
|---|---|---|
| Mean global reward | **0.0546** | **0.2306** |
| Final file count | 328 | ~65 (episode mean) |
| Compactions | 571 (all C32) | 173 (C64 95 / C128 45 / C32 35) |
| Agreement with rule-based delegation | 22.7 % | — |

Delegation choices: oracle 571 compaction / 219 partition / 210 NOOP;
rule-based 891 partition / 61 NOOP / 48 compaction.

## Why the oracle fails — the measurable explanation

| Quantity | Value |
|---|---|
| Median per-step spread between the three branch rewards | **0.0069** |
| Latency-driven reward noise (σ of −0.30·l_norm across steps) | **0.0120** |
| Oracle's gain over always-NOOP | **+0.00016 per step** |
| Correlation(chosen branch's trial reward, realised reward) | **0.156** |

Three things follow:

1. **The selection signal is below the noise floor.** The median difference
   between the best and worst delegation is 0.0069, while query-latency
   variation alone moves the reward by σ = 0.012 (latency p10–p90:
   173–1,630 ms). The oracle is therefore largely selecting the branch that
   happened to draw a fast query, not the branch that is actually better — as
   the 0.156 correlation between trial and realised reward confirms.
2. **One-step reward cannot see delayed payoff.** Maintenance is an investment:
   compaction costs now and pays off over subsequent steps. Greedy one-step
   selection gains +0.00016 per step over doing nothing at all, i.e. essentially
   nothing.
3. **The greedy trajectory degenerates.** Because the specialist selects
   COMPACT_32KB in this state region, and C32 produces file proliferation, the
   oracle compacted 571 times yet still ended at 328 files. Each compaction
   looked locally acceptable while the trajectory drifted into a state where the
   agent kept choosing the worst target — a failure mode the rule-based
   controller's cooldowns prevent by spacing interventions.

## What this says about the rule-based meta-controller

The experiment does not produce the upper bound the reviewer asked for, and we
say so plainly. What it does produce is arguably more useful: **evidence that the
"when to act" decision is not a simple greedy optimisation problem.** The
rule-based controller's cooldowns and urgency thresholds encode temporal
structure — do not compact again for 4 steps; do not re-partition for 20 — that a
myopic optimiser cannot discover, and its 22.7 % agreement with the greedy oracle
shows the two are solving genuinely different problems.

### The practical bound we *can* report

Phase 3's meta-controller grid answers the reviewer's question within the
rule-based family, at no extra compute: varying the constants one at a time,
the best configuration (compaction cooldown 2 instead of 4) scored **0.2200 vs
the default's 0.2033 — +8.2 %**. So the loss attributable to *suboptimal
meta-controller constants* is on the order of 8 %, not the order-of-magnitude
gap a naive reading of the oracle number would suggest.

## Honest limitations

- **Single seed, single protocol.** The 5 × 500-step eval protocol was cancelled
  after the std1000 result made its outcome predictable; it would have cost ~17 h
  to reproduce the same noise-dominated behaviour. Stated as a scope decision,
  not a result.
- **A true upper bound needs multi-step lookahead.** Branching k steps deep costs
  3^k environment executions per step; at the measured 19–24 s/step for k = 1,
  even k = 2 is ~10× that. Left as future work, with the reason quantified.
- **The oracle is greedy by construction**, so "oracle" here means
  "best immediate delegation", not "optimal policy". The paper should use the
  former term.

## Artefacts

| Path | Contents |
|---|---|
| `raw/Oracle_ddqn/seed1/std1000/oracle_trace_ep1.csv` | per step: rule vs oracle delegation, all three branch rewards, chosen action, realised reward, agreement |
| `raw/Oracle_ddqn/seed1/std1000/transitions_ep1.csv` | standard transition log |
| `raw/Oracle_ddqn/seed1/std1000/confusion.json` | rule-based → oracle delegation confusion counts |
| `raw/Oracle_ddqn/seed1/std1000/episode_summary.csv` | episode metrics incl. agreement rate |
