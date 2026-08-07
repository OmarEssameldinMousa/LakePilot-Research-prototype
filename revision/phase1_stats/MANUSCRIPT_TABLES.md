# Rewritten manuscript tables (Phase 1)

Means are over independent seeds; intervals are percentile bootstrap 95% CIs
(10,000 resamples) over episodes. "Seed sd" is the standard deviation of
per-seed means — i.e. training-run variability, the quantity the published
tables did not report.

## Table 1 (rewritten) — 1,000-step v5 standard workload

| Rank | Policy | Reward (95% CI) | Seed sd | Latency (ms) | Files | Pruning | Seeds | Published |
|---|---|---|---|---|---|---|---|---|
| 1 | DDQN (hierarchical RL) | 0.2306 [0.2110, 0.2463] | 0.0238 | 343.9 | 64.7 | 0.235 | 5 | 0.2196 |
| 2 | WorkloadAwareThreshold | 0.2139 [0.2079, 0.2219] | 0.0090 | 572.5 | 91.5 | 0.298 | 5 | 0.2111 |
| 3 | AttentivePPO-clip (hierarchical RL) | 0.1923 [0.1564, 0.2232] | 0.0421 | 498.1 | 89.7 | 0.241 | 5 | 0.2201 |
| 4 | AlwaysCompact_C128 | 0.1870 [0.1866, 0.1873] | 0.0005 | 348.1 | 43.5 | 0.000 | 5 | 0.1869 |
| 5 | Threshold10_C128 | 0.1759 [0.1756, 0.1762] | 0.0004 | 330.7 | 46.3 | 0.000 | 5 | 0.1734 |
| 6 | MLP-PPO (hierarchical RL) | 0.1731 [0.1508, 0.1971] | 0.0294 | 637.2 | 127.7 | 0.251 | 5 | 0.2124 |
| 7 | Threshold10_C64 | 0.1284 [0.1276, 0.1291] | 0.0010 | 570.8 | 88.4 | 0.000 | 5 | — |
| 8 | CompactOnly (ablation) | 0.1190 [0.0836, 0.1505] | 0.0428 | 548.0 | 94.0 | 0.000 | 5 | 0.1699 |
| 9 | AttentivePPO-AWR (published recipe) | -0.0940 [-0.0940, -0.0940] | — | 4686.4 | 1004.7 | 0.258 | 1 | 0.2201 |

## Table 2 (rewritten) — 1,000-step v5 standard workload

| Agent | Reward (95% CI) | Latency ms (95% CI) | Files (95% CI) | Pruning (95% CI) | Seeds |
|---|---|---|---|---|---|
| DDQN (hierarchical RL) | 0.2306 [0.2110, 0.2463] | 343.9 [321.4, 370.9] | 64.7 [63.5, 66.1] | 0.235 [0.1723, 0.2816] | 5 |
| AttentivePPO-clip (hierarchical RL) | 0.1923 [0.1564, 0.2232] | 498.1 [373.1, 623.1] | 89.7 [58.8, 120.6] | 0.241 [0.2029, 0.2735] | 5 |
| MLP-PPO (hierarchical RL) | 0.1731 [0.1508, 0.1971] | 637.2 [543.3, 738.4] | 127.7 [104.3, 149.4] | 0.251 [0.2135, 0.2815] | 5 |

The published Table 2 reported step-level mean ± s.d. pooled over 5,000
steps. That s.d. describes step-to-step variation within episodes and
treats autocorrelated steps as independent samples; it is not an
uncertainty estimate for the mean. The intervals above are computed over
independent evaluation episodes, and the seed sd column reports
across-training-run variability.

## Table 1 (rewritten) — 500-step compressed evaluation workload

| Rank | Policy | Reward (95% CI) | Seed sd | Latency (ms) | Files | Pruning | Seeds | Published |
|---|---|---|---|---|---|---|---|---|
| 1 | DDQN (hierarchical RL) | 0.2105 [0.2050, 0.2155] | 0.0104 | 219.6 | 37.4 | 0.204 | 5 | 0.2196 |
| 2 | WorkloadAwareThreshold | 0.1943 [0.1923, 0.1964] | 0.0020 | 323.6 | 49.9 | 0.248 | 5 | 0.2111 |
| 3 | AlwaysCompact_C128 | 0.1903 [0.1901, 0.1904] | 0.0002 | 197.9 | 22.1 | 0.000 | 5 | 0.1869 |
| 4 | AttentivePPO-clip (hierarchical RL) | 0.1823 [0.1669, 0.1963] | 0.0403 | 331.1 | 51.6 | 0.225 | 5 | 0.2201 |
| 5 | Threshold10_C128 | 0.1710 [0.1708, 0.1712] | 0.0003 | 203.3 | 26.1 | 0.000 | 5 | 0.1734 |
| 6 | MLP-PPO (hierarchical RL) | 0.1675 [0.1579, 0.1772] | 0.0254 | 316.9 | 61.6 | 0.221 | 5 | 0.2124 |
| 7 | Threshold10_C64 | 0.1273 [0.1267, 0.1280] | 0.0011 | 321.3 | 46.8 | 0.000 | 5 | — |
| 8 | CompactOnly (ablation) | 0.1202 [0.1086, 0.1315] | 0.0323 | 314.2 | 48.4 | 0.000 | 5 | 0.1699 |
| 9 | PartitionOnly (ablation) | -0.1099 [-0.1195, -0.1011] | 0.0116 | 3642.4 | 793.0 | 0.080 | 3 | -0.1765 |

## Table 2 (rewritten) — 500-step compressed evaluation workload

| Agent | Reward (95% CI) | Latency ms (95% CI) | Files (95% CI) | Pruning (95% CI) | Seeds |
|---|---|---|---|---|---|
| DDQN (hierarchical RL) | 0.2105 [0.2050, 0.2155] | 219.6 [212.2, 227.1] | 37.4 [36.6, 38.1] | 0.204 [0.1886, 0.2188] | 5 |
| AttentivePPO-clip (hierarchical RL) | 0.1823 [0.1669, 0.1963] | 331.1 [287.7, 379.6] | 51.6 [44.8, 58.9] | 0.225 [0.2014, 0.2453] | 5 |
| MLP-PPO (hierarchical RL) | 0.1675 [0.1579, 0.1772] | 316.9 [297.1, 338.0] | 61.6 [56.6, 66.8] | 0.221 [0.2031, 0.2389] | 5 |

The published Table 2 reported step-level mean ± s.d. pooled over 5,000
steps. That s.d. describes step-to-step variation within episodes and
treats autocorrelated steps as independent samples; it is not an
uncertainty estimate for the mean. The intervals above are computed over
independent evaluation episodes, and the seed sd column reports
across-training-run variability.
