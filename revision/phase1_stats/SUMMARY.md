# Phase 1 — multi-seed statistical validation (SUMMARY)

Episodes are matched across policies by environment seed, so every
comparison below is paired. CIs are percentile bootstrap (10,000 resamples);
p-values are paired permutation tests, Holm–Bonferroni corrected within each
(protocol, metric) family. Effect sizes are Cliff's delta.

## std1000

| policy | reward (95% CI) | latency ms | files | pruning | seeds |
|---|---|---|---|---|---|
| MultiAgentRL_ddqn | 0.2306 [0.2110, 0.2463] | 344 | 64.7 | 0.235 | 5 |
| WorkloadAwareThreshold | 0.2139 [0.2079, 0.2219] | 573 | 91.5 | 0.298 | 5 |
| MultiAgentRL_attentive_ppoclip | 0.1923 [0.1564, 0.2232] | 498 | 89.7 | 0.241 | 5 |
| AlwaysCompact_C128 | 0.1870 [0.1866, 0.1873] | 348 | 43.5 | 0.000 | 5 |
| Threshold10_C128 | 0.1759 [0.1756, 0.1762] | 331 | 46.3 | 0.000 | 5 |
| MultiAgentRL_mlp_ppo | 0.1731 [0.1508, 0.1971] | 637 | 127.7 | 0.251 | 5 |
| Threshold10_C64 | 0.1284 [0.1276, 0.1291] | 571 | 88.4 | 0.000 | 5 |
| CompactOnlyRL_attentive_ppoclip | 0.1190 [0.0836, 0.1505] | 548 | 94.0 | 0.000 | 5 |
| MultiAgentRL_attentive_ppo | -0.0940 [-0.0940, -0.0940] | 4686 | 1004.7 | 0.258 | 1 |

## eval500

| policy | reward (95% CI) | latency ms | files | pruning | seeds |
|---|---|---|---|---|---|
| MultiAgentRL_ddqn | 0.2105 [0.2050, 0.2155] | 220 | 37.4 | 0.204 | 5 |
| WorkloadAwareThreshold | 0.1943 [0.1923, 0.1964] | 324 | 49.9 | 0.248 | 5 |
| AlwaysCompact_C128 | 0.1903 [0.1901, 0.1904] | 198 | 22.1 | 0.000 | 5 |
| MultiAgentRL_attentive_ppoclip | 0.1823 [0.1669, 0.1963] | 331 | 51.6 | 0.225 | 5 |
| Threshold10_C128 | 0.1710 [0.1708, 0.1712] | 203 | 26.1 | 0.000 | 5 |
| MultiAgentRL_mlp_ppo | 0.1675 [0.1579, 0.1772] | 317 | 61.6 | 0.221 | 5 |
| Threshold10_C64 | 0.1273 [0.1267, 0.1280] | 321 | 46.8 | 0.000 | 5 |
| CompactOnlyRL_attentive_ppoclip | 0.1202 [0.1086, 0.1315] | 314 | 48.4 | 0.000 | 5 |
| PartitionOnlyRL_attentive_ppoclip | -0.1099 [-0.1195, -0.1011] | 3642 | 793.0 | 0.080 | 3 |

## Answers to the reviewer questions

_Reference protocol: eval500._

**(a) Is RL significantly better than the best heuristic on reward?**

Best heuristic: `WorkloadAwareThreshold` = 0.1943 [0.1923, 0.1964]  
Best RL: `MultiAgentRL_ddqn` = 0.2105 [0.2050, 0.2155]

Paired test `MultiAgentRL_ddqn` vs `WorkloadAwareThreshold`: difference +0.0162, Holm-corrected permutation p = 0.0018, Wilcoxon p = 0.0002865, Cliff's delta = +0.71 (large) → **SIGNIFICANT**.

**(b) Are the RL architectures distinguishable from each other?**

- **Global reward**: 9/10 RL-vs-RL pairs significant after correction.
  - ✓ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_attentive_ppoclip: -0.0621, p=0.0018, delta=-0.76 (large)
  - ✓ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_ddqn: -0.0903, p=0.0018, delta=-1.00 (large)
  - ✓ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_mlp_ppo: -0.0473, p=0.0018, delta=-0.75 (large)
  - ✓ CompactOnlyRL_attentive_ppoclip vs PartitionOnlyRL_attentive_ppoclip: +0.2294, p=0.0018, delta=+1.00 (large)
  - ✓ MultiAgentRL_attentive_ppoclip vs MultiAgentRL_ddqn: -0.0282, p=0.002, delta=-0.44 (medium)
  - ✗ MultiAgentRL_attentive_ppoclip vs MultiAgentRL_mlp_ppo: +0.0148, p=0.3729, delta=+0.33 (medium)
  - ✓ MultiAgentRL_attentive_ppoclip vs PartitionOnlyRL_attentive_ppoclip: +0.2990, p=0.0018, delta=+1.00 (large)
  - ✓ MultiAgentRL_ddqn vs MultiAgentRL_mlp_ppo: +0.0430, p=0.0018, delta=+0.84 (large)
  - ✓ MultiAgentRL_ddqn vs PartitionOnlyRL_attentive_ppoclip: +0.3199, p=0.0018, delta=+1.00 (large)
  - ✓ MultiAgentRL_mlp_ppo vs PartitionOnlyRL_attentive_ppoclip: +0.2739, p=0.0018, delta=+1.00 (large)
- **Latency (ms)**: 7/10 RL-vs-RL pairs significant after correction.
  - ✗ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_attentive_ppoclip: -16.8226, p=1, delta=-0.09 (negligible)
  - ✓ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_ddqn: +94.6285, p=0.00195, delta=+0.58 (large)
  - ✗ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_mlp_ppo: -2.6985, p=1, delta=-0.05 (negligible)
  - ✓ CompactOnlyRL_attentive_ppoclip vs PartitionOnlyRL_attentive_ppoclip: -3345.2457, p=0.0018, delta=-1.00 (large)
  - ✓ MultiAgentRL_attentive_ppoclip vs MultiAgentRL_ddqn: +111.4512, p=0.0018, delta=+0.66 (large)
  - ✗ MultiAgentRL_attentive_ppoclip vs MultiAgentRL_mlp_ppo: +14.1242, p=1, delta=-0.06 (negligible)
  - ✓ MultiAgentRL_attentive_ppoclip vs PartitionOnlyRL_attentive_ppoclip: -3297.6831, p=0.0018, delta=-1.00 (large)
  - ✓ MultiAgentRL_ddqn vs MultiAgentRL_mlp_ppo: -97.3270, p=0.0018, delta=-0.93 (large)
  - ✓ MultiAgentRL_ddqn vs PartitionOnlyRL_attentive_ppoclip: -3427.9931, p=0.0018, delta=-1.00 (large)
  - ✓ MultiAgentRL_mlp_ppo vs PartitionOnlyRL_attentive_ppoclip: -3325.1685, p=0.0018, delta=-1.00 (large)
- **File count**: 8/10 RL-vs-RL pairs significant after correction.
  - ✗ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_attentive_ppoclip: -3.1409, p=0.1297, delta=-0.21 (small)
  - ✓ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_ddqn: +11.0551, p=0.01995, delta=+0.20 (small)
  - ✓ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_mlp_ppo: -13.1354, p=0.002, delta=-0.42 (medium)
  - ✓ CompactOnlyRL_attentive_ppoclip vs PartitionOnlyRL_attentive_ppoclip: -744.5765, p=0.0018, delta=-1.00 (large)
  - ✓ MultiAgentRL_attentive_ppoclip vs MultiAgentRL_ddqn: +14.1960, p=0.0048, delta=+0.23 (small)
  - ✗ MultiAgentRL_attentive_ppoclip vs MultiAgentRL_mlp_ppo: -9.9945, p=0.0792, delta=-0.31 (small)
  - ✓ MultiAgentRL_attentive_ppoclip vs PartitionOnlyRL_attentive_ppoclip: -740.6548, p=0.0018, delta=-1.00 (large)
  - ✓ MultiAgentRL_ddqn vs MultiAgentRL_mlp_ppo: -24.1905, p=0.0018, delta=-0.89 (large)
  - ✓ MultiAgentRL_ddqn vs PartitionOnlyRL_attentive_ppoclip: -755.1689, p=0.0018, delta=-1.00 (large)
  - ✓ MultiAgentRL_mlp_ppo vs PartitionOnlyRL_attentive_ppoclip: -730.7340, p=0.0018, delta=-1.00 (large)
- **Pruning ratio**: 7/10 RL-vs-RL pairs significant after correction.
  - ✓ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_attentive_ppoclip: -0.2245, p=0.0018, delta=-1.00 (large)
  - ✓ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_ddqn: -0.2044, p=0.0018, delta=-1.00 (large)
  - ✓ CompactOnlyRL_attentive_ppoclip vs MultiAgentRL_mlp_ppo: -0.2208, p=0.0018, delta=-1.00 (large)
  - ✓ CompactOnlyRL_attentive_ppoclip vs PartitionOnlyRL_attentive_ppoclip: -0.0802, p=0.0018, delta=-1.00 (large)
  - ✗ MultiAgentRL_attentive_ppoclip vs MultiAgentRL_ddqn: +0.0201, p=0.4896, delta=+0.37 (medium)
  - ✗ MultiAgentRL_attentive_ppoclip vs MultiAgentRL_mlp_ppo: +0.0037, p=1, delta=+0.16 (small)
  - ✓ MultiAgentRL_attentive_ppoclip vs PartitionOnlyRL_attentive_ppoclip: +0.1753, p=0.0018, delta=+1.00 (large)
  - ✗ MultiAgentRL_ddqn vs MultiAgentRL_mlp_ppo: -0.0164, p=0.3385, delta=-0.16 (small)
  - ✓ MultiAgentRL_ddqn vs PartitionOnlyRL_attentive_ppoclip: +0.1226, p=0.0018, delta=+1.00 (large)
  - ✓ MultiAgentRL_mlp_ppo vs PartitionOnlyRL_attentive_ppoclip: +0.1414, p=0.0018, delta=+1.00 (large)

## Seed-level stability

Per-seed spread of mean global reward (large spread = unstable training):

| policy | protocol | min | max | sd |
|---|---|---|---|---|
| CompactOnlyRL_attentive_ppoclip | std1000 | 0.0573 | 0.1677 | 0.0428 |
| MultiAgentRL_attentive_ppoclip | std1000 | 0.1347 | 0.2378 | 0.0421 |
| MultiAgentRL_attentive_ppoclip | eval500 | 0.1242 | 0.2200 | 0.0403 |
| CompactOnlyRL_attentive_ppoclip | eval500 | 0.0836 | 0.1591 | 0.0323 |
| MultiAgentRL_mlp_ppo | std1000 | 0.1357 | 0.2172 | 0.0294 |
| MultiAgentRL_mlp_ppo | eval500 | 0.1430 | 0.2023 | 0.0254 |
| MultiAgentRL_ddqn | std1000 | 0.1920 | 0.2551 | 0.0238 |
| PartitionOnlyRL_attentive_ppoclip | eval500 | -0.1232 | -0.1020 | 0.0116 |
| MultiAgentRL_ddqn | eval500 | 0.1981 | 0.2244 | 0.0104 |
| WorkloadAwareThreshold | std1000 | 0.2050 | 0.2286 | 0.0090 |
| WorkloadAwareThreshold | eval500 | 0.1915 | 0.1964 | 0.0020 |
| Threshold10_C64 | eval500 | 0.1261 | 0.1288 | 0.0011 |
| Threshold10_C64 | std1000 | 0.1271 | 0.1296 | 0.0010 |
| AlwaysCompact_C128 | std1000 | 0.1862 | 0.1875 | 0.0005 |
| Threshold10_C128 | std1000 | 0.1752 | 0.1763 | 0.0004 |
| Threshold10_C128 | eval500 | 0.1707 | 0.1713 | 0.0003 |
| AlwaysCompact_C128 | eval500 | 0.1900 | 0.1905 | 0.0002 |
| MultiAgentRL_attentive_ppo | std1000 | -0.0940 | -0.0940 | 0.0000 |
