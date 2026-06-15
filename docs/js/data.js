/* ============================================================================
   LakePilot — data extracted verbatim from the paper & comprehensive notebook
   "Workload-Aware Data Lakehouse Maintenance Using Hierarchical Deep RL"
   All numbers trace to Tables 1–15 and Figures 1–14.
   ========================================================================== */

const DB = {
  // ── Agent colour identity (matches the notebook design system) ──
  color: {
    attentive: '#185FA5',
    mlp:       '#1D9E75',
    ddqn:      '#E8593C',
    red:       '#FF3621', // Databricks lava
    navy:      '#1B3139',
    amber:     '#EF9F27',
    gray:      '#A8A69C',
    pink:      '#D4537E',
    grid:      '#E3E1DA',
  },

  // ── Table 1 : full policy ranking (top-12 + bottom-2) ──
  ranking: [
    { rank: 1,  policy: 'AttentivePPO',          reward: 0.2201, latency: 323.6, files: 54.9,  pruning: 0.191, kind: 'rl' },
    { rank: 2,  policy: 'DDQN',                   reward: 0.2196, latency: 323.4, files: 58.8,  pruning: 0.197, kind: 'rl' },
    { rank: 3,  policy: 'MLP-PPO',                reward: 0.2124, latency: 387.8, files: 62.7,  pruning: 0.203, kind: 'rl' },
    { rank: 4,  policy: 'WorkloadAwareThreshold', reward: 0.2111, latency: 439.9, files: 93.4,  pruning: 0.285, kind: 'wa' },
    { rank: 5,  policy: 'AlwaysCompact_C128',     reward: 0.1869, latency: 344.3, files: 43.5,  pruning: 0.000, kind: 'heur' },
    { rank: 6,  policy: 'Random_Compact',         reward: 0.1809, latency: 460.8, files: 52.1,  pruning: 0.000, kind: 'heur' },
    { rank: 7,  policy: 'Threshold10_C128',       reward: 0.1734, latency: 429.7, files: 46.8,  pruning: 0.000, kind: 'heur' },
    { rank: 8,  policy: 'Periodic10_C128',        reward: 0.1734, latency: 399.1, files: 47.6,  pruning: 0.000, kind: 'heur' },
    { rank: 9,  policy: 'CompactOnly (ablation)',  reward: 0.1699, latency: 344.1, files: 48.9,  pruning: 0.000, kind: 'abl' },
    { rank: 10, policy: 'PartitionExploration',    reward: 0.1668, latency: 437.2, files: 90.8,  pruning: 0.138, kind: 'wa' },
    { rank: 11, policy: 'AdaptiveCompaction',      reward: 0.1411, latency: 609.5, files: 71.4,  pruning: 0.000, kind: 'heur' },
    { rank: 12, policy: 'AlwaysCompact (C64)',     reward: 0.1393, latency: 554.0, files: 88.0,  pruning: 0.000, kind: 'heur' },
    { rank: 21, policy: 'No_Maintenance',          reward: -0.1492, latency: 4298.1, files: 873.7, pruning: 0.000, kind: 'bad' },
    { rank: 22, policy: 'PartitionOnly (ablation)', reward: -0.1765, latency: 5306.9, files: 1245.0, pruning: 0.110, kind: 'abl' },
  ],

  // ── Table 3 / Fig 5 : reward decomposition ──
  decomposition: [
    { policy: 'No maintenance',          compact: -0.0851, partition:  0.0000, global: -0.1492 },
    { policy: 'Always compact',          compact:  0.3608, partition: -0.0500, global:  0.1393 },
    { policy: 'Adaptive compact',        compact:  0.1417, partition: -0.0428, global:  0.1411 },
    { policy: 'Compact only (abl.)',     compact:  0.1855, partition: -0.0085, global:  0.1699 },
    { policy: 'Partition only (abl.)',   compact: -0.1397, partition: -0.0179, global: -0.1765 },
    { policy: 'AttentivePPO',            compact:  0.1581, partition:  0.0535, global:  0.2201 },
    { policy: 'MLP-PPO',                 compact:  0.1840, partition:  0.0510, global:  0.2124 },
    { policy: 'DDQN',                    compact:  0.1659, partition:  0.0675, global:  0.2196 },
  ],

  // ── Table 2 : three-way RL comparison (mean ± s.d., n = 5000) ──
  threeway: {
    labels: ['AttentivePPO', 'MLP-PPO', 'DDQN'],
    reward:  [0.2201, 0.2124, 0.2196],
    rewardSd:[0.0646, 0.0643, 0.0650],
    latency: [323.6, 387.8, 323.4],
    files:   [54.9, 62.7, 58.8],
    pruning: [0.191, 0.203, 0.197],
  },

  // ── Table 9 : normalised cross-architecture profile (0–1, higher better) ──
  normalised: {
    metrics: ['Global reward', 'Latency (inv.)', 'File count (inv.)', 'Pruning ratio'],
    attentive: [1.000, 0.997, 1.000, 0.000],
    mlp:       [0.000, 0.000, 0.000, 1.000],
    ddqn:      [0.935, 1.000, 0.507, 0.475],
  },

  // ── Table 4 / Fig 6 : compaction target size sensitivity ──
  targetSize: {
    labels: ['32 KB', '64 KB', '128 KB'],
    reward:  [0.0393, 0.1393, 0.1869],
    latency: [1283.0, 554.0, 344.3],
    files:   [179.1, 88.0, 43.5],
  },

  // ── Table 5 : action distribution (% share, frozen agents) ──
  actions: {
    labels: ['NOOP', 'COMPACT_128KB', 'COMPACT_64KB', 'COMPACT_32KB',
             'PARTITION_REGION', 'PARTITION_EVENT_TYPE', 'PARTITION_HOUR', 'REMOVE_PARTITION'],
    attentive: [76.2, 10.8, 4.6, 2.7, 3.4, 0.6, 0.7, 0.9],
    mlp:       [77.6, 1.4, 16.7, 0.1, 0.8, 1.1, 1.1, 1.2],
    ddqn:      [79.3, 5.6, 5.0, 5.9, 1.0, 1.1, 1.1, 0.9],
  },

  // ── Table 7 / Fig 9 : adaptive online learning (mean reward per episode) ──
  adaptive: {
    episodes: ['Ep 1', 'Ep 2', 'Ep 3', 'Ep 4', 'Ep 5'],
    attentive: [0.1934, 0.2017, 0.2226, 0.2149, 0.2326],
    mlp:       [0.1994, 0.2018, 0.1818, 0.1939, 0.2091],
    ddqn:      [0.2181, 0.2133, 0.2298, 0.2047, 0.1696],
    ceiling: { attentive: 0.2201, mlp: 0.2124, ddqn: 0.2196 },
    gain:    { attentive: '+20.3%', mlp: '+4.9%', ddqn: '-22.2%' },
  },

  // ── Table 11 / Fig 11 : commercial cost breakdown ($ per 1000-step episode) ──
  cost: {
    labels: ['No maintenance', 'Always compact', 'Adaptive compact', 'AttentivePPO', 'MLP-PPO', 'DDQN'],
    query:   [42.98, 5.54, 6.10, 3.24, 3.88, 3.23],
    compact: [0.00, 5.00, 4.51, 0.91, 0.91, 0.83],
    storage: [6.88, 4.63, 4.56, 4.45, 4.53, 4.47],
    total:   [49.86, 15.17, 15.17, 8.60, 9.32, 8.53],
    saving:  ['—', '-69.6%', '-69.6%', '-82.8%', '-81.3%', '-82.9%'],
  },

  // ── Table 6 : frozen agent consistency (CV %) ──
  consistency: [
    { metric: 'Global reward', a: '0.2182 (CV 5.86%)', m: '0.1991 (CV 6.99%)', d: '0.2065 (CV 6.38%)' },
    { metric: 'Latency (ms)',  a: '190.3 (CV 2.33%)',  m: '258.4 (CV 3.24%)',  d: '246.2 (CV 6.73%)' },
    { metric: 'File count',    a: '31.8 (CV 2.40%)',   m: '37.2 (CV 2.88%)',   d: '35.2 (CV 3.28%)' },
    { metric: 'Pruning ratio', a: '0.214 (CV 18.26%)', m: '0.191 (CV 24.80%)', d: '0.193 (CV 24.70%)' },
  ],

  // ── Table 14 / Fig 13 : pruning matrix (query type × partition strategy) ──
  pruning: {
    rows: ['TIME_RANGE', 'REGION_FILTER', 'SENSOR_LOOKUP', 'TYPE_FILTER', 'FULL_SCAN'],
    cols: ['HOUR', 'REGION', 'EVENT_TYPE', 'NONE'],
    grid: [
      [0.92, 0,    0,    0],
      [0,    0.67, 0,    0],
      [0.75, 0,    0,    0],
      [0,    0,    0.67, 0],
      [0,    0,    0,    0],
    ],
  },

  // ── Table 12 : action space & costs ──
  actionSpace: [
    { action: 'NOOP',                 type: '—',          cost: 0.0 },
    { action: 'COMPACT_32KB',         type: 'Compaction', cost: 0.9 },
    { action: 'COMPACT_64KB',         type: 'Compaction', cost: 1.0 },
    { action: 'COMPACT_128KB',        type: 'Compaction', cost: 1.1 },
    { action: 'PARTITION_HOUR',       type: 'Partition',  cost: 2.0 },
    { action: 'PARTITION_REGION',     type: 'Partition',  cost: 2.0 },
    { action: 'PARTITION_EVENT_TYPE', type: 'Partition',  cost: 2.0 },
    { action: 'REMOVE_PARTITION',     type: 'Partition',  cost: 0.7 },
  ],

  // ── Table 13 : workload query profiles ──
  profiles: {
    cols: ['Time range', 'Region filter', 'Sensor lookup', 'Type filter', 'Full scan'],
    rows: [
      { name: 'time_heavy',   v: [0.60, 0.10, 0.10, 0.10, 0.10] },
      { name: 'region_heavy', v: [0.10, 0.60, 0.10, 0.10, 0.10] },
      { name: 'mixed',        v: [0.20, 0.20, 0.20, 0.20, 0.20] },
      { name: 'type_heavy',   v: [0.10, 0.10, 0.10, 0.60, 0.10] },
      { name: 'scan_heavy',   v: [0.10, 0.10, 0.10, 0.10, 0.60] },
    ],
  },

  // ── Training feature vectors (from training/data_utils.py) ──
  features: {
    compact: [
      { f: 'rows_ingested',               norm: 'z-score' },
      { f: 'ingestion_rate_rows_per_sec', norm: 'z-score' },
      { f: 'latency_ms',                  norm: '÷ 15,000 → [0,1]' },
      { f: 'file_count',                  norm: '÷ 2,000 → [0,1]' },
      { f: 'block_utilization',           norm: 'clip [0,1]' },
      { f: 'total_size_kb',               norm: '÷ 50,000 → [0,1]' },
      { f: 'file_size_skew_kb',           norm: '÷ 50 → [0,1]' },
      { f: 'steps_since_compact',         norm: 'exp(−x / 10)' },
    ],
    partition: [
      { f: 'latency_ms',                  norm: '÷ 15,000 → [0,1]' },
      { f: 'file_count',                  norm: '÷ 2,000 → [0,1]' },
      { f: 'partition_strategy',          norm: 'one-hot (4 dims)' },
      { f: 'partition_pruning_ratio',     norm: 'already [0,1]' },
      { f: 'avg_pruning_ratio',           norm: 'already [0,1]' },
      { f: 'query_hist_time_range',       norm: 'fraction [0,1]' },
      { f: 'query_hist_region_filter',    norm: 'fraction [0,1]' },
      { f: 'query_hist_sensor_lookup',    norm: 'fraction [0,1]' },
      { f: 'query_hist_type_filter',      norm: 'fraction [0,1]' },
      { f: 'query_hist_full_scan',        norm: 'fraction [0,1]' },
      { f: 'steps_since_partition_change',norm: 'exp decay' },
    ],
  },

  // ── Offline pre-training — PER ARCHITECTURE (each agent trained differently) ──
  //    AttentivePPO is CONFIRMED from its notebooks; the other two are PRELIMINARY
  //    placeholders pending the final training notebooks.
  offlineByArch: {
    attentive: {
      name: 'AttentivePPO', status: 'final',
      method: 'Advantage-Weighted Regression (AWR / AWAC)',
      blurb: 'The policy is regressed toward high-advantage logged actions, weighted by min(exp(adv/β), 20) — strong imitation of good decisions, suppression of poor ones, with no environment interaction.',
      cfg: [
        ['Method', 'Advantage-Weighted Regression'],
        ['Objective', 'min(exp(adv / β), 20) · −log π(a|s)'],
        ['β (temperature)', '1.0'],
        ['Policy LR', '3 × 10⁻⁴'],
        ['Epochs', '200'],
        ['Entropy coef', '0.10'],
        ['Batch size', '64'],
        ['Sampling', 'Balanced / stratified per action'],
      ],
    },
    mlp: {
      name: 'MLP-PPO', status: 'final',
      method: 'Offline clipped-PPO (surrogate objective)',
      blurb: 'Trained on the static logs with the PPO clipped surrogate, using standardized advantages from the logged returns; the reference log-probs are snapshotted from the initial policy. The compaction and partition roles share identical settings.',
      cfg: [
        ['Method', 'Offline clipped-PPO surrogate'],
        ['Objective', 'min(r·Â, clip(r, 1±ε)·Â)'],
        ['Clip ratio ε', '0.2'],
        ['Policy LR', '3 × 10⁻⁴'],
        ['Epochs', '150'],
        ['Value coef', '0.5'],
        ['Entropy coef', '0.01'],
        ['Batch size', '128'],
        ['Grad clip', '0.5'],
      ],
    },
    ddqn: {
      name: 'DDQN', status: 'final',
      method: 'Offline Double-DQN · dueling · replay over logs',
      blurb: 'The logged transitions seed a replay buffer; a dueling Double-DQN regresses Q-values toward Bellman targets — the online network selects the next action, the target network evaluates it.',
      note: 'Confirmed for the partition role. The compaction role is taken to share the same training loop and objective (its dedicated notebook is separate / pending), differing only in the role-specific feature/action dimensions.',
      cfg: [
        ['Method', 'Offline Double-DQN (dueling)'],
        ['Objective', 'MSE( Q − [r + γ·Q⁻(s′, argmaxQ)] )'],
        ['Discount γ', '0.99'],
        ['Learning rate', '1 × 10⁻⁴'],
        ['Epochs', '150'],
        ['Target update', 'every 50 batches'],
        ['Replay buffer', '100,000'],
        ['Batch size', '64'],
        ['Grad clip', '1.0'],
      ],
    },
  },
  // shaped reward blend used for the AttentivePPO partition specialist offline (confirmed)
  offlineReward: [
    ['Outcome', '0.30', 'latency + pruning improvement'],
    ['Alignment', '0.50', 'dominant query → ideal partition match'],
    ['Global', '0.20', 'the v5 global reward'],
  ],

  // ── Online fine-tuning hyper-parameters (from online_training.py) ──
  trainCfg: {
    ppo: [
      ['Algorithm', 'On-policy PPO (clipped)'], ['Discount γ', '0.99'],
      ['GAE λ', '0.95'], ['Clip ratio', '0.2'], ['Actor LR', '3 × 10⁻⁴'],
      ['Value coef', '0.5'], ['Entropy coef', '0.01'], ['Rollout steps', '256'],
      ['Update epochs', '4'], ['Minibatch', '64'], ['Grad clip', '0.5'],
    ],
    dqn: [
      ['Algorithm', 'Off-policy Dueling Double-DQN'], ['Discount γ', '0.99'],
      ['Learning rate', '1 × 10⁻⁴'], ['Replay buffer', '10,000'],
      ['Batch size', '64'], ['Target update', 'every 100 steps'],
      ['ε start → end', '1.0 → 0.05'], ['ε decay', '5,000 steps'],
    ],
  },

  // ── Real degradation series (No_Maintenance vs frozen AttentivePPO, ep1) ──
  //    Extracted from LakeGymLite/results/*/transitions_*.csv, downsampled @25 steps
  problem: {
    steps:    [1,26,51,76,101,126,151,176,201,226,251,276,301,326,351,376,401,426,451,476,501,526,551,576,601,626,651,676,701,726,751,776,801,826,851,876,901,926,951,976],
    nmFiles:  [0,25,50,75,100,125,152,178,204,229,254,279,306,409,507,609,713,808,913,1022,1122,1147,1172,1197,1222,1247,1272,1297,1322,1347,1372,1398,1424,1449,1474,1499,1525,1550,1575,1600],
    nmLat:    [0,248,539,402,574,615,844,703,798,902,1111,2524,1218,1688,1647,2526,4037,4184,6020,4712,5076,10055,8145,10471,5181,5696,4386,9100,8527,8609,4183,7088,7007,5742,6380,5174,4287,9559,10343,11835],
    rlFiles:  [0,15,10,12,11,9,10,15,14,23,20,21,21,48,57,59,76,73,90,77,61,37,41,71,74,77,76,77,94,77,51,86,83,90,83,86,87,104,52,60],
    rlLat:    [0,245,181,127,169,226,184,242,164,174,84,164,115,288,164,191,136,778,487,266,331,475,272,742,809,262,144,144,368,411,140,543,132,1011,333,966,362,1146,318,322],
    burst:    [301, 500],   // ingestion burst window (225 rows/step)
  },

  // ── Ablation : gain of multi-agent vs single-agent (Table 10) ──
  ablation: {
    vsCompact:   { attentive: '+29.5%', mlp: '+25.0%', ddqn: '+29.3%' },
    vsPartition: { attentive: '+224.7%', mlp: '+220.3%', ddqn: '+224.4%' },
  },
};
