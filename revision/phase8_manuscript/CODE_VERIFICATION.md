# Phase 8 — code verification for manuscript corrections

Direct answers to the reviewer's implementation questions, with the actual source
lines so the manuscript text can be made to match the code.

## 1. Dueling aggregation

**Implemented formula: Q(s,a) = V(s) + ( A(s,a) − mean_a A(s,a) )** — the standard
Wang et al. (2016) mean-subtraction form. Both implementations agree.

`LakeGymLite/src/agents/model_registry.py` (`DuelingDDQNModel.q_values`, line 193) —
the model used by the application and by all revision experiments:

```python
return v + (a - tf.reduce_mean(a, axis=-1, keepdims=True))
```

`Research_notebook/omar-s-data.ipynb` (cell 16), the original offline trainer:

```python
return val + adv - tf.reduce_mean(adv, axis=-1, keepdims=True), val
```

These are algebraically identical (differing only in parenthesisation). **No
manuscript correction is required for the aggregation formula**; the paper's
stated Q = V + (A − mean(A)) matches the code.

## 2. Double-DQN target

**Implemented as intended: the online network selects the next action, the target
network evaluates it.** `LakeGymLite/src/training/online_training.py`
(`_dqn_step`, lines 503–509):

```python
q_next_online = self.model.q_values(next_states, training=False)
next_actions  = tf.argmax(q_next_online, axis=-1)          # online net selects
q_next_target = self.target_model.q_values(next_states, training=False)
target_q_next = tf.reduce_sum(q_next_target * tf.one_hot(next_actions, n), axis=-1)
targets       = rewards + gamma * target_q_next * (1.0 - dones)   # target net evaluates
```

### ⚠ Important caveat the manuscript must disclose

In the **originally published** code this update operated on the model's
**softmax output**, not on raw Q-values: `DuelingDDQNModel.call()` returns
`(softmax(Q), max(Q))`, and the online update consumed the first return value as
if it were Q. The TD target was therefore regressed against a probability vector
bounded in [0, 1], on the same scale as the rewards — a defect that plausibly
explains the "DDQN adaptation collapse" reported in the paper.

The revision adds an explicit `q_values()` accessor and updates on raw Q-values.
The original behaviour is preserved verbatim as `_dqn_step_legacy_softmax` for
side-by-side comparison in Phase 6.

Note that greedy *action selection* was unaffected, since
`argmax(softmax(Q)) = argmax(Q)`; only the learning update was corrupted. This is
why the frozen DDQN evaluations were not affected by this particular defect
(they were affected by a different one — see `phase1_stats/PIPELINE_HISTORY.md`).

## 3. Table 7 "% gap closed" guard

The reported −3,207 % for DDQN is a division-by-near-zero artifact: the metric
`(final_gap − initial_gap) / |initial_gap|` is meaningless when the initial gap is
close to zero. `revision/stats/gap_metric.py` implements the guarded version —
the percentage is reported only when `|initial_gap|` exceeds a threshold, and
`"n/a"` otherwise.
