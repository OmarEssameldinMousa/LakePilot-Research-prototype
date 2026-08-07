"""
Revision — local offline trainer for extra architecture/algorithm combinations.

Mirrors `revision/phase1_stats/kaggle_train_phase1.ipynb` (pipeline v3) exactly:
same pools, same reward recomputation, same capped staleness features, same
balanced action sampling, same per-action advantage standardization, same
budgets (compaction 150 epochs / partition 200, batch 64). The only thing this
script varies is the (encoder, objective) pair, so results stay comparable with
the Kaggle-trained checkpoints.

Purpose: decouple encoder from objective. The published "AttentivePPO" is
attention + AWR, while "MLP-PPO" is MLP + PPO-clip, so an AttentivePPO-vs-MLP-PPO
comparison confounds the two. Training attention + PPO-clip isolates the encoder.

Usage (inside the container, with NO evaluation running — query latency is the
reward signal and must not contend for CPU):

    docker exec -w /app/src lakegym-lite python3 revision_train_local.py \
        --arch attentive_ppo --algo ppo_clip --label attentive_ppoclip --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse, glob, json, platform, time
from pathlib import Path

import policies  # noqa: F401  (import first: avoids agents circular import)
import numpy as np
import pandas as pd
import tensorflow as tf

from agents.model_registry import build_model

# ── constants (MUST match the notebook and src/simulation.py) ──
ROWS_MEAN, ROWS_STD = 94.0, 71.0
RATE_MEAN, RATE_STD = 318.0, 218.0
FILE_COUNT_MAX, LATENCY_MAX = 2000.0, 15000.0
COMPACT_COST_MAX, PARTITION_COST_MAX, SKEW_MAX = 1.1, 2.0, 2.0
CW_FILES, CW_UTIL, CW_COST, CW_TARGET = 0.25, 0.20, 0.10, 0.45
PW_PRUNING, PW_DELTA, PW_SKEW, PW_COST = 0.55, 0.25, 0.10, 0.10
GW_LATENCY, GW_FILES, GW_PRUNING, GW_UTIL = 0.30, 0.20, 0.30, 0.20
ACTION_COSTS = {0: 0.0, 1: 0.9, 2: 1.0, 3: 1.1, 4: 2.0, 5: 2.0, 6: 2.0, 7: 0.7}
GAMMA, GAMMA_COMPACT = 0.99, 0.95

PARTITION_SOURCES = ('WorkloadAwareThreshold', 'PartitionExploration', 'Random_Partition')
EXCLUDE = ('eval_', 'MultiAgent', 'CompactOnly', 'PartitionOnly')


def set_seeds(seed):
    import random as _r
    _r.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def target_bonus_vec(a, rows):
    tb = np.zeros(len(a), dtype=np.float32)
    low, mid, high = rows <= 50, (rows > 50) & (rows < 150), rows >= 150
    for act, l, m, h in [(1, 1.0, 0.15, -0.6), (2, 0.25, 1.0, 0.4), (3, -0.5, 0.15, 1.0)]:
        s = a == act
        tb[s & low], tb[s & mid], tb[s & high] = l, m, h
    return tb


def recompute_rewards(d):
    d = d.copy()
    a = d['action'].fillna(0).astype(int).values
    cost = np.vectorize(lambda x: ACTION_COSTS.get(x, 0.0))(a)
    fc = d.get('next_file_count', d.get('file_count')).astype(float).values
    util = d.get('next_block_utilization', d.get('block_utilization')).astype(float).values
    lat = d.get('next_latency_ms', d.get('latency_ms')).astype(float).values
    rows = d.get('rows_ingested', pd.Series(np.zeros(len(d)))).astype(float).values
    skew = d.get('file_size_skew_kb', pd.Series(np.zeros(len(d)))).astype(float).values
    pr_c = d.get('partition_pruning_ratio', pd.Series(np.zeros(len(d)))).astype(float).values
    pr_n = d.get('next_partition_pruning_ratio', pd.Series(pr_c)).astype(float).values

    f_n = np.clip(fc / FILE_COUNT_MAX, 0, 1); u_n = np.clip(util, 0, 1)
    l_n = np.clip(lat / LATENCY_MAX, 0, 1);   s_n = np.clip(skew / SKEW_MAX, 0, 1)
    p_n = np.clip(pr_n, 0, 1);                dp = np.clip(pr_n - pr_c, -1, 1)

    d['compact_reward'] = (-CW_FILES * f_n + CW_UTIL * u_n - CW_COST * cost / COMPACT_COST_MAX
                           + CW_TARGET * target_bonus_vec(a, rows))
    d['partition_reward'] = (PW_PRUNING * p_n + PW_DELTA * dp - PW_SKEW * s_n
                             - PW_COST * cost / PARTITION_COST_MAX)
    d['global_reward'] = (-GW_LATENCY * l_n - GW_FILES * f_n + GW_PRUNING * p_n + GW_UTIL * u_n)
    return d


def compact_features(d):
    f = np.zeros((len(d), 8), dtype=np.float32)
    f[:, 0] = (d['rows_ingested'].values - ROWS_MEAN) / ROWS_STD
    f[:, 1] = (d['ingestion_rate_rows_per_sec'].values - RATE_MEAN) / RATE_STD
    f[:, 2] = np.clip(d['latency_ms'].values / 15000., 0, 1)
    f[:, 3] = np.clip(d['file_count'].values / 2000., 0, 1)
    f[:, 4] = np.clip(d['block_utilization'].values, 0, 1)
    f[:, 5] = np.clip(d['total_size_kb'].values / 50000., 0, 1)
    f[:, 6] = np.clip(d['file_size_skew_kb'].values / 50., 0, 1)
    f[:, 7] = np.exp(-np.minimum(d['steps_since_compact'].values, 10) / 10.)
    return f


def partition_features(d):
    f = np.zeros((len(d), 14), dtype=np.float32)
    f[:, 0] = np.clip(d['latency_ms'].values / 15000., 0, 1)
    f[:, 1] = np.clip(d['file_count'].values / 2000., 0, 1)
    ps = d['partition_strategy'].values.astype(int)
    for i in range(4):
        f[:, 2 + i] = (ps == i).astype(np.float32)
    f[:, 6] = np.clip(d['partition_pruning_ratio'].values, 0, 1)
    f[:, 7] = np.clip(d['avg_pruning_ratio'].values, 0, 1)
    for j, c in enumerate(['query_hist_time_range', 'query_hist_region_filter',
                           'query_hist_sensor_lookup', 'query_hist_type_filter',
                           'query_hist_full_scan']):
        f[:, 8 + j] = d[c].values
    f[:, 13] = np.exp(-np.minimum(d['steps_since_partition_change'].values, 60) / 30.)
    return f


def build_windows(F, ep_ids, window):
    out = np.zeros((len(F), window, F.shape[1]), dtype=np.float32)
    for e in np.unique(ep_ids):
        idx = np.where(ep_ids == e)[0]
        for j, i in enumerate(idx):
            s = max(0, j - window + 1)
            seq = F[idx[s:j + 1]]
            out[i, window - len(seq):, :] = seq
    return out


def balanced_indices(A, n_act):
    per = {k: np.where(A == k)[0] for k in range(n_act)}
    spa = min(len(v) for v in per.values())
    idx = np.concatenate([np.random.choice(per[k], spa, replace=True) for k in range(n_act)])
    np.random.shuffle(idx)
    return idx


def standardize_advantages(values, A, n_act, per_action=True):
    adv = np.asarray(values, dtype=np.float32).copy()
    if per_action:
        for k in range(n_act):
            m = A == k
            if m.sum() > 1:
                adv[m] = (adv[m] - adv[m].mean()) / (adv[m].std() + 1e-8)
    else:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv.astype(np.float32)


def train_ppoclip(model, X, A, ADV, R, n_act, epochs, lr=3e-4, batch=64,
                  clip_ratio=0.2, value_coef=0.5, ent=0.01):
    opt = tf.keras.optimizers.Adam(lr)
    init_probs, _ = model(tf.constant(X), training=False)
    old_logp = tf.math.log(tf.reduce_sum(init_probs * tf.one_hot(A, n_act), axis=-1) + 1e-8).numpy()
    hist = []

    @tf.function
    def step(xb, ab, advb, rb, olb):
        with tf.GradientTape() as tape:
            probs, values = model(xb, training=True)
            values = tf.squeeze(values, -1)
            chosen = tf.reduce_sum(probs * tf.one_hot(ab, n_act), axis=-1)
            ratio = tf.exp(tf.math.log(chosen + 1e-8) - olb)
            actor = -tf.reduce_mean(tf.minimum(
                ratio * advb, tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advb))
            critic = tf.reduce_mean(tf.square(rb - values))
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), -1))
            loss = actor + value_coef * critic - ent * entropy
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for ep in range(epochs):
        order = balanced_indices(A, n_act)
        tot, nb = 0.0, 0
        for s in range(0, len(order), batch):
            bi = order[s:s + batch]
            tot += float(step(tf.constant(X[bi]), tf.constant(A[bi]), tf.constant(ADV[bi]),
                              tf.constant(R[bi]), tf.constant(old_logp[bi]))); nb += 1
        hist.append({'epoch': ep + 1, 'loss': tot / nb})
    return hist


def train_awr(model, X, A, ADV, TARGET, n_act, epochs, lr=3e-4, beta=1.0, batch=64, ent=0.01):
    opt = tf.keras.optimizers.Adam(lr)
    hist = []

    @tf.function
    def step(xb, ab, advb, tb):
        with tf.GradientTape() as tape:
            probs, values = model(xb, training=True)
            values = tf.squeeze(values, -1)
            critic = tf.reduce_mean(tf.square(values - tb))
            w = tf.minimum(tf.exp(advb / beta), 20.0)
            logp = tf.math.log(probs + 1e-8)
            lp = tf.reduce_sum(logp * tf.one_hot(ab, n_act), axis=-1)
            actor = -tf.reduce_mean(w * lp)
            entropy = tf.reduce_mean(-tf.reduce_sum(probs * logp, axis=-1))
            loss = actor + 0.5 * critic - ent * entropy
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for ep in range(epochs):
        order = balanced_indices(A, n_act)
        tot, nb = 0.0, 0
        for s in range(0, len(order), batch):
            bi = order[s:s + batch]
            tot += float(step(tf.constant(X[bi]), tf.constant(A[bi]),
                              tf.constant(ADV[bi]), tf.constant(TARGET[bi]))); nb += 1
        hist.append({'epoch': ep + 1, 'loss': tot / nb})
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='attentive_ppo', choices=['attentive_ppo', 'mlp_ppo', 'ddqn'])
    ap.add_argument('--algo', default='ppo_clip', choices=['ppo_clip', 'awr'])
    ap.add_argument('--label', required=True, help='weight-file label, e.g. attentive_ppoclip')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--out', default='/app/revision/phase1_stats/phase1_weights')
    args = ap.parse_args()

    out = Path(args.out)
    (out / 'weights').mkdir(parents=True, exist_ok=True)
    (out / 'curves').mkdir(parents=True, exist_ok=True)

    # ── data ──
    files = sorted(f for f in glob.glob('/app/results/**/*_ep*.csv', recursive=True)
                   if 'transitions' in Path(f).name and not any(x in f for x in EXCLUDE))
    files += sorted(glob.glob('/app/revision/phase1_stats/recovery_data/**/transitions_ep1.csv',
                              recursive=True))
    files_p = [f for f in files if any(s in f for s in PARTITION_SOURCES)]
    files_c = [f for f in files if f not in files_p]

    def load(pool):
        fr = []
        for i, f in enumerate(pool):
            d = pd.read_csv(f); d['ep'] = i; fr.append(d)
        return pd.concat(fr, ignore_index=True).sort_values(['ep', 'step']).reset_index(drop=True)

    df_c, df_p_all = recompute_rewards(load(files_c)), recompute_rewards(load(files_p))

    ep_c = df_c['ep'].values
    W_c = build_windows(compact_features(df_c), ep_c, 10)
    rew_c = df_c['compact_reward'].values.astype(np.float32)
    G = np.zeros(len(df_c), dtype=np.float32)
    for e in np.unique(ep_c):
        idx = np.where(ep_c == e)[0]
        g = 0.0
        for i in idx[::-1]:
            g = rew_c[i] + GAMMA_COMPACT * g
            G[i] = g
    mc = df_c['action'].isin([0, 1, 2, 3]).values
    Xc, Ac, Gc, Rc = W_c[mc], df_c['action'].values[mc].astype(np.int32), G[mc], rew_c[mc]

    ep_p = df_p_all['ep'].values
    W_p = build_windows(partition_features(df_p_all), ep_p, 20)
    PA = {0: 0, 4: 1, 5: 2, 6: 3, 7: 4}
    mp = df_p_all['action'].isin(PA).values
    dfp = df_p_all[mp].reset_index(drop=True)
    Xp = W_p[mp]
    Ap = np.array([PA[a] for a in dfp['action'].values], dtype=np.int32)

    lat_b = dfp['latency_ms'].values.astype(np.float32)
    lat_a = dfp['next_latency_ms'].values.astype(np.float32)
    li = np.clip((lat_b - lat_a) / (lat_b + 1.0), -1, 1)
    pr_b = dfp['partition_pruning_ratio'].values.astype(np.float32)
    pr_a = dfp['next_partition_pruning_ratio'].values.astype(np.float32)
    pi_ = np.clip(pr_a - pr_b, -1, 1)
    qh = np.column_stack([dfp['query_hist_time_range'], dfp['query_hist_region_filter'],
                          dfp['query_hist_sensor_lookup'], dfp['query_hist_type_filter']]).astype(np.float32)
    dq, ds = np.argmax(qh, axis=-1), np.max(qh, axis=-1)
    ideal = np.zeros(len(Ap), dtype=np.int32)
    ideal[dq == 0], ideal[dq == 1], ideal[dq == 3] = 1, 2, 3
    cps = dfp['partition_strategy'].values.astype(int)
    okm = (cps == ideal) & (ideal > 0)
    al = np.zeros(len(Ap), dtype=np.float32)
    al[okm & (Ap == 0)] = 0.5
    al[okm & (Ap == 4)] = -0.8
    for a in [1, 2, 3]:
        al[okm & (Ap == a) & (Ap != ideal)] = -0.5
    bad = ~okm
    for a in [1, 2, 3]:
        al[bad & (Ap == a) & (ideal == a)] = 1.0
    al[bad & (Ap == 0)] = -0.3
    al[bad & (Ap == 4) & (cps > 0)] = 0.3
    al *= np.clip(ds * 2.5 - 0.5, 0.3, 1.5)
    custom = (0.3 * (0.5 * li + 0.5 * pi_) + 0.5 * al
              + 0.2 * dfp['global_reward'].values.astype(np.float32)).astype(np.float32)

    adv_c = standardize_advantages(Gc, Ac, 4)
    adv_p = standardize_advantages(custom, Ap, 5)
    print(f'compaction {len(Xc)} | partition {len(Xp)}', flush=True)

    EPOCHS = {'compaction': 150, 'partition': 200}
    ENT = {'compaction': 0.01, 'partition': 0.10}

    for seed in args.seeds:
        for spec in ['compaction', 'partition']:
            tag = f'{spec}_{args.label}_seed{seed}'
            wp = out / 'weights' / f'{tag}.weights.h5'
            if wp.exists():
                print(f'skip {tag}', flush=True); continue
            set_seeds(seed)
            t0 = time.time()
            if spec == 'compaction':
                na, nf, ws, X, A, ADV, TARGET = 4, 8, 10, Xc, Ac, adv_c, Gc
            else:
                na, nf, ws, X, A, ADV, TARGET = 5, 14, 20, Xp, Ap, adv_p, custom
            model = build_model(args.arch, na, nf, ws)
            kw = dict(epochs=EPOCHS[spec], ent=ENT[spec])
            hist = (train_ppoclip(model, X, A, ADV, TARGET, na, **kw) if args.algo == 'ppo_clip'
                    else train_awr(model, X, A, ADV, TARGET, na, **kw))
            wall = time.time() - t0
            model.save_weights(str(wp))
            pd.DataFrame(hist).to_csv(out / 'curves' / f'{tag}_curve.csv', index=False)
            json.dump({
                'tag': tag, 'seed': seed, 'arch': args.arch, 'specialist': spec,
                'label': args.label,
                'algorithm': f'{args.algo}_balanced',
                'advantage_treatment': 'per_action_zscore',
                'class_balance': 'balanced_action_sampling',
                'hyperparams': {**kw, 'batch': 64, 'lr': 3e-4,
                                'gamma': GAMMA_COMPACT if spec == 'compaction' else GAMMA,
                                'window_size': ws, 'num_features': nf, 'num_actions': na},
                'normalization': {'rows_mean': ROWS_MEAN, 'rows_std': ROWS_STD,
                                  'rate_mean': RATE_MEAN, 'rate_std': RATE_STD},
                'n_train_samples': int(len(X)),
                'trainable_params': int(sum(np.prod(v.shape) for v in model.trainable_variables)),
                'wall_clock_sec': round(wall, 1), 'final_loss': hist[-1]['loss'],
                'gpu': 'none (local CPU)', 'platform': platform.platform(),
                'tf_version': tf.__version__, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }, open(out / 'weights' / f'{tag}_manifest.json', 'w'), indent=2)
            print(f'  {tag}: {wall:.0f}s final_loss={hist[-1]["loss"]:.4f}', flush=True)

    print('done', flush=True)


if __name__ == '__main__':
    main()
