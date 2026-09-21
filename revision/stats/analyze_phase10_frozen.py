"""
Phase 10 — does online adaptation help, or was the comparison confounded?

Round-2 reviewer point: "The adaptive means are far above the corresponding
frozen ceilings — MLP-PPO reaches 0.2093 against a frozen 0.1675 — yet the text
concludes adaptation is neutral. The Ep.1->5 column measures drift within the
window and cannot detect a gain that was already present at episode 1."

The reviewer is right that the two columns were not comparable, and the audit
found why: Phase 1 evaluates with a deterministic argmax, while the adaptive
runner explores (PPO agents sample from their softmax; DDQN is epsilon-greedy),
on a different environment-seed block. This script analyses the control arms that
separate the three factors:

  Table 1  (phase1_stats)   argmax,  no updates, env seeds 15xxx
  frozen   (arm A)          sampled, no updates, env seeds 25xxx
  frozen_greedy (arm B)     argmax,  no updates, env seeds 25xxx
  adaptive (phase6)         sampled, updates,    env seeds 25xxx

Arm A vs adaptive isolates learning; arm B vs arm A isolates action selection;
arm B vs Table 1 isolates the seed block.

Usage:  python3 revision/stats/analyze_phase10_frozen.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import bootstrap_ci, cluster_bootstrap_ci, paired_tests  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE = ROOT / 'revision' / 'phase6_ddqn' / 'online_raw'
FROZEN = ROOT / 'revision' / 'phase10_r2' / 'frozen_raw'
PHASE1 = ROOT / 'revision' / 'phase1_stats' / 'raw'
OUT = ROOT / 'revision' / 'phase10_r2'

ARCHS = {'ddqn': 'DDQN', 'mlp_ppo': 'MLP-PPO', 'attentive_ppoclip': 'AttentivePPO-clip'}
PHASE1_LABEL = {'ddqn': 'MultiAgentRL_ddqn', 'mlp_ppo': 'MultiAgentRL_mlp_ppo',
                'attentive_ppoclip': 'MultiAgentRL_attentive_ppoclip'}


def episode_logs(root: Path, label: str) -> pd.DataFrame:
    """Episode logs for one arm, excluding seeds whose run did not finish.

    The runner appends to episode_log.csv as each episode completes, so an
    interrupted seed leaves a short log behind. Including it would silently
    average a 2-episode seed against 5-episode ones; the manifest is the
    authority on whether a seed is complete.
    """
    rows = []
    for f in sorted(glob.glob(str(root / label / 'seed*' / 'episode_log.csv'))):
        seed_dir = Path(f).parent
        manifest = seed_dir / 'manifest.json'
        if not manifest.exists():
            continue
        if json.loads(manifest.read_text()).get('status') != 'done':
            print(f'  skipping {label}/{seed_dir.name}: run incomplete')
            continue
        d = pd.read_csv(f)
        d['seed'] = int(seed_dir.name[4:])
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def phase1_episodes(arch: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(PHASE1 / PHASE1_LABEL[arch] / 'seed*' / 'eval500' /
                                  'episode_summary.csv'))):
        d = pd.read_csv(f)
        d['seed'] = int(Path(f).parts[-3][4:])
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def describe(name: str, d: pd.DataFrame, col: str = 'avg_global_reward') -> dict:
    if d.empty:
        return {'arm': name, 'n_episodes': 0}
    m, lo, hi = bootstrap_ci(d[col].values)
    rec = {'arm': name, 'n_seeds': d['seed'].nunique(), 'n_episodes': len(d),
           'mean': round(m, 4), 'ci': f'[{lo:.4f}, {hi:.4f}]'}
    if d['seed'].nunique() >= 2 and len(d) > d['seed'].nunique():
        _, clo, chi = cluster_bootstrap_ci(d[col].values, d['seed'].values)
        rec['cluster_ci'] = f'[{clo:.4f}, {chi:.4f}]'
    per_seed = d.groupby('seed')[col].mean()
    rec['seed_sd'] = round(per_seed.std(ddof=1), 4)
    if 'episode' in d:
        ep = d.groupby('episode')[col].mean()
        rec['ep1'] = round(ep.get(1, float('nan')), 4)
        rec['ep5'] = round(ep.get(5, float('nan')), 4)
    return rec


def main() -> None:
    all_rows, contrasts = [], []
    for arch, disp in ARCHS.items():
        adaptive = episode_logs(ADAPTIVE, f'adaptive_{arch}')
        frozen = episode_logs(FROZEN, f'frozen_{arch}')
        greedy = episode_logs(FROZEN, f'frozen_greedy_{arch}')
        p1 = phase1_episodes(arch)

        for name, d in [('Table 1 (argmax, 15xxx)', p1),
                        ('frozen arm B (argmax, 25xxx)', greedy),
                        ('frozen arm A (sampled, 25xxx)', frozen),
                        ('adaptive (sampled + updates)', adaptive)]:
            r = describe(name, d)
            r['architecture'] = disp
            all_rows.append(r)

        # paired contrasts where the env seeds line up (arms share 25xxx)
        def paired(a: pd.DataFrame, b: pd.DataFrame, na: str, nb: str):
            if a.empty or b.empty:
                return
            ka = a.set_index(['seed', 'episode'])['avg_global_reward']
            kb = b.set_index(['seed', 'episode'])['avg_global_reward']
            sh = sorted(set(ka.index) & set(kb.index))
            if len(sh) < 3:
                return
            r = paired_tests(ka.loc[sh].values, kb.loc[sh].values, na, nb)
            contrasts.append({'architecture': disp, 'contrast': f'{na} - {nb}',
                              'n': len(sh), 'diff': round(r.mean_diff, 4),
                              'perm_p': round(r.permutation_p, 4),
                              'cliffs_delta': round(r.cliffs_delta, 3),
                              'isolates': {'adaptive - frozen arm A': 'learning',
                                           'frozen arm A - frozen arm B': 'action selection'}
                                          .get(f'{na} - {nb}', '')})
        paired(adaptive, frozen, 'adaptive', 'frozen arm A')
        paired(frozen, greedy, 'frozen arm A', 'frozen arm B')

    summary = pd.DataFrame(all_rows)
    cols = [c for c in ['architecture', 'arm', 'n_seeds', 'n_episodes', 'mean', 'ci',
                        'cluster_ci', 'seed_sd', 'ep1', 'ep5'] if c in summary]
    print(summary[cols].to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / 'frozen_control_summary.csv', index=False)

    if contrasts:
        c = pd.DataFrame(contrasts)
        print('\npaired contrasts (same env seeds, same checkpoints):')
        print(c.to_string(index=False))
        c.to_csv(OUT / 'frozen_control_contrasts.csv', index=False)
    print(f'\nwrote {OUT}/frozen_control_summary.csv')


if __name__ == '__main__':
    main()
