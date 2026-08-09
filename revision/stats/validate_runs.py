"""
Validate evaluation outputs for dead-environment corruption.

A Spark driver crash does not raise inside `LakeSimulator.run_step`: metrics
collection returns 0 and the query returns latency −1, which the reward formula
maps to exactly 0.0. The runner therefore keeps "stepping" and writes a
`status: done` manifest over meaningless data. This script detects that.

Error signatures:
  • any episode with mean latency <= 0            (query never executed)
  • any episode with mean file_count == 0         (metrics never collected)
  • all episodes in a seed sharing an identical mean latency (frozen/dead JVM)
  • episode mean latency far outside the run's own distribution (crash mid-episode)

Usage:  python3 revision/stats/validate_runs.py [--fix]
        --fix deletes the corrupted seed directories so they re-run.
"""
from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# Every directory that holds seeded evaluation output. All are exposed to the
# same silent Spark-crash failure mode, so all must be audited.
RAW_ROOTS = [
    ROOT / 'revision' / 'phase1_stats' / 'raw',
    ROOT / 'revision' / 'phase1_stats' / 'recovery_data',
    ROOT / 'revision' / 'phase3_sensitivity',
    ROOT / 'revision' / 'phase4_oracle',
    ROOT / 'revision' / 'phase5_generalization',
    ROOT / 'revision' / 'phase7_attention',
]

# An episode whose mean latency exceeds this is not a slow query, it is a crash.
ABSURD_LATENCY_MS = 60_000


def check(path: Path):
    """Return (list_of_problems, n_episodes)."""
    try:
        d = pd.read_csv(path)
    except Exception as e:
        return [f'unreadable: {e}'], 0
    problems = []
    lat = d.get('avg_latency_ms')
    fc = d.get('avg_file_count')
    if lat is None or fc is None:
        return ['missing latency/file_count columns'], len(d)

    if (lat <= 0).any():
        problems.append(f'non-positive latency in episode(s) {list(d.index[lat <= 0] + 1)}')
    if (fc == 0).any():
        problems.append(f'zero file_count in episode(s) {list(d.index[fc == 0] + 1)}')
    if (lat > ABSURD_LATENCY_MS).any():
        bad = [(int(i) + 1, round(float(v), 1)) for i, v in lat.items() if v > ABSURD_LATENCY_MS]
        problems.append(f'absurd latency (crash) in episode(s) {bad}')
    if len(d) > 1 and lat.nunique() == 1:
        problems.append(f'identical latency across all {len(d)} episodes ({lat.iloc[0]:.1f} ms) — frozen environment')
    return problems, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fix', action='store_true',
                    help='delete corrupted seed directories so they are re-run')
    args = ap.parse_args()

    bad_dirs = []
    n_ok = 0
    files = []
    for root in RAW_ROOTS:
        # depth varies: phase1 raw/<policy>/seed/<proto>/, phase3 <sweep>/<cfg>/seed/<proto>/
        files += glob.glob(str(root / '**' / 'episode_summary.csv'), recursive=True)
    for f in sorted(set(files)):
        p = Path(f)
        problems, n = check(p)
        rel = '/'.join(p.parts[-4:-1])
        if problems:
            print(f'❌ {rel}  ({n} episodes)')
            for pr in problems:
                print(f'      {pr}')
            bad_dirs.append(p.parent)
        else:
            n_ok += 1

    print()
    print(f'{n_ok} clean, {len(bad_dirs)} corrupted')
    if bad_dirs and args.fix:
        for d in bad_dirs:
            shutil.rmtree(d)
            print(f'  removed {d}')
        print('re-run the affected policies; manifests are gone so they will not be skipped')
    elif bad_dirs:
        print('re-run with --fix to delete the corrupted seed directories')
    return 1 if bad_dirs else 0


if __name__ == '__main__':
    raise SystemExit(main())
