"""
Phase 9 — how much of measured query latency is fixed overhead?

Round-2 reviewer point: at 4-5 MB, query latency is likely dominated by Spark job
submission and planning rather than scan work, which would make the latency term
of the global reward largely constant and shift the effective objective onto the
file-count and pruning terms.

Consumes revision/phase9_latency/query_latency_vs_files.csv, written by
LakeGymLite/src/revision_bench_query_latency.py (identical data at every file
count; only the number of files differs).

Fits latency = intercept + slope * files by ordinary least squares over the
sweep, and expresses the environment's own operating points (Table 1 latencies)
as a fixed part and a per-file part.

Usage:  python3 revision/stats/analyze_phase9_latency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'revision' / 'phase9_latency'

# Mean file count and mean measured latency per policy, eval500 (Table 1).
OPERATING_POINTS = {
    'DDQN': (37.4, 219.6),
    'WorkloadAwareThreshold': (49.9, 323.6),
    'AlwaysCompact_C128': (22.1, 197.9),
    'AttentivePPO-clip': (51.6, 331.1),
    'MLP-PPO': (61.6, 316.9),
    'PartitionOnly (ablation)': (793.0, 3642.4),
}


def main() -> None:
    f = RAW / 'query_latency_vs_files.csv'
    if not f.exists():
        sys.exit(f'missing {f} — run revision_bench_query_latency.py first')
    d = pd.read_csv(f)
    man = json.loads((RAW / 'manifest.json').read_text())

    sweep = d[d.regime == 'sweep']
    per_file = sweep.groupby('files')['median_ms'].mean()
    x = per_file.index.values.astype(float)
    y = per_file.values
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]

    print(f'data held constant at {man["staged_rows"]:,} rows '
          f'({man["staged_bytes"] / 1024:.0f} KB consolidated)')
    print(f'file counts measured: {sorted(per_file.index)}')
    print(f'\nSELECT 1 (no table access)        : {man["select1_median_ms"]:.1f} ms')
    single = d[d.regime == 'single_row']['median_ms']
    print(f'single-row table, mean over queries: {single.mean():.1f} ms')
    print(f'\nmean latency across the five query templates:')
    print(per_file.round(1).to_string())
    print(f'\nOLS fit over the sweep: latency = {intercept:.1f} ms + {slope:.3f} ms/file '
          f'(r = {r:.3f})')

    # The sweep holds 1.7 MB constant, so its bytes-per-file differ from the
    # policies' tables (~4.4 MB episode mean). The slope therefore cannot be
    # extrapolated to a policy's operating point as if it were the same layout;
    # what transfers is (a) the measured floor, which is a property of the engine
    # and not of the data, and (b) the shape of the dependence on file count.
    rows = []
    for name, (files, measured) in OPERATING_POINTS.items():
        in_range = x.min() <= files <= x.max()
        rows.append({'policy': name, 'mean_files': files,
                     'measured_latency_ms': measured,
                     'floor_select1_ms': round(man['select1_median_ms'], 1),
                     'floor_single_row_ms': round(single.mean(), 1),
                     'floor_share_of_measured_pct': round(100 * single.mean() / measured, 1),
                     'file_count_within_sweep_range': in_range})
    out = pd.DataFrame(rows)
    print('\nthe measured floor as a share of each policy\'s measured latency')
    print('(floor = a query against a single-row table: catalog resolution,')
    print(' planning, job submission, one file opened — no scan work):')
    print(out.to_string(index=False))

    print('\ninterpretation:')
    print(f'  • a hard floor of {man["select1_median_ms"]:.0f}-{single.mean():.0f} ms exists '
          f'that no maintenance policy can remove')
    print(f'  • above it, latency is almost perfectly affine in file count '
          f'(r = {r:.3f}), so at this scale the latency term is not constant --')
    print(f'    it is very nearly a second measurement of the file-count term')

    out.to_csv(RAW / 'latency_decomposition.csv', index=False)
    summary = {'intercept_ms': float(intercept), 'slope_ms_per_file': float(slope),
               'pearson_r': float(r),
               'select1_median_ms': man['select1_median_ms'],
               'single_row_mean_ms': float(single.mean()),
               'sweep_file_counts': [int(v) for v in x],
               'sweep_mean_latency_ms': [round(float(v), 1) for v in y],
               'staged_rows': man['staged_rows'], 'staged_bytes': man['staged_bytes'],
               'caveat': ('slope is measured at 1.7 MB spread over 14-448 files; '
                          'bytes-per-file differs from the policies operating points, '
                          'so the slope is not extrapolated to them')}
    (RAW / 'latency_fit.json').write_text(json.dumps(summary, indent=2))
    print(f'\nwrote {RAW}/latency_decomposition.csv and latency_fit.json')


if __name__ == '__main__':
    main()
