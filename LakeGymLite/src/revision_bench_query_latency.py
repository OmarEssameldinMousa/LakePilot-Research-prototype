#!/usr/bin/env python3
"""
Query-latency decomposition: fixed overhead versus scan cost (round-2 reviewer request).

Reviewer 2 (round 2): "At 4-5 MB with a 64 KB target, query latency of 220-330 ms
is likely dominated by Spark job submission and planning rather than scan work,
which would make the latency term of the global reward largely constant and shift
the effective objective onto the file-count and pruning terms. Reporting the fixed
overhead (e.g., latency of a query against a single-file table) would let readers
judge how much of the measured signal is scan cost."

Design: hold the DATA EXACTLY CONSTANT and vary only the number of files it is
stored in. The table is materialised once, then rewritten at each file count with
`repartition(n)`, so every measurement scans identical rows and differs only in
per-file overhead. Each of the five query templates used by the environment is
timed with the same `clearCache()` discipline as `LakeSimulator.measure_performance`
(simulation.py:949-979).

Three floors are measured for reference:
  • `SELECT 1`            — Spark planning and job submission with no table access
  • single-row table      — catalog resolution + metadata read + one file
  • n-file table          — the environment's actual regime (n = 1 ... 200)

Usage (inside the lakegym-lite container):
    python3 revision_bench_query_latency.py --rows 40000 --reps 30
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from simulation import LakeSimulator, QueryType

# File counts are chosen to stay inside the envelope the environment itself can
# produce. Its largest compaction target is 128 KB, so a table of B bytes never
# occupies fewer than ceil(B / 128 KB) files; measuring below that would time a
# layout no policy in this paper can reach, and a single multi-megabyte Parquet
# file also exhausts the 2 GB driver in Iceberg's vectorized Arrow reader.
MAX_FILE_BYTES = 128 * 1024
FILE_COUNT_MULTIPLIERS = [1, 1.5, 2, 4, 8, 16, 32]
QUERY_TYPES = [QueryType.TIME_RANGE, QueryType.REGION_FILTER, QueryType.SENSOR_LOOKUP,
               QueryType.TYPE_FILTER, QueryType.FULL_SCAN]


def git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd='/app', text=True).strip()
    except Exception:
        return 'unknown'


def time_sql(spark, sql: str, reps: int) -> list:
    """Time a SQL statement `reps` times, clearing the cache before each run."""
    out = []
    for _ in range(reps):
        spark.catalog.clearCache()
        start = time.time()
        spark.sql(sql).collect()
        out.append((time.time() - start) * 1000.0)
    return out


def table_stats(sim) -> dict:
    row = sim._spark.sql(
        f'SELECT count(*) AS n, sum(file_size_in_bytes) AS b '
        f'FROM {sim.FULL_TABLE_NAME}.files').collect()[0]
    n_rows = sim._spark.sql(
        f'SELECT count(*) AS c FROM {sim.FULL_TABLE_NAME}').collect()[0]['c']
    return {'files': int(row['n']), 'bytes': int(row['b'] or 0), 'rows': int(n_rows)}


STAGING_TABLE = 'iceberg.default.events_latency_staging'


def materialise_into(sim, n_files: int, limit: int | None = None) -> None:
    """(Re)create the measured table from the staging copy, split into exactly
    `n_files` data files.

    The rows live in a second Iceberg table rather than in the driver. DROP TABLE
    purges the Parquet files, so a DataFrame still referencing the measured table
    cannot be used to rewrite it; and collecting 10s of thousands of rows through
    Arrow exhausts the 2 GB driver this environment runs with. Staging keeps the
    data in the cluster and guarantees every file count scans identical rows.

    The Spark configuration is deliberately left exactly as
    `SparkConfig.get_spark_configs()` sets it, because the quantity being measured
    is this environment's own query latency.
    """
    spark = sim._spark
    df = spark.read.table(STAGING_TABLE)
    if limit is not None:
        df = df.limit(limit)
    spark.sql(f'DROP TABLE IF EXISTS {sim.FULL_TABLE_NAME}')
    (df.repartition(n_files).writeTo(sim.FULL_TABLE_NAME)
       .using('iceberg').tableProperty('write.format.default', 'parquet').create())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=20000,
                    help='rows to materialise; 20k matches the ~4.4 MB episode-mean table')
    ap.add_argument('--batch', type=int, default=100, help='rows per ingest batch')
    ap.add_argument('--reps', type=int, default=30, help='timed repetitions per query')
    ap.add_argument('--out', default='/app/revision/phase9_latency')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now()

    sim = LakeSimulator()
    print(sim.initialize())
    sim.reset()
    spark = sim._spark

    # ── floor 1: no table access at all ───────────────────────────────────
    print('\n── floor: SELECT 1 (planning + job submission only)')
    floor_select1 = time_sql(spark, 'SELECT 1', args.reps)
    print(f'   median {statistics.median(floor_select1):.1f} ms')

    # ── build the table ───────────────────────────────────────────────────
    sim.config.manual_override_active = True
    sim.config.ingestion_rate_rows = args.batch
    n_batches = max(1, args.rows // args.batch)
    print(f'\n── ingesting {n_batches} × {args.batch} rows')
    t0 = time.time()
    for i in range(n_batches):
        sim.ingest_micro_batch()
        if (i + 1) % 50 == 0:
            print(f'   {i + 1}/{n_batches} batches ({time.time() - t0:.0f}s)')
    sim.config.manual_override_active = False
    base = table_stats(sim)
    print(f'   table: {base["rows"]:,} rows, {base["files"]:,} files, '
          f'{base["bytes"] / 1024:.0f} KB')

    # stage the rows in a second Iceberg table so every regime below scans
    # identical data without routing it through the driver
    spark.sql(f'DROP TABLE IF EXISTS {STAGING_TABLE}')
    (spark.read.table(sim.FULL_TABLE_NAME).repartition(4)
     .writeTo(STAGING_TABLE).using('iceberg').create())
    staged = spark.read.table(STAGING_TABLE).count()
    staged_bytes = int(spark.sql(
        f'SELECT sum(file_size_in_bytes) AS b FROM {STAGING_TABLE}.files'
    ).collect()[0]['b'] or 0)
    print(f'   staged {staged:,} rows ({staged_bytes / 1024:.0f} KB consolidated) '
          f'in {STAGING_TABLE}')

    # ── floor 2: single row, single file ──────────────────────────────────
    records = []
    print('\n── floor: single-row table')
    materialise_into(sim, 1, limit=1)
    for qt in QUERY_TYPES:
        sql, label = sim._build_query(qt)
        lat = time_sql(spark, sql, args.reps)
        records.append({'regime': 'single_row', 'files': 1, 'rows': 1,
                        'bytes': table_stats(sim)['bytes'],
                        'query': qt.name, 'label': label,
                        'median_ms': statistics.median(lat),
                        'mean_ms': statistics.mean(lat),
                        'p05_ms': min(lat), 'p95_ms': max(lat),
                        'reps': args.reps})
        print(f'   {qt.name:16s} median {statistics.median(lat):7.1f} ms')

    # ── sweep: same rows, n files ─────────────────────────────────────────
    # consolidated bytes, not the as-ingested footprint: the table as written by
    # ingestion carries per-file overhead across ~800 tiny files
    n_min = max(1, -(-staged_bytes // MAX_FILE_BYTES))      # ceil division
    file_counts = sorted({int(round(n_min * m)) for m in FILE_COUNT_MULTIPLIERS})
    print(f'\n── file-count sweep: {file_counts} '
          f'(floor {n_min} = {staged_bytes / 1024:.0f} KB / 128 KB per file)')
    for n in file_counts:
        print(f'\n── {n} file(s), identical rows')
        materialise_into(sim, n)
        st = table_stats(sim)
        print(f'   actual: {st["files"]} files, {st["rows"]:,} rows, {st["bytes"]/1024:.0f} KB')
        for qt in QUERY_TYPES:
            sql, label = sim._build_query(qt)
            lat = time_sql(spark, sql, args.reps)
            records.append({'regime': 'sweep', 'files': st['files'], 'rows': st['rows'],
                            'bytes': st['bytes'], 'query': qt.name, 'label': label,
                            'median_ms': statistics.median(lat),
                            'mean_ms': statistics.mean(lat),
                            'p05_ms': min(lat), 'p95_ms': max(lat),
                            'reps': args.reps})
            print(f'   {qt.name:16s} median {statistics.median(lat):7.1f} ms')

    # ── persist ───────────────────────────────────────────────────────────
    import csv
    with open(out / 'query_latency_vs_files.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    manifest = {
        'script': 'revision_bench_query_latency.py',
        'purpose': 'fixed query overhead vs per-file scan cost (round-2 reviewer point 8)',
        'rows_target': args.rows, 'batch': args.batch, 'reps_per_query': args.reps,
        'file_counts': file_counts,
        'max_file_bytes': MAX_FILE_BYTES,
        'base_table': base,
        'staged_rows': staged, 'staged_bytes': staged_bytes,
        'select1_median_ms': statistics.median(floor_select1),
        'select1_mean_ms': statistics.mean(floor_select1),
        'git_commit': git_commit(),
        'platform': platform.platform(), 'python': sys.version.split()[0],
        'started_at': started.isoformat(),
        'finished_at': datetime.now().isoformat(),
        'total_wall_clock_sec': round(time.time() - t0, 1),
    }
    spark.sql(f'DROP TABLE IF EXISTS {STAGING_TABLE}')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'\nwrote {out}/query_latency_vs_files.csv and manifest.json')


if __name__ == '__main__':
    main()
