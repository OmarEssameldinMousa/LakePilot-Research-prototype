r"""
Audit the manuscript's headline numbers against the CSVs that produced them.

Round 2 opened with a reviewer noticing that the manuscript and the response
letter reported the same study with different effect sizes and opposite signs,
and that three different values were given for what looked like one quantity.
Both were true, and both were the kind of drift that happens when a table is
edited by hand after the analysis is re-run. This script makes that class of
error mechanical to catch: it extracts numbers from the .tex and compares them
against the analysis outputs, failing loudly on any mismatch.

It deliberately checks a small set of load-bearing numbers rather than every
figure in the paper: the ones that appear in more than one place, or that a
reviewer has already questioned.

Usage:  python3 revision/stats/audit_manuscript_numbers.py [path/to/manuscript.tex]
        (exit code 1 if any check fails)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / 'revision' / 'manuscript' / 'lakegym_scirep_revised.tex'
P1 = ROOT / 'revision' / 'phase1_stats'
P9 = ROOT / 'revision' / 'phase9_latency'


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, name: str, claimed, actual, tol: float = 5e-5) -> None:
        self.checks += 1
        if claimed is None:
            self.failures.append(f'{name}: not found in manuscript')
            return
        if actual is None:
            self.failures.append(f'{name}: no source value available')
            return
        if abs(float(claimed) - float(actual)) > tol:
            self.failures.append(
                f'{name}: manuscript says {claimed}, source says {actual:.6g}')

    def report(self) -> int:
        print(f'{self.checks} checks run, {len(self.failures)} failed')
        for f in self.failures:
            print(f'  ✗ {f}')
        if not self.failures:
            print('  ✓ every checked number matches its source')
        return 1 if self.failures else 0


def tex_table_row(tex: str, label: str, row_key: str) -> list[str] | None:
    """Return the cells of the row containing `row_key` in the table with `label`."""
    i = tex.find(r'\label{' + label + '}')
    if i < 0:
        # caption labels sit after the tabular, so search the whole table env
        return None
    start = tex.rfind(r'\begin{table}', 0, i)
    block = tex[start:i]
    for line in block.splitlines():
        if row_key in line and '&' in line:
            return [c.strip() for c in line.split('\\\\')[0].split('&')]
    return None


def num(cell: str) -> float | None:
    if cell is None:
        return None
    c = (cell.replace(r'\textbf{', '').replace('}', '')
             .replace('$-$', '-').replace('$', '').replace(r'\,', '')
             .replace('{,}', '').replace(',', '').replace(r'\%', '').strip())
    m = re.search(r'-?\d+\.?\d*', c)
    return float(m.group()) if m else None


def interval(cells: list[str], idx: int) -> tuple[float, float] | None:
    if cells is None or idx >= len(cells):
        return None
    m = re.findall(r'-?\d+\.\d+', cells[idx].replace('$-$', '-'))
    return (float(m[0]), float(m[1])) if len(m) >= 2 else None


def main() -> int:
    tex_path = Path(sys.argv[1]) if len(sys.argv) > 1 else TEX
    tex = tex_path.read_text()
    a = Audit()

    # ── Table 1 against table1_rewritten_eval500.csv ──────────────────────
    t1 = pd.read_csv(P1 / 'table1_rewritten_eval500.csv')
    src = {r['policy']: r for _, r in t1.iterrows()}
    for tex_key, csv_key in [(r'\ddqn{}} (RL)', 'DDQN (hierarchical RL)'),
                             ('WorkloadAwareThreshold', 'WorkloadAwareThreshold'),
                             ('AlwaysCompact', 'AlwaysCompact_C128'),
                             (r'\attentive{}} (RL)', 'AttentivePPO-clip (hierarchical RL)'),
                             (r'\mlpppo{}} (RL)', 'MLP-PPO (hierarchical RL)'),
                             ('CompactOnly', 'CompactOnly (ablation)'),
                             ('PartitionOnly', 'PartitionOnly (ablation)')]:
        cells = tex_table_row(tex, 'tab:ranking', tex_key)
        s = src.get(csv_key)
        if cells is None or s is None:
            a.failures.append(f'Table 1 row {csv_key}: not located')
            a.checks += 1
            continue
        a.check(f'Table 1 {csv_key} reward', num(cells[2]), s['reward'], 1e-4)
        ci = interval(cells, 3)
        a.check(f'Table 1 {csv_key} CI low', ci[0] if ci else None, s['ci_low'], 1e-4)
        a.check(f'Table 1 {csv_key} CI high', ci[1] if ci else None, s['ci_high'], 1e-4)
        cci = interval(cells, 4)
        a.check(f'Table 1 {csv_key} cluster CI low', cci[0] if cci else None,
                s['cluster_ci_low'], 1e-4)
        a.check(f'Table 1 {csv_key} cluster CI high', cci[1] if cci else None,
                s['cluster_ci_high'], 1e-4)
        a.check(f'Table 1 {csv_key} seed sd', num(cells[5]), s['seed_sd'], 1e-4)

    # ── Table 2 against significance_matrix.csv ───────────────────────────
    sig = pd.read_csv(P1 / 'significance_matrix.csv')
    sig = sig[(sig.protocol == 'eval500') & (sig.metric == 'avg_global_reward')]

    def pair(x: str, y: str):
        m = sig[((sig.name_a == x) & (sig.name_b == y)) |
                ((sig.name_a == y) & (sig.name_b == x))]
        if m.empty:
            return None
        r = m.iloc[0]
        flip = -1.0 if r['name_a'] != x else 1.0
        return {'diff': flip * r['mean_diff'], 'delta': flip * r['cliffs_delta'],
                'p': r['permutation_p_corr'], 'n': r['n']}

    D = 'MultiAgentRL_ddqn'
    for tex_key, other in [('WorkloadAwareThreshold', 'WorkloadAwareThreshold'),
                           ('AlwaysCompact', 'AlwaysCompact_C128'),
                           ('Threshold10\\_C128', 'Threshold10_C128'),
                           ('Threshold10\\_C64', 'Threshold10_C64')]:
        cells = tex_table_row(tex, 'tab:significance', tex_key)
        s = pair(D, other)
        if cells is None or s is None:
            a.failures.append(f'Table 2 row {other}: not located')
            a.checks += 1
            continue
        a.check(f'Table 2 {other} n', num(cells[1]), s['n'], 0.5)
        a.check(f'Table 2 {other} delta', num(cells[4]), s['delta'], 0.006)
        a.check(f'Table 2 {other} p_Holm', num(cells[3]), s['p'], 5e-4)

    # ── Abstract quotes the cluster interval ──────────────────────────────
    m = re.search(r'mean global reward of ([\d.]+)\s*\n?\(95\\% CI \[([\d.]+), ([\d.]+)\]',
                  tex)
    ddqn = src['DDQN (hierarchical RL)']
    if m:
        a.check('Abstract reward', float(m.group(1)), ddqn['reward'], 1e-4)
        a.check('Abstract CI low (must be the cluster interval)',
                float(m.group(2)), ddqn['cluster_ci_low'], 1e-4)
        a.check('Abstract CI high (must be the cluster interval)',
                float(m.group(3)), ddqn['cluster_ci_high'], 1e-4)
    else:
        a.failures.append('Abstract: reward/CI sentence not found')
        a.checks += 1

    # ── Action coverage table ─────────────────────────────────────────────
    cov_path = P1 / 'action_coverage.csv'
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        for _, r in cov.iterrows():
            key = r['action'].replace('_', r'\_')
            cells = tex_table_row(tex, 'tab:coverage', key)
            if cells is None:
                continue
            a.check(f"coverage {r['pool']}/{r['action']} count", num(cells[2]),
                    r['count'], 0.5)

    # ── Latency decomposition, if phase 9 has run ─────────────────────────
    fit_path = P9 / 'latency_fit.json'
    if fit_path.exists():
        fit = json.loads(fit_path.read_text())
        for pat, key, tol in [
            (r'single-file query costs ([\d.]+)\\,ms', 'single_row_mean_ms', 0.5),
            (r'planning and submission floor of ([\d.]+)\\,ms', 'select1_median_ms', 0.5),
        ]:
            m = re.search(pat, tex)
            if m:
                a.check(pat[:40], float(m.group(1)), fit[key], tol)

    return a.report()


if __name__ == '__main__':
    sys.exit(main())
