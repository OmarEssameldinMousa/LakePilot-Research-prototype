"""
Guarded "% of gap closed" metric (Phase 8 item 2).

The published Table 7 reports −3,207 % for DDQN. That figure is a
division-by-near-zero artifact: when a policy starts almost at the frozen
ceiling, |initial_gap| ~ 0 and the ratio explodes without carrying any meaning.

This module reports the percentage only when the initial gap is large enough for
the ratio to be interpretable, and returns None ("n/a") otherwise.
"""
from __future__ import annotations

from typing import Optional

# Minimum |initial gap| (in units of mean global reward) for the ratio to be
# reported. The v5 reward is O(0.1–0.3) per step, so a gap below 0.01 is within
# episode-to-episode noise and cannot support a percentage claim.
MIN_ABS_INITIAL_GAP = 0.01


def pct_gap_closed(
    initial: float,
    final: float,
    ceiling: float,
    min_abs_initial_gap: float = MIN_ABS_INITIAL_GAP,
) -> Optional[float]:
    """Percentage of the initial gap to `ceiling` that was closed.

    Args:
        initial: metric value at the first episode.
        final:   metric value at the last episode.
        ceiling: reference value being approached (e.g. frozen-agent performance).

    Returns:
        Percentage in [-inf, 100], or None when the initial gap is too small for
        the ratio to be meaningful (caller should print "n/a").
    """
    initial_gap = ceiling - initial
    if abs(initial_gap) < min_abs_initial_gap:
        return None
    final_gap = ceiling - final
    return (initial_gap - final_gap) / abs(initial_gap) * 100.0


def format_gap_closed(initial: float, final: float, ceiling: float) -> str:
    """Table-ready string: a percentage, or 'n/a' with the reason implied."""
    pct = pct_gap_closed(initial, final, ceiling)
    return 'n/a' if pct is None else f'{pct:+.1f}%'


if __name__ == '__main__':
    # Reproduces the pathological Table 7 case and the guard's behaviour.
    cases = [
        ('normal improvement',      0.05, 0.15, 0.20),
        ('normal regression',       0.15, 0.05, 0.20),
        ('near-zero initial gap',   0.1999, 0.05, 0.20),   # produced -3,207%-style values
        ('exactly at ceiling',      0.20, 0.10, 0.20),
    ]
    for name, i, f, c in cases:
        print(f'{name:24s} initial={i:.4f} final={f:.4f} ceiling={c:.4f} -> '
              f'{format_gap_closed(i, f, c)}')
