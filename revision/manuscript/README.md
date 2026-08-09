# Revised manuscript — Overleaf package

Everything needed to rebuild the revised paper. Verified to compile cleanly:
**24 pages, 0 errors, 0 undefined references, 0 overfull boxes > 10 pt**
(pdfTeX, two passes).

## What to upload to Overleaf

| Upload | From |
|---|---|
| `lakegym_scirep_revised.tex` | this directory |
| `charts/` (all 16 PDFs + PNGs) | this directory |
| `wlscirep.cls` | your existing Overleaf project — unchanged |

`wlscirep.cls` is the Scientific Reports class and is **not** in this repository.
It is already in your Overleaf project; keep it. The revised `.tex` uses the same
`\documentclass[fleqn,10pt]{wlscirep}`, the same package list, the same float
parameters, the same three `\definecolor` entries, the same macro set, and the
same title/author/affiliation block as the original, so it drops into the same
project.

LaTeX picks the `.pdf` of each figure automatically; the `.png` copies are for
previewing only and can be omitted from the upload.

### One added package

```latex
\usepackage{xurl}
```

This is the fix for reviewer 2's truncated-URL comment. Without it the
repository URL overflows the column and is clipped in the PDF — which is exactly
how `…-prototype` became `…-prot` in the original submission. `xurl` is on
Overleaf by default.

## Figures

Eight are regenerated from the revision data by `make_figures.py`; seven are
copied from the phase analysis output so the manuscript cannot drift from the
per-phase summaries; two are your original diagrams, reused unchanged.

| File | Used in | Source |
|---|---|---|
| `fig_ranking` | Fig. 1, policy ranking | regenerated |
| `fig_seed_variability` | Fig. 2, per-seed spread | regenerated |
| `fig_ablation` | Fig. 3, specialist ablation | regenerated |
| `fig_meta_tornado` | Fig. 4, meta-controller sensitivity | phase 3 |
| `fig_pruning_gap` | Fig. 5, pruning-matrix sensitivity | phase 3 |
| `fig_generalization` | Fig. 6, held-out workloads | phase 5 |
| `fig_adaptive` | Fig. 7, online adaptation | regenerated |
| `fig_matched_mlp` | Fig. 8, capacity-matched control | regenerated |
| `fig_attention_profile` | Fig. 9a, attention by position | phase 7 |
| `fig_attention_transitions` | Fig. 9b, attention at transitions | phase 7 |
| `fig_cost` | Fig. 10, cost decomposition | regenerated |
| `fig_workload_ingestion` | Fig. 11, workload schedule | regenerated |
| `fig_pruning_matrix` | Fig. 12, pruning matrix | regenerated |
| `fig_arch_system` | Fig. 13a, system flow | **your original diagram** |
| `fig_arch_models` | Fig. 13b, neural candidates | **your original diagram** |

`fig_forest_eval500` and `fig_forest_std1000` are present but **not used**: they
label policies with internal run identifiers (`MultiAgentRL_ddqn`) rather than
manuscript names, and `fig_ranking` already carries the same confidence
intervals. Use them only if you relabel them first.

`fig_arch_system.png` and `fig_arch_models.png` were recovered from your original
Overleaf bundle, which is currently in the system Trash
(`~/.local/share/Trash/files/Workload_Aware_Data_Lakehouse_…`). They are
schematics of the system and of the three model families, both unchanged by the
revision, so they are reused as-is. **If you empty the Trash, these two files
survive only in this directory** — they have no other source in the repository.

## Regenerating

```bash
python3 revision/manuscript/make_figures.py
```

Reads directly from the phase result CSVs and the raw per-seed run directories;
writes PDF + PNG at 300 dpi into `charts/`. Colours follow the manuscript
convention (`agentblue` / `agentteal` / `agentcoral`).

## Figures from the original submission that are now retired

These are invalidated by the pipeline corrections (they were produced from the
affected checkpoints) and have no replacement, because the analyses they
supported were either withdrawn or restructured:

`fig_cumreward`, `fig_heuristic_baseline`, `fig_rlcomp`, `fig_rl_dynamics`,
`fig_reward_decomp`, `fig_csize`, `fig_action_attentive`, `fig_action_mlpppo`,
`fig_action_ddqn`, `fig_frozen_consistency`, `fig_adaptive_episodes`,
`fig_rl_vs_baselines`, and the old `fig_cost`.

Do not carry them over.

## What changed in the text

Structural changes against `revision/lakegym_scirep.tex`:

- **New** — `Related work` section (reviewer 2: refs 12–28 were never discussed)
- **New** — `Corrections to the training pipeline` subsection, opening Results
- **New** — Results subsections: sensitivity, oracle meta-controller,
  generalization, attention mechanism analysis, computational cost
- **New** — `Code availability` section with the Zenodo DOI
- **Rewritten** — Abstract, ranking, architecture comparison, ablation, online
  adaptation, Discussion, evaluation protocol, Data availability
- **Corrected** — dueling aggregation equation (mean-centred form); Double-DQN
  target equation added
- **Added refs 29–34** — double Q-learning, option-critic, AWR, bootstrap, Holm,
  Cliff

Claims withdrawn: the architecture ranking by attention, the +3.5 % attention
advantage, the DDQN adaptation collapse and the on-policy/off-policy conclusion
drawn from it, and "all three RL agents outperform 17 baselines".

The point-by-point response to reviewers is at
`revision/phase8_manuscript/POINT_BY_POINT_RESPONSE.md`.

## Local compile check

`preview_lakegym_scirep_revised.pdf` in this directory is the verified build.
To reproduce it you need `wlscirep.cls` alongside the `.tex`:

```bash
pdflatex lakegym_scirep_revised.tex && pdflatex lakegym_scirep_revised.tex
```
