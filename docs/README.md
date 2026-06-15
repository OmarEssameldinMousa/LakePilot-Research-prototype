# LakePilot — Project Website

An interactive, in-depth walkthrough of the paper
**"Workload-Aware Data Lakehouse Maintenance Using Hierarchical Deep Reinforcement Learning."**

This is a static site (no build step) styled after the Databricks brand. It covers the
problem (small-file proliferation & partition drift), the hierarchical multi-agent
architecture, the LakeGym v5 environment, the experiment design, the full simulation
workflow, and all collected metrics — with interactive Chart.js visualisations plus the
publication figures exported directly from `Research_notebook/lakegym_comprehensive_study.ipynb`.

## Structure

```
docs/
├── index.html            # single-page site
├── css/styles.css        # Databricks-inspired design system
├── js/data.js            # all numbers, extracted verbatim from the paper tables
├── js/charts.js          # interactive Chart.js renderings
├── js/app.js             # nav, tabs, dynamic tables, heatmap, lightbox
└── assets/figures/*.png  # 22 figures exported from the research notebook
```

## Deploy to GitHub Pages

1. Push this repository to GitHub.
2. **Settings → Pages → Build and deployment**.
3. Source: **Deploy from a branch**. Branch: `main`, folder: **`/docs`**.
4. Save. The site publishes at `https://<user>.github.io/<repo>/`.

The `.nojekyll` file disables Jekyll processing so all assets are served as-is.

## Run locally

```bash
cd docs
python3 -m http.server 8099
# open http://localhost:8099
```

## Regenerating figures

The figures are extracted from the notebook's cell outputs. Re-run
`Research_notebook/lakegym_comprehensive_study.ipynb`, then re-export the embedded PNGs
into `docs/assets/figures/` (one per figure cell).
