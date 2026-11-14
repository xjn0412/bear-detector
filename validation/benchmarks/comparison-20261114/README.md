# Comparison evidence · 2026-11-14

- `comparison.json`: exact plotted values, primary-source URLs, recipe/parameter conventions, own measured metrics.
- `leaderboard-source.json`: a small selection of identifiable rows from the exact requested SOTA2 page, retrieval date and SHA256 of the retrieved HTML. Only100 rendered rows were inspected; no global rank is inferred. Primary sources validate plotted external AP. The full third-party page is not redistributed.
- `assets/benchmarks/comparison-*.{png,svg}`: generated with `python tools/plot_comparison.py` (matplotlib required).

The graphs use current mixed-storage PyTorch AMP AP for cross-model context, backend-specific AP for Bear latency. Historical training AP remains unchanged. No external model was rerun; no GPU benchmark or training was launched. Snapshot tables include multiple training regimes; external latency is never plotted against Bear's5090 timings. Sources are listed per row so future changes can be reviewed.
