# Comparison evidence · 2026-11-14

Primary policy: only COCO detector supervision, fixed640, no TTA; external representation pretraining is explicitly allowed/disclosed. `comparison.json` records16 eligible primary rows,14 supplemental rows,4 unreleased preview rows, and2 COCO-only higher-resolution context references. Unknown detection-pretraining provenance is excluded rather than assumed COCO-only.

`tools/plot_comparison.py` filters `primary_coco_only` for the cost graph. The accuracy-context graph uses eligible640 baselines and separately labeled DINO/Co-DINO COCO-only references at different input recipes. No extra-data model enters either primary chart. Extra-data leaderboard rows remain in COMPARISON_EXTRA_DATA docs.

`leaderboard-source.json` preserves selected numeric facts and the sourceHTML hash, not a complete ranking. Main comparison uses primary author sources per row. No external model was rerun; no training or GPU benchmark was launched. Same detector data does not imply equal representation pretraining, augmentation, loss, schedule, seeds or training budget.

Latest baseline update: RT-DETRv4 replaces v2 in16COCO-only primary rows.4v2 rows retained in`historical_models`. YOLO26 is explicitly visible in a separate requested-baseline chart/table, flagged Objects365→COCO. RT-DETRv4 costs from revised paper2510.25257v2 Table1; current AP49.8/53.7/55.4/57.0 agrees with official repository. All4figures have PNG/SVG outputs.
