# V2 COCO launch and scaling evidence

[Protocol and summarized comparison](../../../docs/benchmarks/V2_COCO_TRAINING.md).

Accepted timing files: `L-g4.json`, `L-g8.json`, `B-g4.json`, `B-g8.json`, `S-g4.json`, `S-g8.json`, `T-g4.json`, `T-g8.json`. Each includes configuration/checkpoint hashes, warmup and measured counts, raw seconds, samples/second, skipped updates, selected precision and peak memory. All accepted timed windows completed20 successful updates without skips.

`B-g4-rejected-warmup20.json` is the rejected initial attempt; it is excluded from the comparison. The accepted replacement uses60 warmup windows. Accepted L comparisons both use20 warmup windows; accepted B/S/T comparisons both use60.

`timing-subset-membership.json` records the1,024 train and512 held-out val image IDs and annotation hashes. These subsets are only for scheduling estimates. Formal training uses the complete datasets recorded in `assets-ready.json`.

`train-archive-verification.json` verifies the full19,336,861,798-byte train ZIP against the official S3 multipart ETag (2,306 parts of8MiB, final part shorter) and records its SHA256. The mirror file is byte-identical under this checksum check. Official annotations/val archives were downloaded directly.

`hardware.json` records GPUs, memory, power limits and topology. `ddp-fit-smoke.json` records the two-GPU synthetic launcher/checkpoint/EMA/early-stop verification. Additional CPU regression tests are in [test_v2_training.py](../../../tests/test_v2_training.py).

No AP in a timing run or synthetic smoke run is a v2 COCO benchmark result. Trained v2 AP will only be reported from formal full-val evaluation.
