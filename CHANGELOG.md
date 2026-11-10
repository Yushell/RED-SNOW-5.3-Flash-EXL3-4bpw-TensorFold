# Changelog

## v1.1.0 · 2026-11-10

- Added a state note: the lane is cold-archived since 2026-10-21 (pack on the Olares One archive, prepared
  caches not on the pair, image removed 2026-10-22). The measured table is the v1.8-v1.9 build, not a live lane.
- Added "Bringing it up to date": KV sharding (`TF_GLM_KV_SHARD=1`, 1.88x pool at 1M x 4 for ~1.5% decode,
  byte-equal replies), the kept-state spill tier, the effort-tail cache, patch `0130` for replies cut inside an
  unclosed think block, `expandable_segments` with the env-forwarding patch for rank-1 memory growth, the
  restart-once-after-image-change rule (`#148`) and the `MAX_NUM_BATCHED_TOKENS=7168` rule.
- No numbers re-validated: the pack was not staged for a window, and the base recipe's own current figures are
  the same-pair reference now.

## v1.0.0 · 2026-10-11

- Initial release: EXL3 4 bpw TensorFold pack of RED-SNOW-5.3-Flash built from the BF16 release
  (23 shards, 164 GiB, 37,161 EXL3 tensors all mcg, 0 parse issues under TensorFold's reader).
- Validated on a 2x DGX Spark pair: 43-54 tok/s single stream, 87.4 tok/s aggregate at 4 streams,
  load ~2.3 min, full-window 1M needle pass (1,035,952-token prompt), vision pass, quality spot-checks pass.
- Serving kit: env overrides, local snapshot staging layout, and the TensorFold-compatible chat template
  (vision-critical: the deployment template from the model repo makes `--vision` refuse to start).
- Build kit: conversion recipe, recipe generator (`scripts/make_recipe.py`), fused-layout fix-up
  (`scripts/split_fused.py`), and the conversion scripts we ran.
