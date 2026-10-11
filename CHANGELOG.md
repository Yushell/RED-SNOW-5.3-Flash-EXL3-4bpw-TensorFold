# Changelog

## v1.0.0 · 2026-10-11

- Initial release: EXL3 4 bpw TensorFold pack of RED-SNOW-5.3-Flash built from the BF16 release
  (23 shards, 164 GiB, 37,161 EXL3 tensors all mcg, 0 parse issues under TensorFold's reader).
- Validated on a 2x DGX Spark pair: 43-54 tok/s single stream, 87.4 tok/s aggregate at 4 streams,
  load ~2.3 min, full-window 1M needle pass (1,035,952-token prompt), vision pass, quality spot-checks pass.
- Serving kit: env overrides, local snapshot staging layout, and the TensorFold-compatible chat template
  (vision-critical: the deployment template from the model repo makes `--vision` refuse to start).
- Build kit: conversion recipe, recipe generator (`scripts/make_recipe.py`), fused-layout fix-up
  (`scripts/split_fused.py`), and the conversion scripts we ran.
