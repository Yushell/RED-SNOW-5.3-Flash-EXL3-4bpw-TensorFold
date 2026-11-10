# Changelog

## v1.2.0 · 2026-11-10

- Measured the pack on the GLM recipe's 125-patch image (label `9be02706cb1e`) with KV sharding
  (`TF_GLM_KV_SHARD=1`), the spill tier (`SPILL_GIB=64`), the effort-tail cache and `expandable_segments`:
  house bench 114.5 / 56.2 / 96.9 t/s (structured / prose / coding, 1 stream); llmapi c1 / c2 / c4
  47.3 / 83.9 / **154.3** Gen TPS at 100% success; pool **4,661,248 tokens** at 4 x 1M; load 141.9 s;
  a 221,885-token needle exact in 130.5 s.
- Added the exact prepare invocation (`PULL=0` is required: the published image predates `0129`, `0130`, `0132`)
  and the NFS-export precondition (a dead export hangs the worker's hard-mounted `glm53-hf` volume).
- The v1.0.0 speed table is kept as the original build record and labelled as such.

## v1.1.0 · 2026-11-10

- Documented the post-v1.0.0 engine settings: KV sharding (`TF_GLM_KV_SHARD=1`, 1.88x pool at 1M x 4 for
  ~1.5% decode, byte-equal replies), the kept-state spill tier, the effort-tail cache, patch `0130` for replies
  cut inside an unclosed think block, `expandable_segments` with the env-forwarding patch for rank-1 memory growth,
  the restart-once-after-image-change rule (`#148`) and the `MAX_NUM_BATCHED_TOKENS=7168` rule.

## v1.0.0 · 2026-10-11

- Initial release: EXL3 4 bpw TensorFold pack of RED-SNOW-5.3-Flash built from the BF16 release
  (23 shards, 164 GiB, 37,161 EXL3 tensors all mcg, 0 parse issues under TensorFold's reader).
- Validated on a 2x DGX Spark pair: 43-54 tok/s single stream, 87.4 tok/s aggregate at 4 streams,
  load ~2.3 min, full-window 1M needle pass (1,035,952-token prompt), vision pass, quality spot-checks pass.
- Serving kit: env overrides, local snapshot staging layout, and the TensorFold-compatible chat template
  (vision-critical: the deployment template from the model repo makes `--vision` refuse to start).
- Build kit: conversion recipe, recipe generator (`scripts/make_recipe.py`), fused-layout fix-up
  (`scripts/split_fused.py`), and the conversion scripts we ran.
