# RED-SNOW-5.3-Flash · EXL3 4 bpw TensorFold pack (2x DGX Spark)

A TensorFold-calibrated **EXL3 4 bpw pack of [RED-SNOW-5.3-Flash](https://huggingface.co/Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16)** built from the BF16 release, plus the serving overrides to run it on two DGX Sparks through the [MiaAI-Lab GLM TensorFold recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).

RED-SNOW-5.3-Flash is a cybersecurity (red-team / adversary-emulation) family variant of GLM-5.3-Flash. The publisher ships BF16 weights only, and TensorFold needs an experts-only EXL3 4-bit pack, so we built one. Everything below was measured on a 2x DGX Spark pair (GB10, RoCE link) in October 2026.

**The pack itself is not hosted here** (164 GiB). This repo is the build record, the conversion recipe, and the serving kit to reproduce it.

## Results at a glance

| Metric | Result |
|---|---|
| Single-stream decode | 43-54 tok/s (MTP drafts); up to 72 tok/s on prose/code with DFlash2 |
| Aggregate decode (DFlash2) | 63.5 tok/s at 2 streams · **87.4 tok/s at 4 streams** |
| Same-pair GLM-5.3-Flash reference | 103.2 tok/s prose at 4 streams (same class; this pack drafts with the base-model DFlash2 on adapted weights) |
| Load time | ~2.2-2.3 min to live |
| Context window | 1,048,576 tokens (fp8 KV) |
| Long-context recall | pass at 48K, 241K, and **full-window 1,035,952 tokens** (buried passphrase at 50% depth, found exactly) |
| Vision | pass (1.05 GiB tower; image probe answered correctly) |
| Quality spot-checks | long-form, arithmetic, code: all pass |
| Pack | 23 shards · 164 GiB · 37,161 EXL3 tensors (all mcg codebook) · TensorFold reader: "every EXL3 tensor is readable", 0 parse issues |
| Build cost | ~20 h conversion on ONE Spark (checkpointed and resumable), plus a small dry run |

## What's in the pack

| Part | Format |
|---|---|
| Routed experts, layers 3-44 | EXL3, 4 bpw, mcg codebook (gate / up / down) |
| MTP layer experts | EXL3, 4 bpw |
| Everything else (attention, KDA, shared experts, dense layers, embeddings, head, vision tower) | BF16, unchanged |

That layout is exactly what TensorFold's GLM CUDA reader accepts; the dense layers are quantized by the serving stack itself (`DENSE=q4` in the recipe).

## Serve it

Prerequisites: a checkout of the [MiaAI-Lab TensorFold recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) set up per its README (TensorFold v0.6.0, two Sparks, RoCE link), and the pack staged on the head Spark. The recipe resolves `MODEL_ID` from the local HF cache, so the pack is never pulled from a Hub.

1. Stage the pack into the head's HF cache (any rev string; it only needs to be consistent):

   ```
   ~/.cache/huggingface/hub/models--local--RED-SNOW-5.3-Flash-EXL3-4bpw-TensorFold/
     refs/main                  # contains the rev string
     snapshots/<rev>/           # the 23 shards + index + config + chat_template.jinja
   ```

   Make sure `chat_template.jinja` is the bundled one: `serving/chat_template.jinja` (why: see Notes).

2. Start the lane from the recipe directory:

   ```bash
   MODEL_ID=local/RED-SNOW-5.3-Flash-EXL3-4bpw-TensorFold \
   WORKER_WEIGHTS=nfs \
   SERVED_NAME=RED-SNOW-5.3-Flash-EXL3 \
   HF_HUB_OFFLINE=1 \
   ./start.sh restart
   ```

   `WORKER_WEIGHTS=nfs` has the worker read the head's cache over NFS read-only (no second 164 GiB copy). Omit `HF_HUB_OFFLINE=1` if you want the recipe free to pull the drafter.

3. Drafters: the recipe defaults work (DFlash2 drafts, vision on, thinking on). The numbers above were measured exactly that way.

## Build it

Toolchain: exllamav3 `convert.py` (fork `vcruz305/exllamav3` @ `764b4ad`, built as image `redsnow-exl3:764b4ad`), TensorFold v0.6.0 for verification.

1. Stage the BF16 source (642.7 GB, 203 files) on a machine with enough disk.
2. Generate the per-tensor recipe: `scripts/make_recipe.py` walks the module graph and emits `recipe/redsnow-recipe.yaml` (experts of layers 3-44 -> 4 bits, everything else -> 16, head -> 16). The checked-in recipe is the exact one used.
3. Convert:

   ```bash
   python convert.py \
     -i <RED-SNOW-5.3-FLASH-BF16 dir> \
     -o <output pack dir> \
     -w <work dir> \
     -b 4.0 -cb mcg -mb 4 -vb 16 \
     -rcp redsnow-recipe.yaml
   ```

   The job is checkpointed (~120 s) and resumable (`-w ... -r`). When resuming, **re-pass `-cb mcg -mb 4 -vb 16`**: otherwise convert.py defaults override the stored job settings (we hit exactly this; `-cb` silently reset to `mul1`).

4. Post-fixes we had to apply (check whether your run needs them):
   - The final shard carried 6 stale layer-45 tensors duplicating earlier keys; TensorFold's loader rejects duplicate checkpoint tensors. Remove them from the last shard.
   - The converter emitted fused `qkv_proj` / `conv1d` tensors, but TensorFold's TP2 split rules expect torch layout (`q_proj`/`k_proj`/`v_proj`, `conv1d` split to q/k/v). `scripts/split_fused.py` splits them (lossless dim-0 concat, both equality checks verified).
   - Shards must be mode 644 (some came out 600, which breaks the worker's read over NFS).
5. Verify:

   ```bash
   python -m tensorfold.cuda.exl3.inspect <pack dir>
   ```

   Expect 37,161 EXL3 tensors, all mcg, "every EXL3 tensor is readable", 0 parse issues.

## Notes and gotchas

- **Chat template (vision-critical):** TensorFold's `--vision` refuses to start unless `chat_template.jinja` contains one exact media-reminder branch (`"<reminder>You are unable to process this " ~ media_type ...`), which it rewrites into GLM's `<|begin_of_image|><|image|><|end_of_image|>` markers (`media_template()` in `families/glm5_next/cuda/app.py`). The RED-SNOW deployment template does not have that branch, and neither does the current zai-org main template (it renders media itself). Use the revision in `serving/chat_template.jinja` (taken from the Mia-AiLab GLM pack, i.e. the template the working GLM lane serves with). Bonus: that template also removes a short-prompt quirk where the deployment template left answers inside the reasoning channel with empty `content`.
- **Thinking and reply sizes:** the model thinks by default. Give long-form requests 2400+ `max_tokens` (thinking runs ~1.7k tokens on open tasks), and per-request `"chat_template_kwargs": {"enable_thinking": false}` works when you want a direct answer.
- **Draft speed:** the DFlash2 drafter is a base-model checkpoint, so on these adapted weights its acceptance is a little lower than on GLM. Expect the ~15% aggregate gap vs the same-pair GLM lane shown above.
- **Disk:** pack ~164 GiB (head) + the BF16 source if you keep it (~643 GB) + ~1 copy of scratch during conversion.
- **Not affiliated** with Blackfrost-AI, MiaAI-Lab, or the TensorFold authors; this is an independent community build. Licenses to respect: model weights MIT (Blackfrost-AI), TensorFold Apache-2.0, exllamav3 MIT, DFlash2 CC BY-NC-ND 4.0 (non-commercial). This repo's own content is Apache-2.0.

## Credits

- RED-SNOW-5.3-Flash weights: Blackfrost-AI.
- TensorFold serving stack and the GLM recipe structure this kit plugs into: ashhart / MiaAI-Lab.
- Chat template revision: Z.ai (via the Mia-AiLab pack).
