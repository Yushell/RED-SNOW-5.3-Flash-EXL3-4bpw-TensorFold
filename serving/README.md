# Serving kit

## chat_template.jinja

Drop this into the pack's snapshot (replace the pack's `chat_template.jinja`) before serving.

It is the Z.ai base template revision in which the media-reminder branch appears exactly once.
TensorFold's `--vision` requires that branch: `media_template()` swaps it for GLM's image/video
markers at load (see `tensorfold/families/glm5_next/cuda/app.py`). Without it, `--vision` refuses
to start with "no media branch TensorFold knows how to extend". The RED-SNOW deployment template
and the current zai-org main template both fail that check.

This exact revision is the one the Mia-AiLab GLM pack ships and the one the working GLM lane
serves with.

## Env overrides (on top of the MiaAI-Lab recipe defaults)

```bash
MODEL_ID=local/RED-SNOW-5.3-Flash-EXL3-4bpw-TensorFold \
WORKER_WEIGHTS=nfs \
SERVED_NAME=RED-SNOW-5.3-Flash-EXL3 \
HF_HUB_OFFLINE=1 \
./start.sh restart
```

- `MODEL_ID`: resolves in the local HF cache under `models--local--RED-SNOW-5.3-Flash-EXL3-4bpw-TensorFold`.
  The recipe pins revisions for Mia's checkpoints only; for a local MODEL_ID set `MODEL_REVISION` yourself
  if you want the pin recorded (any consistent rev string works).
- `WORKER_WEIGHTS=nfs`: rank 1 reads the head's HF cache over NFS (read-only export), so the worker
  needs no second copy of the 164 GiB pack.
- `HF_HUB_OFFLINE=1`: serve strictly from the local cache (recommended once everything is staged).

Recipe defaults that apply: DFlash2 drafts, vision on, thinking on, TF_GLM_EFFORT_TAIL=1, SPILL_GIB=64.

## Staging layout on the head Spark

```
~/.cache/huggingface/hub/models--local--RED-SNOW-5.3-Flash-EXL3-4bpw-TensorFold/
  refs/main         # the rev string (must match MODEL_REVISION when set)
  snapshots/<rev>/  # 23 shards + model.safetensors.index.json + config files + chat_template.jinja
```

File modes: 644 on everything (600 shard modes break the worker's NFS read).
