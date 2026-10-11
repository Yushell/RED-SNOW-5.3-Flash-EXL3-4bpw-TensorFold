#!/usr/bin/env python3
"""Split fused qkv_proj/conv1d back into q/k/v parts (torch layout) using source tensors.
Run inside a torch container on gx10-a. Mounts: /pack = staged snapshot dir; /src = olares models dir."""
import json, os, re, torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from collections import defaultdict

PACK = "/pack"
SRC = "/src/RED-SNOW-5.3-FLASH-BF16"
p_idx_p = os.path.join(PACK, "model.safetensors.index.json")
s_idx_p = os.path.join(SRC, "model.safetensors.index.json")
p_doc = json.load(open(p_idx_p))
pidx = p_doc["weight_map"]
sidx = json.load(open(s_idx_p))["weight_map"]

fused_re = re.compile(r"^(model\.language_model\.layers\.\d+)\.self_attn\.(qkv_proj|conv1d)\.weight$")
suffix = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "conv1d": ("q_conv1d", "k_conv1d", "v_conv1d")}

work = []
for k, shard in list(pidx.items()):
    m = fused_re.match(k)
    if m:
        pre, kind = m.group(1), m.group(2)
        parts = [f"{pre}.self_attn.{p}.weight" for p in suffix[kind]]
        for p in parts:
            assert p in sidx, f"source missing {p}"
        work.append((k, kind, shard, parts))
print(f"fused tensors to split: {len(work)}")

def read_src(t):
    with safe_open(os.path.join(SRC, sidx[t]), framework="pt") as f:
        return f.get_tensor(t)

by_shard = defaultdict(list)
for k, kind, shard, parts in work:
    by_shard[shard].append((k, kind, parts))

changed = []
for shard, items in sorted(by_shard.items()):
    p = os.path.join(PACK, shard)
    sd = load_file(p)
    for k, kind, parts in items:
        fused = sd[k]
        src_parts = [read_src(pt) for pt in parts]
        cat = torch.cat(src_parts, dim=0)
        assert tuple(cat.shape) == tuple(fused.shape), (k, cat.shape, fused.shape)
        assert torch.equal(cat.to(fused.dtype), fused), f"value mismatch: {k}"
        del sd[k]
        for pt, t in zip(parts, src_parts):
            sd[pt] = t.to(fused.dtype)
    save_file(sd, p, metadata={"format": "pt"})
    os.chmod(p, 0o644)
    changed.append(shard)
    print(f"{shard}: split {len(items)} fused tensors; keys now {len(sd)}; mode 644")

for k, kind, shard, parts in work:
    del pidx[k]
    for pt in parts:
        pidx[pt] = shard

out = {"metadata": p_doc.get("metadata", {}), "weight_map": dict(sorted(pidx.items()))}
with open(p_idx_p, "w") as f:
    json.dump(out, f, indent=2)
os.chmod(p_idx_p, 0o644)
print(f"index updated: {len(pidx)} entries; shards changed: {len(changed)}")
print("DONE")
