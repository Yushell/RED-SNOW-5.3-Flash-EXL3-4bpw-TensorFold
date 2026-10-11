#!/usr/bin/env python3
# make_recipe.py v2 — generate an experts-only EXL3 conversion recipe for exllamav3 convert.py.
# Walks the budgeted module graph INCLUDING the MTP component (mirrors convert_model.get_base_model).
# Usage: python make_recipe.py <model_dir> <out_yaml> <expert_regex> <expert_bits> <other_bits> <head_bits>
import re, sys, yaml
from exllamav3 import Config, Model
from exllamav3.modules.linear import Linear

model_dir, out_path, pattern_s, expert_bits, other_bits, head_bits = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
pattern = re.compile(pattern_s)

config = Config.from_directory(model_dir)
model = Model.from_config(config)
print("model_classes:", sorted(config.model_classes))

mtp_model = model.from_config(config, component="mtp") if "mtp" in config.model_classes else None
print("mtp component:", bool(mtp_model))
if "vision" in config.model_classes:
    print("note: vision component present; convert excludes it under -vb 16 (copied)")

cols = {"bits": [], "head_bits": [], "mtp_bits": []}

def walk(m):
    if isinstance(m, Linear) and getattr(m, "qmap", None) is not None:
        cols.get(getattr(m, "qbits_key", "bits"), cols["bits"]).append(m)
    for sm in m.modules:
        walk(sm)

for m in model.modules:
    walk(m)
if mtp_model:
    for m in mtp_model.modules:
        walk(m)

tensors = {}
sum_numel = 0
sum_bits = 0
expert_keys = []
other_keys = []
for m in cols["bits"]:
    is_expert = bool(pattern.search(m.key))
    bpw = expert_bits if is_expert else other_bits
    tensors[m.key] = bpw
    (expert_keys if is_expert else other_keys).append(m.key)
    if bpw <= 8:
        n = m.weights_numel()
        sum_numel += n
        sum_bits += n * bpw

achieved = (sum_bits / sum_numel) if sum_numel else 0.0
out = {"tensors": tensors, "head_bits": head_bits, "target_bpw": round(achieved, 4), "achieved_bpw": round(achieved, 4)}
with open(out_path, "w", encoding="utf8") as f:
    yaml.safe_dump(out, f, sort_keys=True)

print(f"wrote {out_path}")
print(f"budgeted tensors: {len(tensors)} (expert: {len(expert_keys)}, other: {len(other_keys)})")
print(f"head/mtp-class counts: {len(cols['head_bits'])} / {len(cols['mtp_bits'])}")
print(f"achieved_bpw (reporting): {achieved:.4f}")
print("expert sample:", expert_keys[:3])
print("other sample:", other_keys[:6])
