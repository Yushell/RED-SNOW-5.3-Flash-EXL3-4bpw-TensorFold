#!/usr/bin/env bash
# RED-SNOW TF-pack big conversion — RESUMED after workdir migration (2026-10-10).
# Workdir now lives on storage via sshfs (/om/redsnow-work); the old local copy stays on
# spark-b (~/redsnow-tf-build/work) as a checkpoint backup until the run completes.
# Resume reads the latest checkpoint from -w; other job args are restored from the stored job.
set -euo pipefail
docker rm -f redsnow-convert >/dev/null 2>&1 || true
docker run --rm --name redsnow-convert \
  --gpus all --ipc=host \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  --user 1000:1000 -e HOME=/tmp \
  -v /mnt/storage-models:/om \
  -v /mnt/redsnow-build:/build \
  -w /opt/exllamav3 \
  redsnow-exl3:764b4ad \
  python convert.py \
    -r \
    -cb mcg \
    -mb 4 \
    -vb 16 \
    -w /om/redsnow-work \
    -i /om/RED-SNOW-5.3-FLASH-BF16 \
    -o /om/RED-SNOW-5.3-FLASH-EXL3-4bpw-TensorFold \
    --override_anyway
echo "CONVERT DONE: $(date -Is) (exit $?)"
