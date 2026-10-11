#!/usr/bin/env bash
# RED-SNOW TF-pack big conversion — runs on spark-b, source read from storage over sshfs (/om),
# output written back to storage; work dir stays local on spark-b (/build/work).
# Resume after an interruption: re-run with -r appended (replace "python convert.py" below with
# "python convert.py -r" and keep the same -w path).
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
    -i /om/RED-SNOW-5.3-FLASH-BF16 \
    -o /om/RED-SNOW-5.3-FLASH-EXL3-4bpw-TensorFold \
    -w /build/work \
    -rcp /build/redsnow-recipe.yaml \
    -cb mcg -mb 4 -vb 16
echo "CONVERT DONE: $(date -Is) (exit $?)"
