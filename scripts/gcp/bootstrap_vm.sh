#!/usr/bin/env bash
# Install WorldServe's Python deps on the VM. Run AFTER setup_vm.sh.
# Uses the Deep Learning VM's pre-installed CUDA 12.4 + PyTorch 2.5 base
# and only adds the project-specific extras.

set -euo pipefail

if [[ -z "${PROJECT_ID:-}" ]]; then
  echo "ERROR: source scripts/gcp/config.sh first." >&2
  exit 1
fi

# Run on the VM via SSH heredoc — no separate file copy needed.
gcloud compute ssh "${VM_NAME}" \
  --zone="${ZONE}" --project="${PROJECT_ID}" -- bash -s <<'EOF'
set -euo pipefail

echo "==> nvidia-smi"
nvidia-smi | head -n 15

echo "==> Confirming pre-installed PyTorch + CUDA"
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'  torch  : {torch.__version__}')
print(f'  cuda   : {torch.version.cuda}')
print(f'  device : {torch.cuda.get_device_name(0)}')
print(f'  vram   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo "==> Project pip deps (matches benchmarks/common.py _COMMON_PIP)"
pip install --quiet --upgrade pip
pip install --quiet \
  "diffusers>=0.32.0" \
  "transformers>=4.44.0" \
  "accelerate>=0.30.0" \
  "huggingface_hub>=0.24.0" \
  "torchao==0.5.0" \
  einops \
  "imageio[ffmpeg]" \
  nvtx \
  tabulate \
  sentencepiece \
  protobuf \
  packaging \
  ninja \
  timm \
  safetensors

echo "==> SageAttention (best-effort)"
pip install --quiet sageattention || \
  pip install --quiet git+https://github.com/thu-ml/SageAttention.git || \
  echo "  WARNING: SageAttention failed — sage-attn benchmarks will fall back to SDPA"

echo "==> flash-attn (best-effort, no-build-isolation)"
pip install --quiet flash-attn --no-build-isolation || \
  echo "  WARNING: flash-attn failed — FA3 benchmarks will fall back"

echo "==> Verifying imports"
python3 -c "
import diffusers, transformers, torchao, einops, safetensors
print(f'  diffusers   : {diffusers.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  torchao     : {torchao.__version__}')
"

echo "==> All deps installed."
EOF

echo
echo "Bootstrap done. Next:"
echo "  bash scripts/gcp/sync_code.sh    # push project to the VM"
