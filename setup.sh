#!/usr/bin/env bash
# One-time setup on the Linux GPU machine (Ubuntu 22.04/24.04, NVIDIA driver already installed).
#   bash setup.sh            # full precision, needs ~20 GB VRAM
#   QUANT=4bit bash setup.sh # also installs bitsandbytes for 12-16 GB cards
set -euo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "No NVIDIA driver found"; exit 1; }

echo "== system packages =="
sudo apt-get update -y
sudo apt-get install -y ffmpeg python3-venv python3-pip git

echo "== python env =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch with CUDA 12.4 wheels (works with driver >= 550). Change cu124 if your driver is older.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -r requirements-ml.txt
if [[ "${QUANT:-}" == "4bit" ]]; then pip install bitsandbytes; fi

python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory/1e9:.1f} GB")
EOF

echo "== model weights (about 17 GB, one time) =="
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/qwen2.5-vl-7b
huggingface-cli download google/owlv2-base-patch16-ensemble --local-dir models/owlv2-base

echo
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  export VLM_PATH=models/qwen2.5-vl-7b OWL_PATH=models/owlv2-base"
echo "  python forte.py predict --videos data/videos && python forte.py accuracy --write"
