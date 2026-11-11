#!/usr/bin/env bash
# Linux / WSL. Downloads only the two models the verified engine uses.
set -e
pip install -r requirements-verified.txt
huggingface-cli download IDEA-Research/grounding-dino-tiny --local-dir /opt/forte/models/grounding-dino-tiny
huggingface-cli download HuggingFaceTB/SmolVLM2-2.2B-Instruct --local-dir /opt/forte/models/smolvlm2-2.2b-instruct
echo "export DINO_PATH=/opt/forte/models/grounding-dino-tiny SMOLVLM_PATH=/opt/forte/models/smolvlm2-2.2b-instruct"
