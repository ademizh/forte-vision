# Windows PowerShell, venv active. Downloads only the two models the verified engine uses.
pip install -r requirements-verified.txt
huggingface-cli download IDEA-Research/grounding-dino-tiny --local-dir C:\models\grounding-dino-tiny
huggingface-cli download HuggingFaceTB/SmolVLM2-2.2B-Instruct --local-dir C:\models\smolvlm2-2.2b-instruct
Write-Host "Set models.detector_path / models.vlm_path in configs\verified.yaml (or DINO_PATH / SMOLVLM_PATH in .env)"
