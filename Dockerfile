# CPU image for the skeleton. When the model is added, switch the base to a CUDA image,
# e.g. FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime, and bake weights in (see README).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data
WORKDIR /srv

COPY requirements.txt requirements-verified.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# GPU image (verified engine): base pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime, then
# RUN pip install --no-cache-dir -r requirements-verified.txt   and mount the model folders (DINO_PATH, SMOLVLM_PATH)

COPY app ./app
COPY pipeline ./pipeline
COPY catalog ./catalog
COPY norms ./norms
COPY configs ./configs
COPY scripts ./scripts
COPY forte.py ./

EXPOSE 8000
# ONE worker on purpose: the job queue lives in this process and the GPU must not be shared.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
