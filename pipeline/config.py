"""
Configuration of the "verified" engine (configs/verified.yaml) with environment overrides and cache keys.

    cfg = load()                      # dict
    key = stage_key(cfg, "detection") # short hash of the sections a stage depends on (+ code version)

Environment overrides: DINO_PATH / DET_PATH, SMOLVLM_PATH, VLM_DEVICE, VERIFIED_CONFIG (another yaml file).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "configs" / "verified.yaml"
CODE_VERSION = "verified-1.0"

# which config sections each cached stage depends on
STAGE_SECTIONS = {
    "detection": ["models.detector_path", "sampling.coarse_fps", "sampling.max_width", "detection"],
    "vlm": ["models.vlm_path", "vlm", "sampling", "detection", "tracking", "zones", "temporal", "rules"],
}


def load_dotenv(path: Path = ROOT / ".env"):
    """KEY=VALUE lines from .env (never overrides variables already set in the shell)."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    load_dotenv()
    p = Path(path or os.environ.get("VERIFIED_CONFIG", DEFAULT))
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    env = {
        "detector_path": os.environ.get("DINO_PATH") or os.environ.get("DET_PATH"),
        "vlm_path": os.environ.get("SMOLVLM_PATH"),
        "device": os.environ.get("VLM_DEVICE"),
    }
    for k, v in env.items():
        if v:
            cfg["models"][k] = v
    return _merge(cfg, overrides or {})


def get(cfg: dict, dotted: str):
    x = cfg
    for part in dotted.split("."):
        x = x[part]
    return x


def stage_key(cfg: dict, stage: str) -> str:
    parts = {s: get(cfg, s) for s in STAGE_SECTIONS[stage]}
    blob = json.dumps({"code": CODE_VERSION, "stage": stage, **parts}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]
