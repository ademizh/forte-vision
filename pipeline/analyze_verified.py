"""
Service entry point for the verified engine (PIPELINE=verified in .env):

    analyze(video_path) -> {"violations": [{type, timecode_sec, description, evidence_frames, rule, ...}]}

Same shape as the legacy service (app/main.py does not change). Extra fields per violation: violation_id,
status (confirmed_candidate | needs_review), score, threshold, end_sec. The text separates the visual event
from the legal assessment: the output is a suspected violation for human review, never a legal finding.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import yaml

from . import config, engine, evidence

_LOCK = threading.Lock()
_STATE = {}
CATALOG = Path(os.environ.get("CATALOG", config.ROOT / "catalog" / "violations.yaml"))


def _catalog():
    if "cat" not in _STATE:
        cat = yaml.safe_load(open(CATALOG, encoding="utf-8"))
        by = {v["id"]: v for v in cat["violations"]}
        if isinstance(cat.get("other"), dict):
            by["other"] = cat["other"]
        _STATE["cat"] = by
    return _STATE["cat"]


def preload():
    with _LOCK:
        if "models" not in _STATE:
            cfg = config.load()
            m = engine.Models(cfg, mock=os.environ.get("FORTE_MOCK") == "1")
            _ = m.detector
            _ = m.vlm
            _STATE.update(cfg=cfg, models=m, th=engine.load_thresholds(os.environ.get("VERIFIED_THRESHOLDS"), cfg))
    return _STATE


def analyze(video_path: str) -> dict:
    s = preload()
    cfg, models, th = s["cfg"], s["models"], s["th"]
    t0 = time.time()
    with _LOCK:                                     # one GPU: videos are processed one after another
        rec = engine.run_video(video_path, models, cfg, cache_dir=None)
    res = engine.result(rec, th, cfg)
    cat = _catalog()
    out = []
    for e in sorted(res["events"], key=lambda e: e["start"]):
        entry = cat.get(e["type"])
        if not entry:
            continue
        shots = evidence.evidence(video_path, e, cfg)
        status = "требуется проверка человеком" if e["status"] == "needs_review" else "подозрение подтверждено проверкой условий"
        legal = entry.get("legal_status", "")
        out.append({
            "type": entry["type"],
            "violation_id": e["type"],
            "timecode_sec": round(float(e["start"]), 2),
            "end_sec": round(float(e["end"]), 2),
            "status": e["status"],
            "score": e["score"],
            "threshold": e["threshold"],
            "description": (f"Визуальное событие ({status}): {e['explanation']} "
                            f"Оценка {e['score']:.2f} при пороге {e['threshold']:.2f}. "
                            + (" ".join(x["note"] for x in shots if x["note"])[:300] + " " if any(x["note"] for x in shots) else "")
                            + "Это подозрение на нарушение по видеоизображению; юридическую оценку выполняет человек"
                            + (f" (статус нормы в каталоге: {legal})." if legal else ".")),
            "evidence_frames": [{"timestamp_sec": x["t"], "image_base64": evidence.to_b64(x["image"])} for x in shots],
            "rule": entry["source"],
        })
    print(f"[verified] {Path(video_path).name}: {len(out)} event(s) in {time.time() - t0:.0f}s {rec['timing']}")
    return {"violations": out}
