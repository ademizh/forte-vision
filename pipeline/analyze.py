"""
Service entry point: analyze(video_path) -> {"violations": [...]} in the exact shape app/main.py returns.

Flow (pipeline/flow.py): OWLv2 (+ SAHI tiles) -> Qwen facts -> scene/state -> observations -> local norm search
-> condition chains -> judge. Thresholds: catalog/judge_thresholds.json (python forte.py accuracy --write).
"""
import base64
import io
import os
import time
from pathlib import Path

import yaml

from .vlm import OBSERVE_PROMPT, SYSTEM

CATALOG_PATH = Path(os.environ.get("CATALOG", Path(__file__).parent.parent / "catalog" / "violations.yaml"))
DESCRIBE = os.environ.get("STATE_DESCRIBE", "0") == "1"      # optional visible-only description
CHAINS = os.environ.get("CHAINS", "auto")                    # auto = only if calibration switched one on | on | off
_MODELS = None
SRC = {"engine": "Qwen (факты по кадрам) + OWLv2", "chain": "OWLv2 + Qwen: пошаговая проверка условий найденной нормы",
       "change": "анализ изменений изображения (вычитание фона) + OWLv2 + Qwen",
       "vlm": "Qwen: классификация всей последовательности действий с данными OWLv2 и ключевыми кадрами",
       "door": "сравнение конечного и начального состояния области двери (OWLv2 + анализ изображения)"}


def load_catalog() -> dict:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        cat = yaml.safe_load(f)
    cat["by_id"] = {v["id"]: v for v in cat["violations"]}
    if isinstance(cat.get("other"), dict):
        cat["by_id"]["other"] = cat["other"]
    return cat


def _jpeg_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _draw_box(image, box):
    if not box:
        return image
    from PIL import ImageDraw
    im = image.copy()
    ImageDraw.Draw(im).rectangle([int(v) for v in box], outline="#ff3b30", width=3)
    return im


def _pick_evidence(frames, e) -> list:
    """1-3 frames: start of the event, middle, strongest moment (or +-1 s around it)."""
    want = sorted(set(e.get("evidence_times") or [e.get("start", e["timecode_sec"]),
                   e["timecode_sec"], e.get("end", e["timecode_sec"])]))[:3]
    if len(want) == 1:
        want = [want[0] - 1, want[0], want[0] + 1]
    picks = []
    for t in want:
        f = min(frames, key=lambda fr: abs(fr.t - t))
        if f not in picks:
            picks.append(f)
    return picks[:3]


def analyze(video_path: str) -> dict:
    global _MODELS
    from . import judge
    from .flow import Models, process
    t0 = time.time()
    cal = judge.load()
    if not cal:
        raise RuntimeError("catalog/judge_thresholds.json missing: run  python forte.py accuracy --write")
    _MODELS = _MODELS or Models()
    run_chains = CHAINS == "on" or (CHAINS == "auto" and judge.chains_needed(cal))
    run_focus = judge.needed(cal, "focus")
    run_vlm = judge.needed(cal, "vlm") or run_focus
    cat = load_catalog()
    r = process(video_path, _MODELS, cache=None, run_chains=run_chains, run_vlm=run_vlm,
                run_change=judge.needed(cal, "change"), run_focus=run_focus)
    frames = r["frames"]
    events = judge.decide(r["rec"]["features"], r["facts"], cal)
    print(f"[forte] {Path(video_path).name}: {len(events)} event(s) in {time.time() - t0:.0f}s {r['rec']['timing']}")
    violations = []
    for e in sorted(events, key=lambda e: e["timecode_sec"]):
        entry = cat["by_id"].get(e["id"]) or cat["by_id"].get("other") or \
            {"type": "Нетипичное действие с наличными или ценностями", "source": {"document": "", "clause": ""}}
        picks = _pick_evidence(frames, e) or [min(frames, key=lambda f: abs(f.t - e["timecode_sec"]))]  # API: 1-3 frames
        observation = ""
        if DESCRIBE and not _MODELS.mock:
            i = frames.index(min(frames, key=lambda f: abs(f.t - e["timecode_sec"])))
            context = [frames[j] for j in range(max(i - 1, 0), min(i + 3, len(frames)))]
            observation = _MODELS.vlm.generate([(SYSTEM, context, OBSERVE_PROMPT)], max_new_tokens=80)[0].strip()
        norm = e.get("norm") or {}
        norm_txt = (f"Норма: {norm['document']}, {norm['clause']}: «{norm['text']}»"
                    + (" (изложение)" if norm.get("text_status") == "paraphrase" else "")) if norm else ""
        violations.append({
            "type": entry["type"],
            "timecode_sec": round(float(e["timecode_sec"]), 2),
            "description": " ".join(x for x in (
                e.get("why", ""),
                f"На кадрах: {observation}" if observation else "",
                norm_txt,
                "Требуется проверка человеком.",
            ) if x),
            "evidence_frames": [{"timestamp_sec": round(f.t, 2), "image_base64": _jpeg_b64(_draw_box(
                f.image, e.get("box") if "box_timecode" not in e or abs(f.t-e["box_timecode"])<0.6 else None))}
                                for f in picks],
            "rule": entry["source"],                       # document + clause from the catalog, never from the VLM
        })
    return {"violations": violations}
