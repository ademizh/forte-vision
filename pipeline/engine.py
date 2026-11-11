"""
Engine "verified": one video end to end.

    VIDEO -> decode + metadata -> coarse frames (1 fps) -> Grounding DINO (+ tiles) -> ByteTrack
    -> zones (cashier / desk / safe / door / screen, found per video) -> temporal states (TRUE/FALSE/UNKNOWN)
    -> rules -> candidates -> adaptive sampling (focused frames, region close-ups, fallback windows)
    -> SmolVLM2 VERA-style questions -> fusion -> (thresholds: calibration.py) -> evidence

    models = Models(cfg)                 # loads Grounding DINO and SmolVLM2 once, lazily
    rec = run_video(path, models, cfg, cache_dir="eval/cache_verified")

The record is JSON-serialisable (one JSONL line per video) and holds every candidate with its conditions,
VLM answers, fused score and status, plus timings and GPU memory; thresholds are applied afterwards
(calibration.decide), so evaluation and calibration never need the GPU again.

Caches (development only, under cache_dir): detections and VLM answers, keyed by the video identity
(name + size + mtime), the model path and a hash of the config sections the stage depends on
(config.stage_key). A changed parameter or model gives a new key; nothing from the legacy Qwen / OWLv2
engine is ever read. With cache_dir=None (the service) nothing is read or written.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import calibration, config, detections, fusion, questions, rules, sampler, temporal, tracking, video, zones
from .frames import Frame

ENGINE = "verified"


class Models:
    def __init__(self, cfg: dict, mock: bool = False, detector=None, vlm=None):
        self.cfg, self.mock = cfg, mock
        self._det, self._vlm = detector, vlm

    @property
    def detector(self):
        if self._det is None:
            if self.mock:
                from .mock import MockDetector
                self._det = MockDetector("dino")
            else:
                dp = Path(self.cfg["models"]["detector_path"])
                if not (dp / "config.json").exists():
                    raise FileNotFoundError(f"Grounding DINO folder not found: {dp}  (download "
                                            f"IDEA-Research/grounding-dino-tiny; set models.detector_path or DINO_PATH)")
                from .detect import Detector
                self._det = Detector(str(dp), name="dino")
                if self._det.kind != "grounding-dino":
                    raise ValueError(f"{self.cfg['models']['detector_path']} is {self._det.kind}, expected Grounding DINO")
        return self._det

    @property
    def vlm(self):
        if self._vlm is None:
            if self.mock:
                from .smolvlm import MockVLM
                self._vlm = MockVLM()
            else:
                from .smolvlm import SmolVLM
                m, v = self.cfg["models"], self.cfg["vlm"]
                self._vlm = SmolVLM(m["vlm_path"], m.get("device", "cuda"), m.get("dtype", "bfloat16"),
                                    int(v["longest_edge"]), bool(v["image_splitting"]), bool(v.get("both_orders", True)))
        return self._vlm

    def names(self) -> dict:
        return {"detector": "IDEA-Research/grounding-dino-tiny" if not self.mock else "mock",
                "detector_path": self.cfg["models"]["detector_path"],
                "vlm": "HuggingFaceTB/SmolVLM2-2.2B-Instruct" if not self.mock else "mock",
                "vlm_path": self.cfg["models"]["vlm_path"],
                "tracker": f"ByteTrack ({tracking.backend(self.cfg)})"}


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if p and Path(p).exists() else None


def _save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _crop(img, box, scale: float, pad: float = 0.3):
    w, h = img.size
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    b = (max(0, int(x0 - dx)), max(0, int(y0 - dy)), min(w, int(x1 + dx)), min(h, int(y1 + dy)))
    if b[2] - b[0] < 4 or b[3] - b[1] < 4:
        return img
    c = img.crop(b)
    if scale and scale != 1.0:
        c = c.resize((max(1, int(c.width * scale)), max(1, int(c.height * scale))))
    return c


def _fetch(path, times, cfg) -> dict:
    """Decode only the windows the VLM needs, at focus_fps; return {time: Frame} (nearest decoded frame)."""
    if not times:
        return {}
    fps = float(cfg["sampling"]["focus_fps"])
    mw = int(cfg["sampling"]["max_width"])
    ts = sorted(set(times))
    windows, cur = [], [ts[0], ts[0]]
    for t in ts[1:]:
        if t - cur[1] <= 4.0:
            cur[1] = t
        else:
            windows.append(cur)
            cur = [t, t]
    windows.append(cur)
    decoded = []
    for a, b in windows:
        decoded += video.sample(path, fps=max(fps, 2.0), start=max(0.0, a - 0.01), end=b + 0.6, max_width=mw)
    out = {}
    for t in ts:
        if decoded:
            out[t] = min(decoded, key=lambda f: abs(f.t - t))
    return out


def run_video(path: str, models: Models, cfg: dict, cache_dir: str | None = None, overwrite: bool = False,
              max_seconds: float | None = None, log=print) -> dict:
    t_start = time.time()
    path = str(path)
    name = Path(path).stem
    tm = {}
    meta = video.metadata(path)
    vid = video.identity(path)
    max_seconds = float(max_seconds or cfg["decision"].get("max_seconds", 280))

    t0 = time.time()
    frames = video.sample(path, fps=float(cfg["sampling"]["coarse_fps"]), max_width=int(cfg["sampling"]["max_width"]))
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    tm["decode"] = round(time.time() - t0, 2)
    size = frames[0].image.size

    # ---- stage 1: Grounding DINO (+ tiles)
    t0 = time.time()
    dkey = config.stage_key(cfg, "detection")
    dpath = Path(cache_dir) / "det" / f"{name}_{vid}_{dkey}.json" if cache_dir else None
    fdets = None if overwrite else _load(dpath)
    if fdets is None or len(fdets) != len(frames):
        fdets = detections.detect_frames(frames, models.detector, cfg)
        if dpath:
            _save(dpath, fdets)
        tm["detection"] = round(time.time() - t0, 2)
    else:
        tm["detection_cached"] = round(time.time() - t0, 2)

    # ---- tracking, zones, temporal states, rules
    t0 = time.time()
    tracks = tracking.track_all(fdets, cfg)
    tm["tracking"] = round(time.time() - t0, 2)
    t0 = time.time()
    zs = zones.find_zones(fdets, tracks, cfg, size)
    st = temporal.build_states(fdets, tracks, zs, cfg)
    tm["temporal"] = round(time.time() - t0, 2)
    t0 = time.time()
    cands = rules.evaluate(st, tracks, zs, cfg)
    cands += sampler.fallback_candidates(cands, meta["duration"], cfg)
    tm["rules"] = round(time.time() - t0, 2)

    # ---- stages 3-4: adaptive sampling + SmolVLM2
    t0 = time.time()
    jobs, skipped = sampler.plan(cands, meta["duration"], cfg) if cfg["vlm"].get("enabled", True) else ([], [])
    vkey = config.stage_key(cfg, "vlm")
    vpath = Path(cache_dir) / "vlm" / f"{name}_{vid}_{vkey}.json" if cache_dir else None
    vcache = ({} if overwrite else (_load(vpath) or {})) if vpath else {}
    need = [j for j in jobs if _sig(j) not in vcache]
    fetched = _fetch(path, [t for j in need for t in j["times"]], cfg) if need else {}
    answers = []
    scale = float(cfg["vlm"].get("crop_scale", 2.0))
    for j in jobs:
        sig = _sig(j)
        if sig in vcache:
            answers.append(vcache[sig])
            continue
        if time.time() - t_start > 0.85 * max_seconds:
            skipped.append({"cid": j["cid"], "qid": j["qid"], "times": j["times"], "reason": "time budget"})
            answers.append(None)
            continue
        fr = [fetched[t] for t in j["times"] if t in fetched]
        if isinstance(j.get("crop"), list):
            fr = [Frame(f.t, _crop(f.image, j["crop"], scale)) for f in fr]
        try:
            p = models.vlm.choice(fr, questions.text(j["qid"]), questions.ANSWERS)
        except RuntimeError as e:                                # OOM on one image etc.: reported, not hidden
            skipped.append({"cid": j["cid"], "qid": j["qid"], "times": j["times"], "reason": f"vlm error: {e}"})
            answers.append(None)
            continue
        p = [round(float(x), 4) for x in p]
        vcache[sig] = p
        answers.append(p)
    if vpath and need:
        _save(vpath, vcache)
    keep = [(j, a) for j, a in zip(jobs, answers) if a is not None]
    tm["vlm"] = round(time.time() - t0, 2)

    t0 = time.time()
    fused = fusion.fuse(cands, [j for j, _ in keep], [a for _, a in keep], cfg)
    tm["fusion"] = round(time.time() - t0, 2)
    tm["total"] = round(time.time() - t_start, 2)
    mem = models.vlm.memory() if models._vlm is not None else {}
    return {
        "video": name, "file": Path(path).name, "engine": ENGINE, "config_key": vkey,
        "question_version": questions.QUESTION_VERSION, "models": models.names(),
        "duration": round(meta["duration"], 2), "fps": meta["fps"], "size": list(size), "frames": len(frames),
        "zones": {k: v for k, v in zs.items() if k != "roles"}, "roles": zs["roles"],
        "tracks": {k: [{kk: vv for kk, vv in t.items() if kk != "obs"} for t in v] for k, v in tracks.items()},
        "states": {k: v.to_json() for k, v in st.items() if isinstance(v, temporal.Signal)},
        "times": [round(float(x), 2) for x in st["t"]],
        "candidates": fused, "vlm_jobs": len(jobs), "vlm_skipped": skipped,
        "vlm_calls": getattr(models._vlm, "calls", 0) if models._vlm is not None else 0,
        "timing": tm, "gpu": mem,
    }


def _sig(j) -> str:
    crop = j.get("crop")
    crop = ",".join(f"{x:.0f}" for x in crop) if isinstance(crop, list) else str(crop)
    return f"{j['qid']}|{','.join(f'{t:.2f}' for t in j['times'])}|{crop}"


def result(rec: dict, th: dict, cfg: dict) -> dict:
    """Video-level decision with frozen thresholds."""
    events = calibration.decide(rec, th, cfg)
    return {"video": rec["video"], "violation": bool(events), "type": events[0]["type"] if events else "",
            "start": events[0]["start"] if events else None, "end": events[0]["end"] if events else None,
            "score": events[0]["score"] if events else 0.0, "threshold": events[0]["threshold"] if events else None,
            "status": events[0]["status"] if events else "no_violation", "events": events}


def load_thresholds(path: str | Path | None, cfg: dict) -> dict:
    p = Path(path) if path else config.ROOT / "configs" / "thresholds_verified.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"global": float(cfg["decision"]["default_threshold"]), "per_type": {},
            "warning": "not calibrated: default_threshold from configs/verified.yaml"}
