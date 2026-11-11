"""
One video end to end (used by forte.py and by the service):

  video -> 1 fps frames (native resolution)
   [1] OWLv2: full frame + SAHI tiles for small objects            cache: <cache>/det/<video>.json
   [2] Qwen per-frame facts (14 yes/no questions)                    cache: <cache>/facts/<video>.json
       + OWLv2 top scores -> scene (regions, ByteTrack-style tracks, people) + state over time
   [3] observations: timeline of plain facts (observe.py)
   [4] local norm search over norms/ (norms.py) -> rules worth checking
   [5] condition chains: Qwen checks every condition of every found rule (chain.py)   cache: <cache>/chain/
   [5c] video classifier: Qwen gets the whole action sequence + OWLv2 data + key frames (classify.py)
                                                                                     cache: <cache>/vlm/
   [5b] pixel change: lasting changes (safe zone, new objects) + hand motion (change.py)  cache: <cache>/change/
   [5e] door: end state of the door region vs its start state (door_state.py, CPU)   cache: <cache>/door/
   [6] judge: state-engine evidence + chains + change vs calibrated thresholds (judge.py)  cache: <cache>/features/

Every stage is cached; a re-run continues where it stopped. --force det,facts,chain,change,vlm recomputes a stage.
"""
from __future__ import annotations

import json
import hashlib
import os
import time
from pathlib import Path

from . import chain, change, classify, detect, door_state, judge, objects, focus
from .frames import sample_frames, video_info
from .observe import observe
from .state_engine import State
from .vlm import FACTS, SYSTEM, fact_prompt

ROOT = Path(__file__).resolve().parent.parent
MOCK = os.environ.get("FORTE_MOCK", "0") == "1"
FACT_FPS = float(os.environ.get("FACT_FPS", "1"))
FACT_WIDTH = int(os.environ.get("FACT_WIDTH", "1280"))      # 464x848 videos are never downscaled
FACT_BATCH = int(os.environ.get("FACT_BATCH", "8"))
TILE_EVERY = int(os.environ.get("TILE_EVERY", "2"))          # SAHI tiles every N sampled frames (speed)


class Models:
    """Qwen and the detector, loaded only when a stage needs them."""

    def __init__(self, mock: bool = MOCK):
        self.mock = mock
        self._vlm = None
        self._det = None

    @property
    def vlm(self):
        if self._vlm is None:
            if self.mock:
                from .mock import MockVLM
                self._vlm = MockVLM()
            else:
                from .vlm import get_vlm
                self._vlm = get_vlm()
        return self._vlm

    @property
    def detectors(self) -> dict:
        if self._det is None:
            if self.mock:
                from .mock import MockDetector
                self._det = {n: MockDetector(n) for n in detect.wanted()}
            else:
                self._det = {n: detect.get_detector(n) for n in detect.available()}
                if not self._det:
                    print("[flow] WARNING: no detector folder found (OWL_PATH): Qwen facts only")
        return self._det

    def release_detectors(self):
        """Free the detector's GPU memory before Qwen runs (on Windows a full GPU silently spills into system
        RAM and everything becomes ~100x slower instead of failing)."""
        if self._det is None or self.mock:
            return
        import gc
        self._det = None
        with detect._lock:
            detect._instances.clear()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _load(p):
    if not p or not Path(p).exists():
        return None
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None  # interrupted cache writes are recomputed, never treated as a clean result


def _save(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def frames_of(video_path: str):
    return sample_frames(video_path, fps=FACT_FPS, max_width=FACT_WIDTH)


def _cache_identity(video_path: str, cache: Path | None, mock=False):
    """Never reuse another video's observations merely because its name matches."""
    if cache is None:
        return None
    p = Path(video_path)
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    identity = {"sha256": digest.hexdigest(), "fact_fps": FACT_FPS,
                "fact_width": FACT_WIDTH, "tile_every": TILE_EVERY,
                "mock": bool(mock),
                "inference_signature": judge.inference_signature(include_focus=False),
                "state_version": __import__("pipeline.state_engine", fromlist=["VERSION"]).VERSION,
                "detectors": list(detect.wanted()), "fact_keys": sorted(FACTS)}
    manifest = Path(cache) / "identity" / f"{p.stem}.json"
    old = _load(manifest)

    def core(x):    # the state-engine version only changes the judge stage (pure numpy on cached facts)
        if not isinstance(x, dict):
            return x
        y = {k: v for k, v in x.items() if k != "state_version"}
        if isinstance(y.get("inference_signature"), dict):
            y["inference_signature"] = {k: v for k, v in y["inference_signature"].items() if k != "state_version"}
        return y

    if old != identity:
        if core(old) == core(identity):          # only the state engine changed: keep OWL, Qwen, chains, focus
            (Path(cache) / "features" / f"{p.stem}.json").unlink(missing_ok=True)
        else:
            for stage in ("det", "facts", "chain", "vlm", "change", "focus", "door", "features"):
                (Path(cache) / stage / f"{p.stem}.json").unlink(missing_ok=True)
        _save(manifest, identity)
    return identity


def _det_ok(det, n):
    return det is not None and len(det.get("t", [])) == n and "tiles" in det and \
        all(d in det.get("dets", {}) for d in detect.wanted())


def detect_stage(video_path: str, models: Models, cache: Path, force: set = frozenset()) -> tuple[bool, float]:
    """Stage [1] only (Qwen not on the GPU). Returns (computed?, seconds)."""
    t0 = time.time()
    _cache_identity(video_path, cache, models.mock)
    frames = frames_of(video_path)
    p = Path(cache) / "det" / f"{Path(video_path).stem}.json"
    if "det" not in force and _det_ok(_load(p), len(frames)):
        return False, time.time() - t0
    _save(p, objects.run_detectors(frames, models.detectors, TILE_EVERY))
    return True, time.time() - t0


def qwen_facts(frames, vlm) -> dict:
    keys = list(FACTS)
    probs = []
    for s in range(0, len(frames), FACT_BATCH):
        probs += vlm.frame_question_probs(SYSTEM, frames[s:s + FACT_BATCH], [fact_prompt(FACTS[k]) for k in keys], 2)
    return {k: [round(float(probs[i][j][0]), 4) for i in range(len(frames))] for j, k in enumerate(keys)}


def expand_force(force):
    force = set(force)
    if "det" in force:
        force.update(("chain", "vlm", "focus"))
    if "facts" in force:
        force.update(("chain", "vlm", "focus"))
    if "vlm" in force:
        force.add("focus")
    return force


def process(video_path: str, models: Models, cache: Path | None = None, force: set = frozenset(),
            facts_dirs=(ROOT / "eval" / "facts",), run_chains: bool = True, run_vlm: bool = True,
            log=print, run_change: bool = True, run_focus: bool = False) -> dict:
    """Runs (or loads) every stage for one video. cache=None: nothing is read or written (service)."""
    t0 = time.time()
    force = expand_force(force)
    run_vlm = run_vlm or run_focus  # focus budget never replaces whole-timeline coverage
    _cache_identity(video_path, cache, models.mock)
    name = Path(video_path).stem
    info = video_info(video_path)
    frames = frames_of(video_path)
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
    n = len(frames)
    timing = {"frames": round(time.time() - t0, 1)}
    c = (lambda sub: Path(cache) / sub / f"{name}.json") if cache else (lambda sub: None)

    # [1] OWLv2 (+ SAHI tiles)
    t1 = time.time()
    det = _load(c("det")) if "det" not in force else None
    if not _det_ok(det, n):
        det = objects.run_detectors(frames, models.detectors, TILE_EVERY)
        if cache:
            _save(c("det"), det)
        timing["owl"] = round(time.time() - t1, 1)

    # [2] Qwen facts (an earlier facts file made from the same frames is reused)
    t1 = time.time()
    facts = _load(c("facts")) if "facts" not in force else None
    if facts is None or len(facts.get("t", [])) != n or not all(k in facts["p"] for k in FACTS):
        qp = None
        # Inference (cache=None) must NEVER read training facts by basename.
        # Older unbound facts also have no content identity and cannot be reused.
        if cache is not None and "facts" not in force:
            for d in facts_dirs:
                old = _load(Path(d) / f"{name}.json")
                identity = _load(Path(cache) / "identity" / f"{name}.json")
                if old and old.get("source_sha256") == (identity or {}).get("sha256") and \
                        len(old.get("t", [])) == n and all(k in old["p"] for k in FACTS):
                    qp = {k: old["p"][k] for k in FACTS}
                    break
        if qp is None:
            qp = qwen_facts(frames, models.vlm)
            timing["qwen_facts"] = round(time.time() - t1, 1)
        facts = {"video": Path(video_path).name, "fps": FACT_FPS, "duration": info["duration"],
                 "t": [f.t for f in frames], "p": qp}
    facts["p"] = {k: v for k, v in facts["p"].items() if k in FACTS}
    facts["p"].update(objects.legacy_scores(det))
    facts["detectors"] = sorted(det.get("dets", {}))
    if cache:
        _save(c("facts"), facts)

    # scene + state + [3] observations
    scene = objects.Scene(det)
    st = State(facts)
    obs = observe(st, scene)

    # [4] norm search + [5] condition chains
    t1 = time.time()
    ch = _load(c("chain")) if "chain" not in force else None
    if not run_chains:
        ch = {"version": "skipped", "features": {}, "chains": [], "retrieved": [], "checked": []}
    elif ch is None or ch.get("version") != chain.VERSION:
        ch = chain.run(st, scene, obs, frames, models.vlm)
        if cache:
            _save(c("chain"), ch)
        timing["chains"] = round(time.time() - t1, 1)

    # [5c] video classifier (Qwen: whole action sequence + detections + key frames)
    t1 = time.time()
    vc = _load(c("vlm")) if "vlm" not in force else None
    if not run_vlm:
        vc = {"version": "skipped", "features": {}}
    elif vc is None or vc.get("version") != classify.VERSION:
        vc = classify.run(st, scene, obs, frames, models.vlm)
        if cache:
            _save(c("vlm"), vc)
        timing["classify"] = round(time.time() - t1, 1)

    # [5d] independent low-threshold proposals + descriptive crop states.
    t1 = time.time()
    fc = _load(c("focus")) if "focus" not in force else None
    if not run_focus:
        fc = {"version": "skipped", "features": {}, "coverage": {"checked": 0}, "n_questions": 0}
    elif fc is None or fc.get("version") != focus.VERSION or fc.get("signature") != focus.signature():
        fc = focus.run(st, scene, det, frames, models.vlm, vc)
        if cache:
            _save(c("focus"), fc)
        timing["focus"] = round(time.time() - t1, 1)

    # [5b] pixel change (CPU, no model)
    t1 = time.time()
    chg = _load(c("change")) if "change" not in force else None
    if not run_change:
        chg = {"version": "skipped", "features": {}, "n": n}
    elif chg is None or chg.get("version") != change.VERSION or chg.get("n") != n:
        cf = sample_frames(video_path, fps=change.CHANGE_FPS, max_width=FACT_WIDTH)
        chg = {**change.run(cf, scene, st), "n": n}
        if cache:
            _save(c("change"), chg)
        timing["change"] = round(time.time() - t1, 1)

    # [5e] door end state vs start state (CPU, no model): service door left open
    t1 = time.time()
    dr = _load(c("door")) if "door" not in force else None
    if dr is None or dr.get("version") != door_state.VERSION:
        try:
            dr = door_state.run(frames, det)
        except Exception as e:  # noqa: BLE001  a failed pixel check must never stop the video
            dr = {"version": door_state.VERSION, "features": {}, "error": str(e)}
        if cache:
            _save(c("door"), dr)
        timing["door"] = round(time.time() - t1, 1)

    # [6] evidence for the judge
    feats = judge.video_features(facts, ch, chg, vc, fc, dr)
    rec = {"video": name, "file": Path(video_path).name, "duration": info["duration"], "frames": n,
           "mock": bool(models.mock),
           "state_version": __import__("pipeline.state_engine", fromlist=["VERSION"]).VERSION,
           "classifier_version": vc.get("version"),
           "focus_version": fc.get("version"), "focus_coverage": fc.get("coverage"),
           "inference_signature": judge.inference_signature(),
           "features": feats, "scene": scene.summary(), "observations": obs["events"],
           "retrieved": ch.get("retrieved", []), "checked": ch.get("checked", []),
           "n_questions": ch.get("n_questions", 0) + vc.get("n_questions", 0),
           "focus_state_queries": fc.get("n_questions", 0),
           "p_violation": vc.get("p_violation"), "timing": {**timing, "total": round(time.time() - t0, 1)}}
    if cache:
        _save(c("features"), rec)
    return {"rec": rec, "facts": facts, "det": det, "chain": ch, "change": chg, "vlm": vc, "focus": fc, "door": dr, "frames": frames,
            "scene": scene,
            "state": st, "obs": obs}


def load_cached(cache: Path, name: str):
    return _load(Path(cache) / "features" / f"{name}.json"), _load(Path(cache) / "facts" / f"{name}.json")
