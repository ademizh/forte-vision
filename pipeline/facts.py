"""
Per-frame facts: each fact in vlm.FACTS is asked about ONE frame as "A. Yes / B. No" and scored by the
probability of "A" (one forward pass, no generated text, no JSON, no "null").

extract_facts(video) -> {"video", "fps", "duration", "t": [...], "p": {fact: [P(yes) per frame]}}, frames

The result is small JSON, so it is computed once per video (run_vlm.py facts-dataset) and the rules in
rules.py can be tuned offline in seconds, without the GPU.
"""
import os
import time
from pathlib import Path

from . import detect
from .frames import sample_frames, video_info
from .vlm import FACTS, SYSTEM, fact_prompt, get_vlm

FACT_FPS = float(os.environ.get("FACT_FPS", "1"))
FACT_BATCH = int(os.environ.get("FACT_BATCH", "8"))     # frames per batch; all questions reuse them
USE_DET = os.environ.get("USE_DET", "1") == "1"         # run the detector too when DET_PATH exists
FACT_WIDTH = int(os.environ.get("FACT_WIDTH", "1280"))   # keep small objects (cash, keys) visible


def extract_facts(video_path: str, fps: float = FACT_FPS, vlm=None, keys: list[str] | None = None):
    t0 = time.time()
    vlm = vlm or get_vlm()
    keys = keys or list(FACTS)
    info = video_info(video_path)
    frames = sample_frames(video_path, fps=fps, max_width=FACT_WIDTH)
    probs = []
    for s in range(0, len(frames), FACT_BATCH):
        chunk = frames[s:s + FACT_BATCH]
        probs += vlm.frame_question_probs(SYSTEM, chunk, [fact_prompt(FACTS[k]) for k in keys], 2)
    p = {k: [round(float(probs[i][j][0]), 4) for i in range(len(frames))] for j, k in enumerate(keys)}
    facts = {"video": Path(video_path).name, "fps": fps, "duration": info["duration"],
             "t": [f.t for f in frames], "p": p}
    if USE_DET:                                 # second opinions: Grounding DINO (dino_*) and OWLv2 (owl_*)
        for name in detect.available():
            detect.add_to_facts(facts, frames, name=name)
    facts["sec_total"] = round(time.time() - t0, 1)
    return facts, frames
