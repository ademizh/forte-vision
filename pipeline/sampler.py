"""
Adaptive temporal sampling (idea of the anomaly-focused temporal sampler of Holmes-VAU, CVPR 2025, adapted to
our components; not a reproduction of that model):

  stage 1  coarse frames at sampling.coarse_fps -> Grounding DINO -> temporal states (engine.py)
  stage 2  rules turn state transitions into scored candidate intervals (rules.py)
  stage 3  for every candidate: denser frames (sampling.focus_fps) from focus_pre_sec BEFORE the transition (its
           cause) to focus_post_sec after it; higher rule scores are served first
  stage 4  only those frames (or region crops) go to SmolVLM2, max vlm.max_frames per question
  stage 5  fallback: when the detector produced no departure / visit at all, sampling.fallback_windows periodic
           windows get a generic question, so a detector failure does not mean "no violation"

    jobs, skipped = plan(cands, duration, cfg)
      job = {"cid", "qid", "role", "times": [...], "crop": box | "frame" | None}
Budget: vlm.budget_calls forward passes per video (a question costs 2 with both_orders). Questions that do not
fit are returned in `skipped` and reported (never silently dropped).
"""
from __future__ import annotations

import numpy as np

from . import questions as Q


def _dense(times, focus, cfg, duration):
    sc = cfg["sampling"]
    fps = float(sc["focus_fps"])
    out = set(round(t, 2) for t in times)
    for f in focus:
        a, b = max(0.0, f - float(sc["focus_pre_sec"])), min(duration, f + float(sc["focus_post_sec"]))
        out |= {round(float(x), 2) for x in np.arange(a, b + 1e-6, 1.0 / fps)}
    out = sorted(out)
    k = int(cfg["vlm"]["max_frames"])
    if len(out) > k:                                          # keep the requested times, spread the rest
        keep = sorted(set(round(t, 2) for t in times))[:k]
        rest = [t for t in out if t not in keep]
        need = k - len(keep)
        if need > 0 and rest:
            idx = np.unique(np.linspace(0, len(rest) - 1, num=need).round().astype(int))
            keep += [rest[i] for i in idx]
        out = sorted(keep)[:k]
    return out


def fallback_candidates(cands, duration, cfg) -> list[dict]:
    if any(c["family"] in ("departure", "visit") for c in cands):
        return []
    n = int(cfg["sampling"].get("fallback_windows", 0))
    if n <= 0 or duration <= 0:
        return []
    out = []
    for k, t in enumerate(np.linspace(duration / (2 * n), duration - duration / (2 * n), num=n)):
        times = [max(0.0, t - 2), t, min(duration, t + 2)]
        out.append({"cid": f"f{k:02d}", "type": "other", "start": round(float(times[0]), 2),
                    "end": round(float(times[-1]), 2), "family": "fallback", "internal": False,
                    "conditions": [{"name": "detector_found_no_event", "value": -1, "conf": 0.5,
                                    "source": "detector"}],
                    "rule_score": 0.5, "boxes": {}, "focus": [round(float(t), 2)],
                    "questions": [Q.spec("q_fallback", "mandatory", times)]})
    return out


def plan(cands: list[dict], duration: float, cfg: dict) -> tuple[list[dict], list[dict]]:
    budget = int(cfg["vlm"]["budget_calls"])
    per_q = 2 if cfg["vlm"].get("both_orders", True) else 1
    # detector-driven events first; periodic checks, VLM-only types and fallback windows use what is left
    weight = {"periodic": 0.6, "vlm_only": 0.5, "fallback": 0.4}
    order = sorted(cands, key=lambda c: (-c["rule_score"] * weight.get(c.get("family"), 1.0), c["start"]))
    jobs, skipped, used = [], [], 0
    for c in order:
        for q in c["questions"]:
            # periodic questions (door) are asked per sample time, not as one clip
            if c["family"] == "periodic":
                groups = [[t] for t in q["times"]]
            elif q.get("crop") not in (None, "frame"):
                groups = [q["times"][: int(cfg["vlm"]["max_frames"])]]
            else:
                groups = [_dense(q["times"], c.get("focus", []), cfg, duration)]
            for g in groups:
                if used + per_q > budget:
                    skipped.append({"cid": c["cid"], "qid": q["id"], "times": g, "reason": "budget"})
                    continue
                jobs.append({"cid": c["cid"], "qid": q["id"], "role": q["role"], "times": g, "crop": q.get("crop")})
                used += per_q
    return jobs, skipped
