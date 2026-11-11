"""
Service door left open: the END state of the door region differs from its START state.

Normal flow (catalog/violations.yaml, normal_routine): whoever enters or leaves closes the door behind them, so the
door region looks at the end of the video as it did at the start. Violation: the door is opened (someone
enters / leaves) and stays open, so the door region keeps looking different until the end.

    rec = run(frames, det)  ->  {"version", "features": {"d_service_door_left_open": {...}}, "detail": {...}}

Measurement (no neural network, CPU, the door region found by OWLv2 in THIS video):
  * gray, blurred door region; frames where a person box overlaps the region are skipped (a person in the
    doorway is not "an open door")
  * reference = median of the first 3 unoccluded frames (door as it was at the start)
  * score = median absolute difference to the reference over the unoccluded frames in the last 40 % of the video,
    divided by 100 (gray levels 0..255 -> about 0..1)
  * onset = first unoccluded frame from which the region stays changed (>= 60 % of the final difference)

Measured on the 40 labelled videos (leave-one-video-out, judge threshold = highest clean value + 0.08):
clean videos 0.07-0.22, X-32 / X-33 (door left open) 0.35; it also fires on X-11 and X-30, where a person leaves
or enters and the door stays open. Assumption: the camera does not move during the video.
"""
from __future__ import annotations

import numpy as np

VERSION = "door-v1"
TAIL = 0.6          # the last 40 % of the video is "the end state"
REF_FRAMES = 3


def _gray(img):
    import cv2
    g = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)


def run(frames, det) -> dict:
    from .objects import Scene
    sc = Scene(det)
    roi = sc.rois.get("door")
    out = {"version": VERSION, "features": {}, "detail": {"roi": roi}}
    feat = {"type": "service_door_left_open", "score": 0.0, "t": float(frames[0].t) if frames else 0.0}
    if roi is None or not frames or sc.n != len(frames):
        out["features"]["d_service_door_left_open"] = feat
        return out
    x0, y0, x1, y1 = [int(round(c)) for c in roi["box"]]
    if x1 - x0 < 8 or y1 - y0 < 8:
        out["features"]["d_service_door_left_open"] = feat
        return out
    occ = []
    for fr in sc.frames:
        occ.append(any(not (p["box"][2] < x0 or p["box"][0] > x1 or p["box"][3] < y0 or p["box"][1] > y1)
                       for p in fr["person"]["high"] + fr["person"]["low"]))
    free = [i for i in range(len(frames)) if not occ[i]]
    if len(free) < REF_FRAMES + 1:
        out["features"]["d_service_door_left_open"] = feat
        return out
    crops = {i: _gray(frames[i].image)[y0:y1, x0:x1] for i in free}
    ref = np.median([crops[i] for i in free[:REF_FRAMES]], axis=0)
    diff = {i: float(np.abs(crops[i] - ref).mean()) for i in free}
    tail = [i for i in free if i >= TAIL * len(frames)]
    if not tail:
        out["features"]["d_service_door_left_open"] = feat
        return out
    level = float(np.median([diff[i] for i in tail]))
    onset = tail[0]
    for i in reversed(free):                       # walk back while the region stays changed
        if diff[i] >= 0.6 * level:
            onset = i
        else:
            break
    score = round(level / 100.0, 4)
    feat.update(score=score, t=float(frames[onset].t), box=[round(c, 1) for c in roi["box"]],
                why=(f"С {frames[onset].t:.0f} с область двери (найдена детектором OWLv2) отличается от начала видео и "
                     f"остаётся такой до конца ({level:.0f} уровней яркости): дверь, по-видимому, оставлена открытой"))
    out["features"]["d_service_door_left_open"] = feat
    out["detail"].update(level=round(level, 2), onset=float(frames[onset].t), free_frames=len(free))
    return out
