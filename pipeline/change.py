"""
Pixel-change evidence (no neural network): what CHANGED in the scene and how long it STAYED changed.

Why: at 464x848 Qwen cannot tell an open safe, a bag or a phone on the desk, or cash being counted. A change of
pixels is still measurable: an opened safe door, a bag put down or an item left on the desk is a region that
looks different from before and stays so; counting is sustained small motion of a person standing still.

    run(video_path, scene, state) -> {"version", "t", "features": {c_...: {...}}, "components": [...]}

1. frames at CHANGE_FPS (2), gray, blurred, brightness-normalised (minus the frame median: light changes)
2. reference = how every pixel looked the first time it was seen free (no person box from OWLv2, no motion)
   and STABLE for REF_SEC (2 s); pixels covered by the cashier at the start get their reference after the
   cashier has left (a person walking away never becomes the reference)
3. changed = differs from the reference; a pixel must stay changed HOLD_SEC (6 s) to count. While a person
   stands in front of it the counter is held, not reset (somebody walking past an open safe)
4. changed pixels -> blobs (connected components); blob score grows with area (log) and with how long it stayed
5. regions come from the scene (OWLv2 found them automatically in this video): blob inside the safe region ->
   "safe" zone, anywhere else -> "desk/room"

Evidence (per second, aligned with the Qwen facts; the video score = its maximum):
    c_safe     a lasting change in the safe region while the cashier is away     -> safe_left_open
    c_object   a new lasting object/blob outside the safe region                 -> personal_items_cash_area
    c_left     the same while the cashier is away                                -> collection_bag_unsecured
    c_hands    a person standing still with sustained small motion in the box,   -> counting_without_customer
               no client at the desk

These signals do not know WHAT changed, so their type is a guess from the zone; for calibration they are
type-agnostic ("generic"): the judge switches one on only if, leave-one-video-out, it fires on NO clean video
and catches at least one violation of any type (judge.calibrate).
"""
from __future__ import annotations

import os

import numpy as np

VERSION = "change-v1"
CHANGE_FPS = float(os.environ.get("CHANGE_FPS", "2"))
REF_PIX = 22.0            # gray-level difference to the reference that counts as changed (0-255, after blur)
MOVE_PIX = 14.0           # frame-to-frame difference that counts as motion
HOLD_SEC = 6.0            # a pixel must stay changed this long
REF_SEC = 2.0             # a pixel must be free and stable this long before its reference is taken
BASE_AREA = 464 * 848     # areas below are for this frame size and scaled to the real one
MIN_AREA = 120            # px: smaller blobs are noise (a phone is ~150-300 px at 464x848)
BIG_AREA = 8000           # px: a blob of this size gets the full area score
PERSON_PAD = 0.15         # person boxes are enlarged by this fraction (arms, shadow)

TYPES = {"c_safe": "safe_left_open", "c_object": "personal_items_cash_area",
         "c_left": "collection_bag_unsecured", "c_hands": "counting_without_customer"}
WHY = {
    "c_safe": "в зоне сейфа изображение изменилось и остаётся изменённым {hold:.0f} с, кассира на месте нет",
    "c_object": "на столе/в помещении появился новый предмет (область {area} пикс.) и лежит {hold:.0f} с",
    "c_left": "появился новый предмет (область {area} пикс.), он остаётся {hold:.0f} с, кассира на месте нет",
    "c_hands": "человек стоит на месте и {hold:.0f} с делает мелкие движения руками (похоже на пересчёт), клиента нет",
}


def _gray(img) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return g - np.median(g)


def _pad(box, w, h, pad=PERSON_PAD):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return [max(0, int(x0 - dx)), max(0, int(y0 - dy)), min(w, int(x1 + dx) + 1), min(h, int(y1 + dy) + 1)]


def _persons(scene, i) -> list:
    if scene is None or not getattr(scene, "n", 0):
        return []
    fr = scene.frames[i]
    return [x["box"] for x in fr["person"]["high"] + fr["person"]["low"]]


def _inside(pt, box, pad=0.2) -> bool:
    if not box:
        return False
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return x0 - dx <= pt[0] <= x1 + dx and y0 - dy <= pt[1] <= y1 + dy


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _nearest(ts: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(ts - t))) if len(ts) else 0


def maps(frames, scene=None) -> dict:
    """Per sampled frame: lasting-change blobs and hand-motion score. frames: list of Frame (t, PIL image)."""
    import cv2
    if not frames:
        return {"t": [], "blobs": [], "hands": []}
    g0 = _gray(frames[0].image)
    h, w = g0.shape
    scale = (w * h) / BASE_AREA
    min_area, big_area = MIN_AREA * scale, BIG_AREA * scale
    fps = 1.0 / max(1e-3, np.median(np.diff([f.t for f in frames]))) if len(frames) > 1 else CHANGE_FPS
    hold_n = max(2, int(round(HOLD_SEC * fps)))
    ref_n = max(2, int(round(REF_SEC * fps)))
    st = np.asarray(scene.t, dtype=float) if scene is not None and getattr(scene, "n", 0) else np.array([])
    safe = (scene.rois.get("safe") or {}).get("box") if scene is not None and getattr(scene, "rois", None) else None

    ref = np.zeros_like(g0)
    known = np.zeros((h, w), bool)
    cnt = np.zeros((h, w), np.int32)
    calm = np.zeros((h, w), np.int32)                           # frames the pixel has been free and stable
    prev = None
    kernel = np.ones((3, 3), np.uint8)
    out_t, blobs, hands, person_hist = [], [], [], []
    for f in frames:
        g = _gray(f.image)
        occ = np.zeros((h, w), bool)
        boxes = _persons(scene, _nearest(st, f.t)) if len(st) else []
        for b in boxes:
            x0, y0, x1, y1 = _pad(b, w, h)
            occ[y0:y1, x0:x1] = True
        diff_prev = np.abs(g - prev) if prev is not None else np.zeros_like(g)
        moving = cv2.dilate((diff_prev > MOVE_PIX).astype(np.uint8), kernel, iterations=2).astype(bool)
        hidden = occ | moving
        steady = ~hidden & (diff_prev < MOVE_PIX / 2) if prev is not None else ~hidden
        calm = np.where(steady, calm + 1, 0)
        new = (calm >= ref_n) & ~known                          # free and stable for REF_SEC = reference
        ref[new] = g[new]
        known |= new
        changed = known & ~hidden & (np.abs(g - ref) > REF_PIX)
        cnt = np.where(hidden, cnt, np.where(changed, cnt + 1, 0))       # held while hidden, reset when back
        lasting = ((cnt >= hold_n) & ~hidden).astype(np.uint8)
        lasting = cv2.morphologyEx(lasting, cv2.MORPH_OPEN, kernel)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(lasting, connectivity=8)
        fb = []
        for k in range(1, n):
            x, y, bw, bh, area = stats[k]
            if area < min_area:
                continue
            hold = float(cnt[lab == k].mean()) / fps
            s_area = 0.2 + 0.8 * min(1.0, np.log(area / min_area) / np.log(big_area / min_area))
            s_time = min(1.0, hold / (2 * HOLD_SEC))
            zone = "safe" if _inside(cents[k], safe) else "other"
            fb.append({"score": round(float(s_area * (0.5 + 0.5 * s_time)), 4), "area": int(area),
                       "hold": round(hold, 1), "zone": zone,
                       "box": [int(x), int(y), int(x + bw), int(y + bh)]})
        fb.sort(key=lambda b: -b["score"])
        blobs.append(fb[:5])
        # hands: a person box that stayed in place for ~2 s, fraction of moving pixels inside it
        hs = 0.0
        past = person_hist[-int(round(2 * fps))] if len(person_hist) >= int(round(2 * fps)) else []
        for b in boxes:
            if not any(_iou(b, p) >= 0.5 for p in past):
                continue
            x0, y0, x1, y1 = _pad(b, w, h, 0.0)
            if x1 - x0 < 4 or y1 - y0 < 4 or prev is None:
                continue
            frac = float((diff_prev[y0:y1, x0:x1] > MOVE_PIX).mean())
            if frac < 0.30:                                     # whole-body motion = walking, not counting
                hs = max(hs, min(1.0, frac / 0.08))
        hands.append(round(hs, 4))
        person_hist.append(boxes)
        out_t.append(f.t)
        prev = g
    return {"t": out_t, "blobs": blobs, "hands": hands, "fps": fps}


def _sustain(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    return np.array([x[max(0, i - k + 1):i + 1].min() for i in range(len(x))]) if len(x) else x


def features(m: dict, state) -> dict:
    """Per-second evidence aligned with the Qwen facts (state.t)."""
    ts = np.asarray(state.t, dtype=float)
    n = len(ts)
    out = {}
    if not n or not m["t"]:
        return out
    mt = np.asarray(m["t"], dtype=float)
    idx = [np.where(np.abs(mt - t) <= 0.5 + 1e-6)[0] for t in ts]   # change frames inside each fact second

    def per_sec(fn):
        val, info = np.zeros(n), [None] * n
        for i, js in enumerate(idx):
            for j in js:
                for b in m["blobs"][j]:
                    if fn(b) and b["score"] > val[i]:
                        val[i], info[i] = b["score"], b
        return val, info

    away = np.asarray(state.away_s, dtype=float)
    client = np.asarray(state.client, dtype=float)
    safe, safe_i = per_sec(lambda b: b["zone"] == "safe")
    obj, obj_i = per_sec(lambda b: b["zone"] == "other")
    hands = np.array([max((m["hands"][j] for j in js), default=0.0) for js in idx])
    no_client = (client < 0.3).astype(float)
    k = max(1, int(round(4 / float(state.dt))))                    # hand motion must last 4 s
    sig = {
        "c_safe": (safe * away, safe_i),
        "c_object": (obj, obj_i),
        "c_left": (obj * away, obj_i),
        "c_hands": (_sustain(hands, k) * no_client, [None] * n),
    }
    for f, (s, info) in sig.items():
        if not len(s) or float(s.max()) <= 0:
            out[f] = {"type": TYPES[f], "score": 0.0, "t": float(ts[0]), "generic": True}
            continue
        i = int(np.argmax(s))
        j = i
        while j > 0 and s[j - 1] >= 0.5 * s[i]:                # first second of the run that holds the peak
            j -= 1
        b = info[i] or {}
        bj = info[j] or {}
        # a blob counts only after HOLD_SEC: the change itself began `hold` seconds before it first counted
        onset = max(float(ts[0]), float(ts[j]) - bj["hold"]) if bj else max(float(ts[0]), float(ts[j]) - 3.0)
        hold = b.get("hold", float(ts[i] - onset))
        out[f] = {"type": TYPES[f], "score": round(float(s[i]), 4), "t": onset, "peak_t": float(ts[i]),
                  "generic": True, "box": b.get("box"), "zone": b.get("zone"),
                  "why": f"Около {onset:.0f} с: " + WHY[f].format(hold=hold, area=b.get("area", 0)) + "."}
    return out


def run(frames, scene, state) -> dict:
    m = maps(frames, scene)
    return {"version": VERSION, "t": m["t"], "features": features(m, state),
            "hands": m["hands"], "blobs": [b[:2] for b in m["blobs"]]}
