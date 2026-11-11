"""
Grounding DINO detections for every coarse frame, with tiled search for small objects.

    dets = detect_frames(frames, detector, cfg)
      -> [{"i": frame index, "t": seconds, "dets": [{"label", "score", "box": [x0, y0, x1, y1], "src"}]}]

* All boxes above detection.min_score are kept (top_k per class per frame), not one argmax box per class:
  two people, three cash bundles or two bags in one frame stay separate.
* Tiles (SAHI idea, arXiv:2202.06934): small objects are searched again in overlapping tiles; tile boxes are
  mapped back to ORIGINAL frame coordinates and merged with the full-frame boxes per class (NMS).
* Coordinates are always in the original frame (the detector resizes internally and returns relative boxes).
* Nothing is invented: a frame without a box for a class simply has none.

`detector` is any object with .boxes(images, groups, k, min_score) -> [{label: [[score, x0, y0, x1, y1], ...]}]
(pipeline/detect.Detector for Grounding DINO, tests use a fake).
"""
from __future__ import annotations

from .detect import iou


def tile_grid(w: int, h: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    def starts(n):
        if n <= size:
            return [0]
        step = max(1, int(size * (1 - overlap)))
        s = list(range(0, n - size + 1, step))
        if s[-1] != n - size:
            s.append(n - size)
        return s
    return [(x, y, min(w, x + size), min(h, y + size)) for y in starts(h) for x in starts(w)]


def nms(dets: list[dict], thr: float) -> list[dict]:
    kept = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(d["label"] != k["label"] or iou(d["box"], k["box"]) <= thr for k in kept):
            kept.append(d)
    return kept


def _rows_to_dets(res: dict, src: str, dx: float = 0.0, dy: float = 0.0, min_score: float = 0.0) -> list[dict]:
    out = []
    for label, rows in res.items():
        for r in rows:
            if r[0] < min_score:
                continue
            out.append({"label": label, "score": round(float(r[0]), 4), "src": src,
                        "box": [round(r[1] + dx, 1), round(r[2] + dy, 1), round(r[3] + dx, 1), round(r[4] + dy, 1)]})
    return out


def detect_frames(frames, detector, cfg: dict) -> list[dict]:
    dc = cfg["detection"]
    groups = [dict(g) for g in dc["prompt_groups"]]
    k, ms = int(dc["top_k"]), float(dc["min_score"])
    images = [f.image for f in frames]
    full = detector.boxes(images, groups, k=k, min_score=ms)
    out = [{"i": i, "t": float(f.t), "dets": _rows_to_dets(full[i], "full", min_score=ms)}
           for i, f in enumerate(frames)]
    tc = dc.get("tiles", {})
    if tc.get("enabled") and images:
        all_prompts = {kk: v for g in groups for kk, v in g.items()}
        small = {o: all_prompts[o] for o in tc.get("objects", []) if o in all_prompts}
        every = max(1, int(tc.get("every", 2)))
        for i in range(0, len(images), every):
            w, h = images[i].size
            grid = tile_grid(w, h, int(tc.get("size", 416)), float(tc.get("overlap", 0.2)))
            if len(grid) <= 1:
                continue
            crops = [images[i].crop(g) for g in grid]
            res = detector.boxes(crops, [small], k=k, min_score=ms)
            for (x0, y0, _, _), r in zip(grid, res):
                out[i]["dets"] += _rows_to_dets(r, "tile", x0, y0, ms)
    thr = float(dc.get("nms_iou", 0.5))
    for fr in out:
        fr["dets"] = nms(fr["dets"], thr)
    return out


def of_label(frame_dets: dict, label: str, min_score: float = 0.0) -> list[dict]:
    return [d for d in frame_dets["dets"] if d["label"] == label and d["score"] >= min_score]
