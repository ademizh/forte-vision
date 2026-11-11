"""
Scene layout found automatically in EVERY video (no hand-set coordinates; a new camera angle gets its own):

    z = find_zones(frame_dets, tracks, cfg, frame_size)
      z["rois"]        fixed regions: safe, laptop, monitor, door, drawer  (median box over the video, or None)
      z["cashier"]     id of the person track taken as the cashier (or None when it cannot be decided)
      z["cashier_zone"] box where the cashier works (median cashier box, enlarged)
      z["desk_zone"]   larger box around it (cash / items "on the desk")
      z["roles"]       {track_id: "cashier" | "visitor" | "unknown"}

Cashier = the person track present the longest share of the video, if that share is at least
zones.cashier_min_share; when two long tracks compete, the one closer to the laptop / drawer / safe wins;
when it still cannot be decided the role is "unknown" (nobody is called a cashier just for standing
behind the desk).
"""
from __future__ import annotations

import numpy as np

from .detect import iou

STATIC = ("safe", "laptop", "monitor", "door", "drawer")


def expand(box, f, size=None):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * f, (y1 - y0) * f
    b = [x0 - dx, y0 - dy, x1 + dx, y1 + dy]
    if size:
        b = [max(0, b[0]), max(0, b[1]), min(size[0], b[2]), min(size[1], b[3])]
    return [round(v, 1) for v in b]


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def inside(pt, box) -> bool:
    return box is not None and box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def overlap_frac(a, b) -> float:
    """Share of box a covered by box b."""
    if a is None or b is None:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def _roi(frame_dets, label, min_score):
    rows = []
    for fr in frame_dets:
        ds = [d for d in fr["dets"] if d["label"] == label]
        if ds:
            rows.append(max(ds, key=lambda d: d["score"]))
    if len(rows) < 2:
        return None
    sc = np.array([r["score"] for r in rows])
    if float(np.median(sc)) < min_score:
        return None
    keep = [r for r in rows if r["score"] >= np.percentile(sc, 50)]
    box = np.median(np.array([r["box"] for r in keep]), axis=0).round(1).tolist()
    stab = float(np.median([iou(r["box"], box) for r in rows]))
    return {"box": box, "score": round(float(np.median(sc)), 3), "stability": round(stab, 3)}


def find_zones(frame_dets: list[dict], tracks: dict, cfg: dict, size) -> dict:
    zc = cfg["zones"]
    n = max(1, len(frame_dets))
    rois = {k: _roi(frame_dets, k, float(zc["roi_min_score"])) for k in STATIC}
    persons = tracks.get("person", [])
    share = {p["id"]: p["frames"] / n for p in persons}
    anchors = [r["box"] for k, r in rois.items() if r and k in ("laptop", "monitor", "drawer", "safe")]
    cands = sorted(persons, key=lambda p: -share[p["id"]])
    cashier = None
    if cands and share[cands[0]["id"]] >= float(zc["cashier_min_share"]):
        top = [p for p in cands if share[p["id"]] >= 0.8 * share[cands[0]["id"]]]
        if len(top) == 1:
            cashier = top[0]
        elif anchors:
            def dist(p):
                c = center(p["box"])
                return min(np.hypot(c[0] - center(a)[0], c[1] - center(a)[1]) for a in anchors)
            top.sort(key=dist)
            cashier = top[0]
    roles = {p["id"]: "unknown" for p in persons}
    czone = dzone = None
    if cashier is not None:
        czone = expand(cashier["box"], float(zc["cashier_expand"]), size)
        dzone = expand(cashier["box"], float(zc["desk_expand"]), size)
        for p in persons:
            if p["id"] == cashier["id"]:
                roles[p["id"]] = "cashier"
            elif overlap_frac(p["box"], czone) < 0.5:
                roles[p["id"]] = "visitor"
    elif anchors:                                 # no cashier track: the desk is around the work objects
        xs = np.array(anchors)
        dzone = expand([xs[:, 0].min(), xs[:, 1].min(), xs[:, 2].max(), xs[:, 3].max()], 0.3, size)
    return {"rois": rois, "cashier": cashier["id"] if cashier else None, "cashier_zone": czone,
            "desk_zone": dzone, "roles": roles, "size": list(size)}
