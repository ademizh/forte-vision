"""
Scene from the object detector (OWLv2 by default): boxes -> detections -> fixed regions + object tracks + people.

    run_detectors(frames, detectors) -> rec  {"t", "size", "dets": {"owl": [frame -> {obj: [[s, x0, y0, x1, y1]..]}]},
                                              "tiles": {"owl": [...same, SAHI tiles...]}}         (cached per video)
    legacy_scores(rec)  -> {"owl_cash": [...], ...}  top-1 full-frame score per frame (state engine signals)
    Scene(rec)
        .rois[name]      fixed regions found automatically in THIS video: "safe", "laptop", "door"
        .tracks[obj]     objects followed over time (cash, key, cup, phone, bag, handbag, backpack)
        .persons         number of people per frame
        .strength[obj]   per-frame strength of the object (elevation above this video's usual level)

No coordinates are set by hand: every video is searched again, so another camera angle gets its own regions.

Detections: a box counts as HIGH if its score is above max(absolute minimum, the video's usual level + delta),
as LOW if it is above half of that margin. Full-frame boxes and tile boxes (SAHI) have separate usual levels.

Tracking (ByteTrack idea, Zhang et al. 2022, arXiv:2110.06864): high boxes are matched to the existing tracks
first; LOW boxes are then used only to continue a track that already exists (never to start one). An object that
is briefly blurred or half hidden keeps its track instead of breaking into pieces, while weak noise cannot
create new objects.

With two detectors (DETS=dino,owl), boxes of the same object with IoU >= 0.3 are fused (both agree).
"""
from __future__ import annotations

import numpy as np

from . import detect
from .detect import ALL_OBJECTS, OBJECTS, iou, nms

ABS_MIN = {"dino": 0.30, "owl": 0.12}
DELTA = {"dino": 0.08, "owl": 0.05}
ROI_MIN = {"dino": 0.20, "owl": 0.06}
STATIC = {"safe": ("safe", "cabinet"), "laptop": ("laptop",), "door": ("door",)}
MOVABLE = ("cash", "key", "cup", "phone", "bag", "handbag", "backpack", "person")
NAMES = detect.NAMES
MAX_AGE = 2            # frames a track survives without a detection


def run_detectors(frames, detectors: dict, tile_every: int = 1) -> dict:
    images = [f.image for f in frames]
    rec = {"t": [f.t for f in frames], "size": list(images[0].size) if images else [0, 0], "dets": {}, "tiles": {}}
    for name, det in detectors.items():
        rec["dets"][name] = det.boxes(images)
        tl = [None] * len(images)
        idx = list(range(0, len(images), max(1, tile_every)))
        for i, d in zip(idx, det.tiles([images[i] for i in idx])):
            tl[i] = d
        rec["tiles"][name] = tl
    return rec


def legacy_scores(rec: dict) -> dict:
    out = {}
    for d, per in rec.get("dets", {}).items():
        for obj in OBJECTS:
            out[f"{d}_{obj}"] = [round(float(fr[obj][0][0]), 4) if fr.get(obj) else 0.0 for fr in per]
    return out


def _median_box(boxes):
    return [float(np.median([b[i] for b in boxes])) for i in range(4)]


class Track:
    def __init__(self, obj: str):
        self.obj, self.idx, self.boxes, self.scores, self.low = obj, [], [], [], []

    def add(self, i, det, low=False):
        self.idx.append(i)
        self.boxes.append(det["box"])
        self.scores.append(det["strength"])
        self.low.append(low)

    @property
    def last(self):
        return self.idx[-1]

    def summary(self, t) -> dict:
        b = int(np.argmax(self.scores))
        return {"obj": self.obj, "i0": self.idx[0], "i1": self.idx[-1], "t0": float(t[self.idx[0]]),
                "t1": float(t[self.idx[-1]]), "frames": len(self.idx),
                "duration": float(t[self.idx[-1]] - t[self.idx[0]]),
                "mean": float(np.mean(self.scores)), "max": float(self.scores[b]),
                "low_frac": float(np.mean(self.low)), "box": _median_box(self.boxes),
                "best_i": self.idx[b], "best_box": self.boxes[b],
                "path": [[i, [round(v, 1) for v in bx]] for i, bx in zip(self.idx, self.boxes)]}


class Scene:
    def __init__(self, rec: dict):
        self.t = np.asarray(rec.get("t", []), dtype=float)
        self.n = len(self.t)
        self.size = rec.get("size", [0, 0])
        dets = rec.get("dets", {})
        self.dets_present = [d for d in detect.wanted() if d in dets and len(dets[d]) == self.n] or \
                            [d for d in dets if len(dets[d]) == self.n]
        per = {d: dets[d] for d in self.dets_present}
        tiles = {d: rec.get("tiles", {}).get(d) or [None] * self.n for d in self.dets_present}
        self.top, self.base = {}, {}
        for d in self.dets_present:
            for obj in ALL_OBJECTS:
                x = np.array([fr[obj][0][0] if fr.get(obj) else 0.0 for fr in per[d]], dtype=float)
                self.top[(d, obj)] = x
                self.base[(d, obj, "full")] = float(np.percentile(x, 20)) if len(x) else 0.0
                xt = [fr[obj][0][0] for fr in tiles[d] if fr and fr.get(obj)]
                self.base[(d, obj, "tile")] = float(np.percentile(xt, 20)) if xt else 0.0
        self.rois = {name: self._roi(per, cands) for name, cands in STATIC.items()}
        self.frames = [self._frame(per, tiles, i) for i in range(self.n)]
        self.strength = {obj: np.array([max((x["strength"] for x in fr[obj]["high"]), default=0.0)
                                        for fr in self.frames]) for obj in MOVABLE}
        self.persons = np.array([len(fr["person"]["high"]) for fr in self.frames], dtype=float)
        if self.n >= 3:
            self.persons = np.array([np.median(self.persons[max(0, i - 1):i + 2]) for i in range(self.n)])
        self.tracks = {obj: self._track(obj) for obj in MOVABLE if obj != "person"}

    # ---- detections of one frame
    def _level(self, d, obj, src):
        return max(ABS_MIN[d], self.base.get((d, obj, src), 0.0) + DELTA[d])

    def _cands(self, d, obj, full, tile):
        rows = [r + ["full"] for r in (full.get(obj, []) if full else [])]
        rows += [r + ["tile"] for r in (tile.get(obj, []) if tile else [])]
        high, low = [], []
        for r in rows:
            s, src = r[0], r[5]
            lvl = self._level(d, obj, src)
            elev = max(0.0, s - self.base.get((d, obj, src), 0.0))
            item = {"box": r[1:5], "score": s, "src": src, "strength": elev, "det": d}
            if s >= lvl:
                high.append(item)
            elif s >= lvl - DELTA[d] / 2:
                low.append(item)

        def dedupe(items):
            kept = []
            for x in sorted(items, key=lambda x: -x["score"]):
                if all(iou(x["box"], y["box"]) <= 0.5 for y in kept):
                    kept.append(x)
            return kept[:detect.TOP_K]
        return dedupe(high), dedupe(low)

    def _frame(self, per, tiles, i) -> dict:
        out = {}
        nd = max(1, len(self.dets_present))
        for obj in MOVABLE:
            by = {d: self._cands(d, obj, per[d][i], tiles[d][i]) for d in self.dets_present}
            res = {}
            for level in (0, 1):                                      # 0 = high, 1 = low
                lists = [by[d][level] for d in self.dets_present]
                if nd == 1:
                    res["high" if level == 0 else "low"] = lists[0] if lists else []
                    continue
                fused, used = [], set()
                a, o = lists[0], lists[1]
                for x in a:
                    m = max(((j, iou(x["box"], y["box"])) for j, y in enumerate(o) if j not in used),
                            key=lambda z: z[1], default=(None, 0))
                    if m[0] is not None and m[1] >= 0.3:
                        used.add(m[0])
                        y = o[m[0]]
                        fused.append({**x, "box": [(p + q) / 2 for p, q in zip(x["box"], y["box"])],
                                      "strength": (x["strength"] + y["strength"]) / 2, "both": True})
                    else:
                        fused.append({**x, "strength": x["strength"] / nd})
                fused += [{**y, "strength": y["strength"] / nd} for j, y in enumerate(o) if j not in used]
                res["high" if level == 0 else "low"] = sorted(fused, key=lambda z: -z["strength"])
            out[obj] = res
        return out

    def _roi(self, per, cands):
        best = []
        for d in self.dets_present:
            for obj in cands:
                rows = [fr[obj][0] for fr in per[d] if fr.get(obj)]
                if len(rows) < 2:
                    continue
                sc = np.array([r[0] for r in rows])
                keep = [r for r in rows if r[0] >= np.percentile(sc, 50)]
                box = _median_box([r[1:5] for r in keep])
                stab = float(np.median([iou(r[1:5], box) for r in rows]))
                med = float(np.median(sc))
                if med < ROI_MIN[d]:
                    continue
                best.append({"det": d, "prompt": ALL_OBJECTS[obj], "box": box, "score": round(med, 3),
                             "stability": round(stab, 3), "rank": stab * med / ROI_MIN[d]})
        if not best:
            return None
        best.sort(key=lambda r: -r["rank"])
        top = best[0]
        other = next((r for r in best if r["det"] != top["det"]), None)
        if other and iou(top["box"], other["box"]) >= 0.3:
            return {**top, "box": [(x + y) / 2 for x, y in zip(top["box"], other["box"])], "agree": True}
        return {**top, "agree": False}

    def _track(self, obj):
        """ByteTrack-style association: high boxes first, then low boxes only extend existing tracks."""
        done, active = [], []

        def match(tracks, dets, i):
            pairs = sorted(((iou(tr.boxes[-1], d["box"]), ti, di) for ti, tr in enumerate(tracks)
                            for di, d in enumerate(dets)), reverse=True)
            mt, md = set(), set()
            for v, ti, di in pairs:
                if v < 0.3:
                    break
                if ti in mt or di in md:
                    continue
                mt.add(ti)
                md.add(di)
                yield ti, di

        for i, fr in enumerate(self.frames):
            high, low = fr[obj]["high"], fr[obj]["low"]
            live = [tr for tr in active if i - tr.last <= MAX_AGE]
            used_t, used_h = set(), set()
            for ti, di in match(live, high, i):
                live[ti].add(i, high[di])
                used_t.add(ti)
                used_h.add(di)
            rest = [tr for k, tr in enumerate(live) if k not in used_t]
            for ti, di in match(rest, low, i):
                rest[ti].add(i, low[di], low=True)
            for di, d in enumerate(high):
                if di not in used_h:
                    nt = Track(obj)
                    nt.add(i, d)
                    active.append(nt)
            still = []
            for tr in active:
                (still if i - tr.last <= MAX_AGE else done).append(tr)
            active = still
        done += active
        return sorted((tr.summary(self.t) for tr in done), key=lambda s: -s["mean"] * s["frames"])

    def box_at(self, track: dict, i: int):
        """Box of a track at frame i (nearest frame of its path)."""
        p = min(track["path"], key=lambda x: abs(x[0] - i))
        return p[1]

    def det_text(self, obj: str, i: int) -> str:
        parts = []
        for d in self.dets_present:
            x = self.top.get((d, obj))
            if x is not None and len(x):
                parts.append(f"{NAMES[d]} {x[i]:.2f} (usual {self.base[(d, obj, 'full')]:.2f})")
        return ", ".join(parts)

    def summary(self) -> dict:
        return {"rois": self.rois,
                "tracks": {k: [{kk: vv for kk, vv in x.items() if kk != "path"} for x in v[:3]]
                           for k, v in self.tracks.items() if v},
                "persons_max": float(self.persons.max()) if self.n else 0.0,
                "detectors": self.dets_present}


__all__ = ["run_detectors", "legacy_scores", "Scene", "nms"]
