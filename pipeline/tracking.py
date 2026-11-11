"""
ByteTrack (Zhang et al., ECCV 2022, arXiv:2110.06864) over the Grounding DINO boxes, one tracker per class.

    tracks = track_all(frame_dets, cfg)
      -> {"person": [Track], "cash": [Track], ...}
    Track = {"id": "person_1", "label", "obs": [{"i", "t", "box", "score"}], "t0", "t1", "duration", "box"}

Backend: the maintained Roboflow implementation (`pip install trackers`, ByteTrackTracker) when installed;
otherwise the built-in two-stage ByteTrack below (same association idea: high-score boxes first, then
low-score boxes may only continue existing tracks, never start one; IoU matching; a lost track survives
`lost_buffer_sec`). The built-in one keeps the last box instead of a Kalman prediction, which is adequate
at 1 fps for people who mostly stand or sit.

Timestamps are the decoded frame times; the tracker runs in fixed-rate mode on the 1 fps sequence, so
`lost_buffer_sec` is converted to frames with the real sampling rate.
"""
from __future__ import annotations

import numpy as np

from .detect import iou


def _fps(frame_dets) -> float:
    if len(frame_dets) < 2:
        return 1.0
    dt = np.median(np.diff([f["t"] for f in frame_dets]))
    return 1.0 / dt if dt > 0 else 1.0


class SimpleByteTrack:
    """Built-in ByteTrack association (no Kalman filter)."""

    def __init__(self, high: float, activation: float, lost_frames: int, min_iou: float):
        self.high, self.act, self.lost, self.min_iou = high, activation, lost_frames, min_iou
        self.tracks = []          # dicts: id, box, last, hits
        self.next_id = 1

    def _match(self, tracks, boxes):
        pairs = sorted(((iou(t["box"], b), ti, bi) for ti, t in enumerate(tracks) for bi, b in enumerate(boxes)),
                       reverse=True)
        mt, mb, out = set(), set(), []
        for v, ti, bi in pairs:
            if v < self.min_iou:
                break
            if ti in mt or bi in mb:
                continue
            mt.add(ti)
            mb.add(bi)
            out.append((ti, bi))
        return out

    def update(self, boxes: list, scores: list, frame: int) -> list[int]:
        ids = [-1] * len(boxes)
        live = [t for t in self.tracks if frame - t["last"] <= self.lost]
        hi = [j for j, s in enumerate(scores) if s >= self.high]
        lo = [j for j, s in enumerate(scores) if s < self.high]
        used = set()
        for ti, bj in self._match(live, [boxes[j] for j in hi]):
            t, j = live[ti], hi[bj]
            t.update(box=boxes[j], last=frame)
            ids[j] = t["id"]
            used.add(id(t))
        rest = [t for t in live if id(t) not in used]
        for ti, bj in self._match(rest, [boxes[j] for j in lo]):
            t, j = rest[ti], lo[bj]
            t.update(box=boxes[j], last=frame)
            ids[j] = t["id"]
        for j in hi:
            if ids[j] == -1 and scores[j] >= self.act:
                t = {"id": self.next_id, "box": boxes[j], "last": frame}
                self.next_id += 1
                self.tracks.append(t)
                ids[j] = t["id"]
        self.tracks = [t for t in self.tracks if frame - t["last"] <= self.lost]
        return ids


class RoboflowByteTrack:
    def __init__(self, high: float, activation: float, lost_frames: int, min_iou: float):
        from trackers import ByteTrackTracker
        self.t = ByteTrackTracker(lost_track_buffer=lost_frames, frame_rate=1.0, track_activation_threshold=activation,
                                  minimum_consecutive_frames=1, high_conf_det_threshold=high,
                                  minimum_iou_threshold=min_iou)

    def update(self, boxes: list, scores: list, frame: int) -> list[int]:
        import supervision as sv
        if not boxes:
            self.t.update(sv.Detections.empty())
            return []
        d = sv.Detections(xyxy=np.array(boxes, dtype=float), confidence=np.array(scores, dtype=float),
                          class_id=np.arange(len(boxes)))           # class_id carries the input index back
        r = self.t.update(d)
        ids = [-1] * len(boxes)
        for k, tid in zip(r.class_id.tolist(), r.tracker_id.tolist()):
            ids[int(k)] = int(tid) + 1 if tid >= 0 else -1
        return ids


def backend(cfg: dict) -> str:
    want = cfg["tracking"].get("backend", "auto")
    if want in ("auto", "trackers"):
        import importlib.util
        if importlib.util.find_spec("trackers") and importlib.util.find_spec("supervision"):
            return "trackers"
        if want == "trackers":
            raise ImportError("tracking.backend: trackers requires  pip install trackers==2.6.1 supervision==0.31.0")
    return "builtin"


def _new_tracker(cfg, fps):
    tc = cfg["tracking"]
    lost = max(1, int(round(float(tc.get("lost_buffer_sec", 4)) * fps)))
    cls = RoboflowByteTrack if backend(cfg) == "trackers" else SimpleByteTrack
    return cls(float(tc["high_conf"]), float(tc["activation"]), lost, float(tc.get("min_iou", 0.1)))


def track_label(frame_dets: list[dict], label: str, cfg: dict) -> list[dict]:
    fps = _fps(frame_dets)
    tr = _new_tracker(cfg, fps)
    by_id, pending = {}, []
    for fr in frame_dets:
        ds = [d for d in fr["dets"] if d["label"] == label]
        ids = tr.update([d["box"] for d in ds], [d["score"] for d in ds], fr["i"])
        for d, tid in zip(ds, ids):
            ob = {"i": fr["i"], "t": fr["t"], "box": d["box"], "score": d["score"]}
            if tid == -1:
                pending.append(ob)
            else:
                by_id.setdefault(tid, []).append(ob)
    # a box the tracker returned without id (first frame of a track, low boxes) joins the track that overlaps
    # it in a neighbouring frame; otherwise it is dropped from tracks (it stays in the raw detections)
    for ob in pending:
        best, bv = None, 0.5
        for tid, obs in by_id.items():
            for o in obs:
                if abs(o["i"] - ob["i"]) <= 1 and o["i"] != ob["i"]:
                    v = iou(o["box"], ob["box"])
                    if v >= bv:
                        best, bv = tid, v
        if best is not None and all(o["i"] != ob["i"] for o in by_id[best]):
            by_id[best].append(ob)
    tracks = []
    for n, (tid, obs) in enumerate(sorted(by_id.items(), key=lambda kv: min(o["t"] for o in kv[1])), 1):
        obs.sort(key=lambda o: o["t"])
        boxes = np.array([o["box"] for o in obs])
        tracks.append({"id": f"{label}_{n}", "label": label, "obs": obs, "t0": obs[0]["t"], "t1": obs[-1]["t"],
                       "duration": obs[-1]["t"] - obs[0]["t"] + 1.0 / fps, "frames": len(obs),
                       "box": np.median(boxes, axis=0).round(1).tolist(),
                       "mean_score": round(float(np.mean([o["score"] for o in obs])), 4)})
    return tracks


def track_all(frame_dets: list[dict], cfg: dict) -> dict:
    labels = ["person"] + list(cfg["tracking"].get("object_classes", []))
    return {lab: track_label(frame_dets, lab, cfg) for lab in labels}


def box_at(track: dict, i: int, max_gap: int = 0):
    """Box of the track at frame i (None if the track has no observation within max_gap frames)."""
    best = min(track["obs"], key=lambda o: abs(o["i"] - i))
    return best["box"] if abs(best["i"] - i) <= max_gap else None
