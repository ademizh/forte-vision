"""Test helpers: a deterministic colour-blob 'detector', synthetic frame detections and synthetic videos."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import config  # noqa: E402
from pipeline.frames import Frame  # noqa: E402

# label -> RGB colour painted in synthetic videos
COLORS = {"person": (40, 40, 60), "cash": (30, 160, 60), "key": (230, 210, 20), "bag": (200, 110, 30),
          "cup": (150, 40, 200), "safe": (90, 90, 90), "laptop": (20, 20, 20), "door": (120, 70, 20)}


class BlobDetector:
    """Finds solid colour blobs (a real detection on synthetic images; nothing is invented)."""

    def __init__(self, score=0.8):
        self.score = score
        self.calls = 0

    def boxes(self, images, groups, k=5, min_score=0.0):
        import cv2
        self.calls += 1
        keys = [kk for g in groups for kk in g]
        out = []
        for im in images:
            a = np.asarray(im.convert("RGB")).astype(int)
            d = {}
            for key in keys:
                col = COLORS.get(key)
                if col is None:
                    d[key] = []
                    continue
                mask = (np.abs(a - np.array(col)).sum(axis=2) <= 30).astype(np.uint8)
                n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                rows = []
                for j in range(1, n):
                    x, y, w, h, area = stats[j]
                    if area >= 20:
                        rows.append([self.score, float(x), float(y), float(x + w), float(y + h)])
                d[key] = sorted(rows, key=lambda r: -r[0])[:k]
            out.append(d)
        return out


def cfg(**over):
    c = config.load()
    c["models"]["detector_path"] = "test"
    c["models"]["vlm_path"] = "test"
    c["detection"]["tiles"]["enabled"] = False
    for k, v in over.items():
        c[k].update(v)
    return c


def det(label, box, score=0.8):
    return {"label": label, "score": score, "box": list(map(float, box)), "src": "full"}


CASHIER = [200, 300, 300, 450]          # the cashier sits here
CASH = [320, 420, 360, 440]             # cash on the desk next to the cashier
VISITOR = [220, 80, 320, 220]           # client side


def scenario(seconds, fn):
    """fn(t) -> list of detections for second t. Returns frame_dets at 1 fps."""
    return [{"i": i, "t": float(i), "dets": fn(i)} for i in range(seconds)]


def frames_blank(n, size=(464, 848)):
    from PIL import Image
    return [Frame(float(i), Image.new("RGB", size, (128, 128, 128))) for i in range(n)]


def write_video(path, seconds, paint, fps=10, size=(464, 848)):
    """paint(img_bgr numpy, t) draws the scene for time t."""
    import cv2
    vw = cv2.VideoWriter(str(path), 0, fps, size)          # uncompressed .avi: colours stay exact
    for k in range(int(seconds * fps)):
        img = np.full((size[1], size[0], 3), 128, np.uint8)
        paint(img, k / fps)
        vw.write(img)
    vw.release()


def rect(img, box, label):
    import cv2
    r, g, b = COLORS[label]
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]) - 1, int(box[3]) - 1), (b, g, r), -1)
