"""
Fake detectors and a fake VLM with the same interface as the real ones (FORTE_MOCK=1).
Only for testing that the whole flow runs end to end on a machine without GPU / weights. The numbers are
deterministic noise derived from the pixels, never use them for results.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .detect import OBJECTS, SCENE_OBJECTS, SMALL_OBJECTS, nms, tile_grid


def _rng(*parts) -> np.random.Generator:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return np.random.default_rng(int(h[:8], 16))


class MockDetector:
    def __init__(self, name: str):
        self.name = name

    def boxes(self, images, groups=None, k=5, min_score=0.03):
        groups = groups or [OBJECTS, SCENE_OBJECTS]
        keys = [kk for g in groups for kk in g]
        out = []
        for im in images:
            w, h = im.size
            m = float(np.asarray(im.convert("L").resize((8, 8))).mean())
            d = {}
            for key in keys:
                r = _rng(self.name, key, round(m, 1))
                bx = []
                for _ in range(3):
                    x0, y0 = r.uniform(0, w * 0.7), r.uniform(0, h * 0.7)
                    bx.append([round(float(r.uniform(0.02, 0.6)), 4), x0, y0,
                               x0 + r.uniform(10, w * 0.3), y0 + r.uniform(10, h * 0.3)])
                d[key] = sorted(bx, key=lambda b: -b[0])
            out.append(d)
        return out

    def tiles(self, images, objects=None, k=5, min_score=0.05):
        out = []
        for img in images:
            w, h = img.size
            merged = {}
            for (x0, y0, x1, y1) in tile_grid(w, h):
                for obj, rows in self.boxes([img.crop((x0, y0, x1, y1))], [objects or SMALL_OBJECTS])[0].items():
                    merged.setdefault(obj, []).extend([[r[0], r[1] + x0, r[2] + y0, r[3] + x0, r[4] + y0] for r in rows])
            out.append({o: nms(r, 0.5, k) for o, r in merged.items()})
        return out

    def scores(self, images):
        return [{k: (v[0][0], v[0][1:]) for k, v in d.items()} for d in self.boxes(images)]


class MockVLM:
    def choice_probs(self, prompts, n_options, return_features=False):
        out = []
        for _, fr, u in prompts:
            m = float(np.asarray(fr[0].image.convert("L").resize((8, 8))).mean()) if fr else 0
            p = _rng(u, round(m, 1)).dirichlet(np.ones(n_options)).tolist()
            out.append(p)
        return (out, np.zeros((len(out), 4))) if return_features else out

    def frame_question_probs(self, system, frames, questions, n_options=2):
        return [[_rng(q, f.t).dirichlet(np.ones(n_options)).tolist() for q in questions] for f in frames]

    def generate(self, prompts, max_new_tokens=None):
        return ["(тестовый режим: описание не генерируется)" for _ in prompts]
