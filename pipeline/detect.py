"""
Open-vocabulary object detector: OWLv2 (Apache-2.0), fully local.  (Grounding DINO still works: DETS=dino,owl)

  get_detector("owl").boxes(images)      top-k boxes per object per image (full frame)
  get_detector("owl").tiles(images)      SAHI: the same small objects searched in overlapping tiles

Why tiles (SAHI, Akyon et al. 2022, arXiv:2202.06934)
  OWLv2 resizes every image to 960x960. A key in the safe lock or a banknote stack is a few pixels in the
  464x848 frame. Cut into overlapping ~416 px tiles, every tile is enlarged ~2.3x before the detector sees it,
  so small objects get many more pixels. Tile boxes are mapped back to frame coordinates and merged (NMS).

Paths: OWL_PATH=/models/owlv2-base   (DINO_PATH for Grounding DINO).  A missing folder is skipped.
"""
import json
import os
import threading
import time
from pathlib import Path

PATHS = {
    "dino": os.environ.get("DINO_PATH", os.environ.get("DET_PATH", "/opt/forte/models/grounding-dino-tiny")),
    "owl": os.environ.get("OWL_PATH", "/opt/forte/models/owlv2-base"),
}
NAMES = {"dino": "Grounding DINO", "owl": "OWLv2"}
DET_BATCH = int(os.environ.get("DET_BATCH", "2"))   # small: OWLv2 attention at 960 px needs ~1 GB per image
TOP_K = int(os.environ.get("DET_TOPK", "5"))

# object -> text prompt (lowercase short nouns). The first 6 feed the state engine (owl_cash, owl_key, ...).
OBJECTS = {
    "cash": "banknotes",
    "key": "key",
    "person": "person",
    "cup": "cup",
    "phone": "mobile phone",
    "bag": "bag",
}
# Scene objects: fixed regions (safe, laptop, door) and personal items.
SCENE_OBJECTS = {
    "safe": "safe",
    "cabinet": "metal cabinet",
    "laptop": "laptop",
    "door": "door",
    "handbag": "handbag",
    "backpack": "backpack",
}
ALL_OBJECTS = {**OBJECTS, **SCENE_OBJECTS}
# Small objects searched again in tiles (SAHI)
SMALL_OBJECTS = {k: ALL_OBJECTS[k] for k in ("cash", "key", "cup", "phone", "handbag")}
TILE = int(os.environ.get("TILE_SIZE", "416"))
TILE_OVERLAP = float(os.environ.get("TILE_OVERLAP", "0.2"))

_lock = threading.Lock()
_instances = {}


def wanted() -> list[str]:
    return [x for x in os.environ.get("DETS", "owl").split(",") if x]


def available() -> list[str]:
    """Detectors to use (DETS, default owl) whose model folder exists."""
    return [n for n in wanted() if n in PATHS and (Path(PATHS[n]) / "config.json").exists()]


def get_detector(name: str = "owl"):
    with _lock:
        if name not in _instances:
            _instances[name] = Detector(PATHS[name], name=name)
    return _instances[name]


def iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def tile_grid(w: int, h: int, tile: int = TILE, overlap: float = TILE_OVERLAP) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles covering the whole image (SAHI slicing)."""
    def starts(n):
        if n <= tile:
            return [0]
        step = max(1, int(tile * (1 - overlap)))
        s = list(range(0, n - tile + 1, step))
        if s[-1] != n - tile:
            s.append(n - tile)
        return s
    return [(x, y, min(w, x + tile), min(h, y + tile)) for y in starts(h) for x in starts(w)]


def nms(rows: list[list], thr: float = 0.5, k: int = TOP_K) -> list[list]:
    kept = []
    for r in sorted(rows, key=lambda r: -r[0]):
        if all(iou(r[1:5], q[1:5]) <= thr for q in kept):
            kept.append(r)
            if len(kept) >= k:
                break
    return kept


class Detector:
    def __init__(self, path: str, device: str | None = None, name: str | None = None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if not (Path(path) / "config.json").exists():
            raise FileNotFoundError(f"Detector folder not found: {path} (set OWL_PATH / DINO_PATH)")
        t0 = time.time()
        self.kind = json.loads((Path(path) / "config.json").read_text())["model_type"]   # owlv2 | grounding-dino
        if self.kind not in ("grounding-dino", "owlv2"):
            raise ValueError(f"unsupported detector type {self.kind}")
        self.name = name or ("dino" if self.kind == "grounding-dino" else "owl")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True)
        self.model = self.model.to(self.device).eval()
        self.torch = torch
        print(f"[det] loaded {self.name} ({self.kind}) from {path} in {time.time() - t0:.0f}s")

    # Grounding DINO: one caption "banknotes. key. person." -> the token span of each prompt
    def _spans(self, input_ids, texts):
        tok = self.processor.tokenizer
        dot = tok.convert_tokens_to_ids(".")
        special = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
        spans, cur = [], []
        for i, t in enumerate(input_ids.tolist()):
            if t == dot or t in special:
                if cur:
                    spans.append(cur)
                cur = []
            else:
                cur.append(i)
        if cur:
            spans.append(cur)
        if len(spans) != len(texts):
            raise RuntimeError(f"prompt split mismatch: {len(spans)} spans for {len(texts)} prompts")
        return spans

    def boxes(self, images, groups: list[dict] | None = None, k: int = TOP_K, min_score: float = 0.03) -> list[dict]:
        """[{obj: [[score, x0, y0, x1, y1], ...]}] per image. DINO: one caption per group; OWL: all prompts at once
        (OWL scores every text query independently, so adding prompts does not change the others)."""
        groups = groups or [OBJECTS, SCENE_OBJECTS]
        if self.kind == "owlv2":
            groups = [{kk: v for g in groups for kk, v in g.items()}]
        out = [{} for _ in images]
        for g in groups:
            keys, texts = list(g), list(g.values())
            for s in range(0, len(images), DET_BATCH):
                for j, d in enumerate(self._boxes(images[s:s + DET_BATCH], keys, texts, k, min_score)):
                    out[s + j].update(d)
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        return out

    def tiles(self, images, objects: dict | None = None, k: int = TOP_K, min_score: float = 0.05) -> list[dict]:
        """SAHI: small objects searched in overlapping tiles; boxes in frame coordinates, merged with NMS."""
        objects = objects or SMALL_OBJECTS
        out = []
        for img in images:
            w, h = img.size
            grid = tile_grid(w, h)
            crops = [img.crop(g) for g in grid]
            res = self.boxes(crops, [objects], k=k, min_score=min_score)
            merged = {}
            for (x0, y0, _, _), d in zip(grid, res):
                for obj, rows in d.items():
                    merged.setdefault(obj, []).extend(
                        [[r[0], r[1] + x0, r[2] + y0, r[3] + x0, r[4] + y0] for r in rows if r[0] >= min_score])
            out.append({obj: nms(rows, 0.5, k) for obj, rows in merged.items()})
        return out

    def _boxes(self, images, keys, texts, k, min_score) -> list[dict]:
        torch = self.torch
        if self.kind == "grounding-dino":
            caption = ". ".join(texts) + "."
            inputs = self.processor(images=images, text=[caption] * len(images), return_tensors="pt",
                                    padding=True).to(self.device)
        else:
            inputs = self.processor(text=[texts] * len(images), images=images, return_tensors="pt",
                                    padding=True).to(self.device)
        with torch.inference_mode():
            o = self.model(**inputs)
        prob = o.logits.float().sigmoid()                 # GDINO: (B, queries, tokens)  OWLv2: (B, boxes, prompts)
        bx = o.pred_boxes.float()                         # (B, N, 4) cx, cy, w, h relative
        res = []
        for b, img in enumerate(images):
            w, h = img.size
            if self.kind == "grounding-dino":
                spans = self._spans(inputs["input_ids"][b], texts)
                per = [prob[b][:, sp].max(dim=1).values for sp in spans]
                sx, sy = w, h
            else:
                per = [prob[b][:, j] for j in range(len(texts))]
                sx = sy = max(w, h)                       # OWLv2 boxes are relative to the padded square
            d = {}
            for key, sc in zip(keys, per):
                order = torch.argsort(sc, descending=True)[:64].tolist()
                kept = []
                for i in order:
                    score = float(sc[i])
                    if kept and score < min_score:
                        break
                    cx, cy, bw, bh = bx[b, i].tolist()
                    box = [max(0.0, (cx - bw / 2) * sx), max(0.0, (cy - bh / 2) * sy),
                           min(float(w), (cx + bw / 2) * sx), min(float(h), (cy + bh / 2) * sy)]
                    if kept and (box[2] - box[0] < 2 or box[3] - box[1] < 2):
                        continue                          # the top box is always kept (= the object's frame score)
                    if any(iou(box, kb[1:]) > 0.5 for kb in kept):
                        continue
                    kept.append([round(score, 4)] + [round(v, 1) for v in box])
                    if len(kept) >= k:
                        break
                d[key] = kept
            res.append(d)
        return res
