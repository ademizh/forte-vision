"""
Evidence images: real video frames with the boxes the detector actually returned at that moment.

    shots = evidence(video_path, event, cfg, max_shots=3)
      -> [{"t", "image": PIL, "boxes": [...], "note"}]
    to_b64(image), save(shots, folder, prefix)

* Frames are decoded at the event's own timestamps (start, strongest detector frame, end).
* A box is drawn only if it is stored in the event (rules.py copied it from a detection at that time);
  for events established by time + VLM only (e.g. safe open, door open) the image has no box and says so.
* Region close-ups that the VLM saw are outlined with a dashed region box, labelled as "region", never as a
  detection.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import ImageDraw

from . import video

COLORS = {"person": "#2b6cff", "cash": "#ff3b30", "key": "#ffcc00", "bag": "#ff9500", "cup": "#af52de",
          "handbag": "#ff9500", "backpack": "#ff9500", "phone": "#34c759"}


def _draw(img, boxes, region=None):
    im = img.copy()
    d = ImageDraw.Draw(im)
    for b in boxes:
        x0, y0, x1, y1 = [int(v) for v in b["box"]]
        col = COLORS.get(b.get("label", ""), "#ff3b30")
        d.rectangle([x0, y0, x1, y1], outline=col, width=3)
        d.text((x0 + 3, max(0, y0 - 12)), f"{b.get('label', '')} {b.get('score', 0):.2f}", fill=col)
    if region:
        x0, y0, x1, y1 = [int(v) for v in region]
        for k in range(x0, x1, 12):
            d.line([k, y0, min(k + 6, x1), y0], fill="#ffffff", width=2)
            d.line([k, y1, min(k + 6, x1), y1], fill="#ffffff", width=2)
        for k in range(y0, y1, 12):
            d.line([x0, k, x0, min(k + 6, y1)], fill="#ffffff", width=2)
            d.line([x1, k, x1, min(k + 6, y1)], fill="#ffffff", width=2)
        d.text((x0 + 3, y0 + 3), "region", fill="#ffffff")
    return im


def evidence(video_path: str, ev: dict, cfg: dict, max_shots: int = 3) -> list[dict]:
    times = []
    for t in sorted(ev.get("boxes", {}), key=float):
        times.append(float(t))
    for t in (ev["start"], *(ev.get("focus") or []), ev.get("end", ev["start"])):
        if all(abs(float(t) - x) >= 0.75 for x in times):
            times.append(float(t))
    times = sorted(times)[:max_shots] if len(times) > max_shots else sorted(times)
    region = next((q.get("crop") for q in ev.get("questions", []) if isinstance(q.get("crop"), list)), None)
    out = []
    mw = int(cfg["sampling"]["max_width"])
    for t in times:
        fr = video.sample(video_path, fps=25.0, start=max(0.0, t), end=t + 0.2, max_width=mw)
        if not fr:
            continue
        f = fr[0]
        bx = ev.get("boxes", {}).get(f"{t:.2f}", [])
        note = "" if bx else ("область, которую проверял SmolVLM2 (рамка детектора отсутствует)" if region
                              else "рамка отсутствует: событие установлено по времени и проверке SmolVLM2")
        out.append({"t": round(f.t, 2), "image": _draw(f.image, bx, region if not bx else None), "boxes": bx,
                    "note": note})
    return out


def to_b64(image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def save(shots, folder, prefix) -> list[str]:
    Path(folder).mkdir(parents=True, exist_ok=True)
    paths = []
    for k, s in enumerate(shots):
        p = Path(folder) / f"{prefix}_{k}_{s['t']:06.2f}s.jpg"
        s["image"].convert("RGB").save(p, quality=90)
        paths.append(str(p))
    return paths
