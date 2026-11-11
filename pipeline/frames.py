"""Frame sampling and sliding windows."""
from dataclasses import dataclass

import cv2
from PIL import Image


@dataclass
class Frame:
    t: float          # seconds from video start
    image: Image.Image


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "duration": n / fps, "width": w, "height": h}


def _to_pil(frame_bgr, max_width: int) -> Image.Image:
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def sample_frames(path: str, fps: float = 2.0, start: float = 0.0, end: float | None = None,
                  max_width: int = 640) -> list[Frame]:
    """Decode sequentially (faster and more exact than seeking) and keep frames at `fps`."""
    info = video_info(path)
    end = info["duration"] if end is None else min(end, info["duration"])
    cap = cv2.VideoCapture(path)
    src_fps = info["fps"]
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * src_fps))
    idx = int(start * src_fps)
    next_t, out = start, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / src_fps
        if t > end:
            break
        if t + 1e-6 >= next_t:
            out.append(Frame(round(t, 2), _to_pil(frame, max_width)))
            next_t += 1.0 / fps
        idx += 1
    cap.release()
    return out


def sliding_windows(frames: list[Frame], window_sec: float = 4.0, step_sec: float = 2.0) -> list[list[Frame]]:
    """Group sampled frames into overlapping windows."""
    if not frames:
        return []
    last_t = frames[-1].t
    windows, start = [], frames[0].t
    while True:
        w = [f for f in frames if start <= f.t < start + window_sec]
        if w:
            windows.append(w)
        if start + window_sec >= last_t:
            break
        start += step_sec
    return windows
