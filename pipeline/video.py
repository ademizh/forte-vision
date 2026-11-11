"""
Video decoding with real timestamps (seconds from the start of the video).

    meta = metadata(path)                       # fps, frames, duration, width, height
    frames = sample(path, fps=1.0)              # [Frame(t, image)] at native resolution
    frames = sample(path, fps=2.0, start=10, end=16)   # focused window
    vid = identity(path)                        # name + size + mtime: part of every cache key

Timestamps come from the decoded frame index / the container fps, never from the sample counter, so
1 fps samples of a 25 fps video are at 0.0, 1.0, 2.0 ... and a 29.97 fps video keeps its real times.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .frames import Frame, sample_frames, video_info


def metadata(path: str) -> dict:
    return video_info(str(path))


def sample(path: str, fps: float = 1.0, start: float = 0.0, end: float | None = None,
           max_width: int = 1280) -> list[Frame]:
    return sample_frames(str(path), fps=fps, start=max(0.0, start), end=end, max_width=max_width)


def identity(path: str) -> str:
    p = Path(path)
    st = p.stat()
    return hashlib.sha1(f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]


def nearest(frames: list[Frame], t: float) -> int:
    return min(range(len(frames)), key=lambda i: abs(frames[i].t - t)) if frames else -1
