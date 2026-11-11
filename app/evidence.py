"""Video helpers shared by the service and the ML pipeline."""
import base64

import cv2

JPEG_QUALITY = 85
MAX_WIDTH = 1280  # keeps each base64 frame around 100-250 KB


def video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot decode video (unsupported codec or corrupted file)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if frames <= 0:
        raise ValueError("Video has no frames")
    return frames / fps


def evidence_frames(video_path: str, timestamps: list[float], max_frames: int = 3) -> list[dict]:
    """Return 1-3 {"timestamp_sec", "image_base64"} JPEG frames at the given times."""
    duration = video_duration(video_path)
    cap = cv2.VideoCapture(video_path)
    out = []
    for t in timestamps[:max_frames]:
        t = min(max(t, 0.0), max(duration - 0.05, 0.0))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            out.append({"timestamp_sec": round(t, 2), "image_base64": base64.b64encode(buf).decode()})
    cap.release()
    return out
