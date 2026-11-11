"""
Bridge between the service and the ML pipeline.

    PIPELINE=vlm   -> real models (pipeline/analyze.py): Qwen2.5-VL + OWLv2, loaded at startup
    PIPELINE=mock  -> placeholder result, no GPU needed

Contract: analyze(video_path) -> {"violations": [...]}; see README for the fields.
"""
import os
import threading
import time

from forte import load_env
load_env()  # uvicorn does not read .env automatically; load before PIPELINE/API_KEY are read

from .evidence import evidence_frames, video_duration

PIPELINE = os.environ.get("PIPELINE", "mock")
print(f"[service] PIPELINE={PIPELINE}  (.env: {__import__('forte').ROOT / '.env'})  "
      + ("REAL MODELS" if PIPELINE == "vlm" else "MOCK - set PIPELINE=vlm in .env"), flush=True)
MOCK_MODE = os.environ.get("MOCK_MODE", "violation")
MOCK_DELAY_SEC = float(os.environ.get("MOCK_DELAY_SEC", "3"))

if PIPELINE == "vlm":
    from pipeline.analyze import analyze as _vlm_analyze
    from pipeline.vlm import get_vlm

    def _preload():
        """Load Qwen and OWLv2 now, not on the jury's first video."""
        get_vlm()
        try:
            from pipeline import detect
            for name in detect.available():
                detect.get_detector(name)
        except Exception as e:  # noqa: BLE001
            print(f"[preload] detector not loaded: {e}")

    threading.Thread(target=_preload, daemon=True).start()


def analyze(video_path: str) -> dict:
    if PIPELINE == "vlm":
        return _vlm_analyze(video_path)
    duration = video_duration(video_path)
    time.sleep(MOCK_DELAY_SEC)
    if MOCK_MODE == "none":
        return {"violations": []}
    t = round(duration / 2, 1)
    return {"violations": [{
        "type": "ТЕСТ: заглушка (модель ещё не подключена)",
        "timecode_sec": t,
        "description": "Тестовый результат. Кадр взят из середины видео.",
        "evidence_frames": evidence_frames(video_path, [t - 1, t, t + 1]),
        "rule": {"document": "—", "clause": "—"},
    }]}
