"""End to end on synthetic videos with a real (colour-blob) detector and a fake VLM: no GPU, no model files."""
import json
import subprocess
import sys

import pytest

from fakes import CASH, CASHIER, ROOT, BlobDetector, cfg, rect, write_video
from pipeline import calibration, engine
from pipeline.smolvlm import MockVLM


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("vids")

    def viol(img, t):                          # cashier leaves at 8 s, cash stays on the desk
        rect(img, CASH, "cash")
        if t < 8:
            rect(img, CASHIER, "person")

    def clean(img, t):                         # cashier stays; cash taken away at 5 s
        rect(img, CASHIER, "person")
        if t < 5:
            rect(img, CASH, "cash")
    write_video(d / "V-1.avi", 20, viol)
    write_video(d / "C-1.avi", 20, clean)
    return d


def _models(c, answers=None):
    return engine.Models(c, detector=BlobDetector(), vlm=MockVLM(answers or {"cash remain": [0.85, 0.1, 0.05]}))


def test_single_video_end_to_end(videos, tmp_path):
    c = cfg()
    rec = engine.run_video(str(videos / "V-1.avi"), _models(c), c, cache_dir=str(tmp_path))
    for key in ("video", "candidates", "timing", "zones", "states", "models", "vlm_skipped"):
        assert key in rec
    assert rec["timing"]["total"] > 0 and "detection" in rec["timing"]
    cu = [x for x in rec["candidates"] if x["type"] == "cash_unattended"]
    assert cu and 7.0 <= cu[0]["start"] <= 9.5
    assert cu[0]["boxes"], "evidence boxes must come from real detections"
    json.dumps(rec)                                      # serialisable
    res = engine.result(rec, {"global": 0.2, "per_type": {}}, c)
    assert res["violation"] and res["type"] == "cash_unattended"
    clean = engine.run_video(str(videos / "C-1.avi"), _models(c), c, cache_dir=None)
    assert not [x for x in clean["candidates"] if x["type"] == "cash_unattended"]


def test_cache_is_reused_and_keyed_by_config(videos, tmp_path):
    c = cfg()
    det = BlobDetector()
    m = engine.Models(c, detector=det, vlm=MockVLM())
    engine.run_video(str(videos / "V-1.avi"), m, c, cache_dir=str(tmp_path))
    n = det.calls
    engine.run_video(str(videos / "V-1.avi"), m, c, cache_dir=str(tmp_path))
    assert det.calls == n                                # detections read from the cache
    c2 = cfg(detection={"min_score": 0.2})
    engine.run_video(str(videos / "V-1.avi"), engine.Models(c2, detector=det, vlm=MockVLM()), c2, cache_dir=str(tmp_path))
    assert det.calls > n                                 # another config -> another key -> recomputed


def test_evidence_frames_only_real_boxes(videos):
    from pipeline import evidence
    c = cfg()
    rec = engine.run_video(str(videos / "V-1.avi"), _models(c), c, cache_dir=None)
    ev = calibration.decide(rec, {"global": 0.05, "per_type": {}}, c)
    shots = evidence.evidence(str(videos / "V-1.avi"), ev[0], c)
    assert 1 <= len(shots) <= 3
    for s in shots:
        assert s["boxes"] or s["note"]


def test_missing_model_files_give_useful_errors():
    c = cfg()
    c["models"]["detector_path"] = "/nonexistent/dino"
    c["models"]["vlm_path"] = "/nonexistent/smolvlm"
    m = engine.Models(c)
    with pytest.raises(FileNotFoundError, match="grounding-dino"):
        _ = m.detector
    with pytest.raises(FileNotFoundError, match="SmolVLM2"):
        _ = m.vlm


def test_no_qwen_or_owl_in_the_verified_engine(videos):
    code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "from fakes import cfg, BlobDetector\nfrom pipeline import engine\nfrom pipeline.smolvlm import MockVLM\n"
            "c = cfg()\nengine.run_video(%r, engine.Models(c, detector=BlobDetector(), vlm=MockVLM()), c)\n"
            "bad = [m for m in sys.modules if m in ('pipeline.vlm', 'pipeline.objects', 'pipeline.state_engine', "
            "'pipeline.chain', 'pipeline.classify')]\nprint(bad)\nassert not bad, bad\n"
            % (str(ROOT), str(ROOT / 'tests'), str(videos / "V-1.avi")))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    c = cfg()
    assert "qwen" not in json.dumps(c).lower() and "owl" not in json.dumps(c).lower()


def test_dataset_cli_mock(videos, tmp_path):
    out = tmp_path / "x.jsonl"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_vlm.py"), "dataset", "--mock", "--videos",
                        str(videos), "--out", str(out), "--overwrite", "--cache", str(tmp_path / "c")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert {x["video"] for x in rows} == {"V-1", "C-1"}
