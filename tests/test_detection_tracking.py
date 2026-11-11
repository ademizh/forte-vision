"""Detection (multiple objects, coordinates, tiles, timestamps, nothing invented) and ByteTrack tests."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from fakes import COLORS, BlobDetector, cfg, det, scenario
from pipeline import detections, tracking
from pipeline.frames import Frame


def _img(rects, size=(464, 848)):
    im = Image.new("RGB", size, (128, 128, 128))
    d = ImageDraw.Draw(im)
    for label, b in rects:
        d.rectangle([b[0], b[1], b[2] - 1, b[3] - 1], fill=COLORS[label])
    return im


def test_multiple_objects_same_class_are_kept():
    c = cfg()
    fr = [Frame(0.0, _img([("cash", (50, 50, 90, 70)), ("cash", (300, 600, 340, 620)), ("person", (100, 200, 200, 400))]))]
    out = detections.detect_frames(fr, BlobDetector(), c)
    cash = [d for d in out[0]["dets"] if d["label"] == "cash"]
    assert len(cash) == 2
    assert sorted(d["box"][0] for d in cash) == [50.0, 300.0]


def test_tiles_restore_original_coordinates():
    c = cfg(detection={"tiles": {"enabled": True, "every": 1, "size": 200, "overlap": 0.2, "objects": ["key"]}})
    box = (370, 700, 385, 712)                       # a small key in the bottom-right tile
    fr = [Frame(3.0, _img([("key", box)]))]
    out = detections.detect_frames(fr, BlobDetector(), c)
    keys = [d for d in out[0]["dets"] if d["label"] == "key"]
    assert len(keys) == 1                            # full-frame + tile boxes merged by NMS
    assert np.allclose(keys[0]["box"], box, atol=1.0)


def test_tile_grid_covers_frame():
    g = detections.tile_grid(464, 848, 416, 0.2)
    assert min(x[0] for x in g) == 0 and max(x[2] for x in g) == 464
    assert min(x[1] for x in g) == 0 and max(x[3] for x in g) == 848


def test_timestamps_and_no_fabricated_detections():
    c = cfg()
    fr = [Frame(0.0, _img([("cash", (10, 10, 40, 30))])), Frame(1.04, _img([]))]
    out = detections.detect_frames(fr, BlobDetector(), c)
    assert [f["t"] for f in out] == [0.0, 1.04]
    assert out[1]["dets"] == []


@pytest.mark.parametrize("backend", ["builtin", "trackers"])
def test_stable_ids_through_short_occlusion(backend):
    if backend == "trackers":
        pytest.importorskip("trackers")
    c = cfg(tracking={"backend": backend})
    A, B = [100, 100, 200, 300], [300, 100, 400, 300]

    def fn(t):
        ds = [det("person", [A[0] + t, A[1], A[2] + t, A[3]])]
        if t != 4:                                     # B missed for one frame
            ds.append(det("person", [B[0] - t, B[1], B[2] - t, B[3]]))
        return ds
    tr = tracking.track_label(scenario(10, fn), "person", c)
    assert len(tr) == 2                                # two people, identities kept through the gap
    assert sorted(t["frames"] for t in tr) == [9, 10]


def test_track_lost_longer_than_buffer_gets_new_id():
    c = cfg(tracking={"backend": "builtin", "lost_buffer_sec": 2})
    A = [100, 100, 200, 300]
    tr = tracking.track_label(scenario(12, lambda t: [det("person", A)] if t < 3 or t > 8 else []), "person", c)
    assert len(tr) == 2


def test_low_score_box_does_not_start_a_track_but_continues_one():
    c = cfg(tracking={"backend": "builtin"})
    A = [100, 100, 200, 300]
    weak_only = tracking.track_label(scenario(5, lambda t: [det("person", A, 0.2)]), "person", c)
    assert weak_only == []
    cont = tracking.track_label(scenario(5, lambda t: [det("person", A, 0.9 if t < 2 else 0.2)]), "person", c)
    assert len(cont) == 1 and cont[0]["frames"] == 5
