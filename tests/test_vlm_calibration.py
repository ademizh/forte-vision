"""VLM answer handling (structure, malformed output, frame references, letter bias) and calibration."""
import json

from fakes import cfg
from pipeline import calibration, vlm_parse
from pipeline.smolvlm import SmolVLM
from pipeline.vlm_parse import FALSE, TRUE, UNKNOWN


# ---------------------------------------------------------------- structured answers

def test_valid_structured_answer():
    r = vlm_parse.parse_explanation('ok {"observed": true, "reason": "cash on desk", "supporting_frames": [0, 2],'
                                    ' "uncertainty": "small"} done', 3, [10.0, 11.0, 12.0])
    assert r["ok"] and r["observed"] is True and r["supporting_times"] == [10.0, 12.0] and not r["errors"]


def test_malformed_answer():
    r = vlm_parse.parse_explanation("I think there is cash", 3)
    assert not r["ok"] and r["observed"] is None and r["errors"]


def test_repaired_quotes_and_trailing_comma():
    r = vlm_parse.parse_explanation("{'observed': 'yes', 'reason': 'x', 'supporting_frames': [1],}", 2)
    assert r["ok"] and r["observed"] is True and r["supporting_frames"] == [1]


def test_missing_observed_and_confidence_ignored():
    r = vlm_parse.parse_explanation('{"reason": "x", "confidence": 0.99}', 2)
    assert not r["ok"]
    assert any("confidence ignored" in e for e in r["errors"])


def test_invalid_frame_references_dropped():
    r = vlm_parse.parse_explanation('{"observed": false, "supporting_frames": [0, 7, "x", -1]}', 3)
    assert r["supporting_frames"] == [0] and len(r["errors"]) == 3


def test_contradictory_or_flat_answers_are_unknown():
    assert vlm_parse.answer_state([0.45, 0.45, 0.10], 0.55, 0.60) == UNKNOWN
    assert vlm_parse.answer_state([0.80, 0.10, 0.10], 0.55, 0.60) == TRUE
    assert vlm_parse.answer_state([0.10, 0.80, 0.10], 0.55, 0.60) == FALSE


def test_both_orders_cancel_letter_bias():
    vlm = SmolVLM.__new__(SmolVLM)                      # no model needed: the letter step is replaced
    vlm.both_orders = True
    vlm._forward_probs = lambda frames, text, n: [0.7] + [0.3 / (n - 1)] * (n - 1)   # always prefers "A"
    p = vlm.choice([], "q", ["yes", "no", "unclear"])
    assert abs(p[0] - p[2]) < 1e-9 and abs(sum(p) - 1) < 1e-9


# ---------------------------------------------------------------- calibration

def _rec(scores, status="confirmed_candidate"):
    return {"candidates": [{"type": t, "score": s, "status": status, "internal": False, "start": 1.0, "end": 2.0}
                           for t, s in scores.items()]}


def _data():
    truth, recs = {}, {}
    for k in range(10):                                  # clean videos: low scores
        truth[f"C{k}"] = {"label": "no_violation", "times": [], "types": set()}
        recs[f"C{k}"] = _rec({"cash_unattended": 0.04 * k})
    for k in range(6):                                   # violations: higher scores
        truth[f"V{k}"] = {"label": "violation", "times": [1.0], "types": {"cash_unattended"}}
        recs[f"V{k}"] = _rec({"cash_unattended": 0.40 + 0.05 * k})
    return truth, recs


def test_fit_respects_max_fpr_and_is_reproducible():
    truth, recs = _data()
    c = cfg(decision={"max_fpr": 0.0})
    th1, th2 = calibration.fit(recs, truth, c), calibration.fit(recs, truth, c)
    assert json.dumps(th1, sort_keys=True) == json.dumps(th2, sort_keys=True)
    assert th1["train"]["fp"] == 0 and th1["train"]["tp"] == 6


def test_loo_never_uses_the_left_out_label():
    truth, recs = _data()
    c = cfg()
    base = {r["video"]: bool(r["events"]) for r in calibration.loo(recs, truth, c)}
    flipped = dict(truth)
    flipped["V0"] = {"label": "no_violation", "times": [], "types": set()}   # change ONLY V0's label
    v0 = next(r for r in calibration.loo(recs, flipped, c) if r["video"] == "V0")
    assert bool(v0["events"]) == base["V0"]


def test_rejected_and_internal_candidates_never_count():
    rec = {"candidates": [{"type": "cash_unattended", "score": 0.9, "status": "rejected", "internal": False},
                          {"type": "key_transfer", "score": 0.9, "status": "confirmed_candidate", "internal": True}]}
    assert calibration.decide(rec, {"global": 0.1, "per_type": {}}, cfg()) == []


def test_whole_video_score_spans_all_types():
    rec = _rec({"cash_unattended": 0.2, "safe_left_open": 0.7})
    assert calibration.video_score(rec) == 0.7


def test_conformal_rank_and_small_sample_warning():
    assert calibration.conformal_p(0.9, [0.1, 0.2, 0.3]) == 0.25
    assert calibration.conformal_p(0.2, [0.1, 0.2, 0.3]) == 0.75
    truth, recs = _data()
    small = {k: v for k, v in recs.items() if k in ("C0", "C1", "V0")}
    out = calibration.conformal_loo(small, truth, 0.1)
    assert any("smallest p-value" in w for w in out["warnings"])
    assert any("descriptive only" in w for w in out["warnings"])


def test_small_clean_set_warning_in_fit():
    truth, recs = _data()
    th = calibration.fit(recs, truth, cfg(decision={"max_fpr": 0.05}))
    assert "warning" in th
