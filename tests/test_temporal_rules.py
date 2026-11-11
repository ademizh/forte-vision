"""Three-valued temporal logic and the violation sequences (cases A-H of the specification)."""
import numpy as np

from fakes import CASH, CASHIER, VISITOR, cfg, det, scenario
from pipeline import calibration, fusion, rules, temporal, tracking, zones
from pipeline.smolvlm import MockVLM
from pipeline.temporal import FALSE, TRUE, UNKNOWN, make

SAFE = [20, 500, 120, 700]


def run(fd, c=None, answers=None):
    c = c or cfg()
    tr = tracking.track_all(fd, c)
    z = zones.find_zones(fd, tr, c, (464, 848))
    st = temporal.build_states(fd, tr, z, c)
    cands = rules.evaluate(st, tr, z, c)
    if answers is None:
        return st, z, cands
    from pipeline import questions, sampler
    jobs, _ = sampler.plan(cands, fd[-1]["t"] + 1, c)
    vlm = MockVLM(answers)
    ans = [vlm.choice([], questions.text(j["qid"]), questions.ANSWERS) for j in jobs]
    return st, z, fusion.fuse(cands, jobs, ans, c)


def types(cands):
    return {x["type"] for x in cands}


# ---------------------------------------------------------------- operators

def test_kleene_and_or_not():
    a = make([0, 1, 2], [TRUE, FALSE, UNKNOWN])
    b = make([0, 1, 2], [UNKNOWN, TRUE, UNKNOWN])
    assert temporal.AND(a, b).v.tolist() == [UNKNOWN, FALSE, UNKNOWN]
    assert temporal.OR(a, b).v.tolist() == [TRUE, TRUE, UNKNOWN]
    assert temporal.NOT(a).v.tolist() == [FALSE, TRUE, UNKNOWN]


def test_bridge_min_run_hysteresis_transitions():
    s = make(range(7), [TRUE, TRUE, UNKNOWN, TRUE, FALSE, TRUE, TRUE])
    assert temporal.bridge(s, 1).v.tolist()[2] == TRUE
    assert temporal.min_run(s, FALSE, 2).v.tolist()[4] == UNKNOWN
    h = temporal.hysteresis(make(range(6), [FALSE, TRUE, FALSE, TRUE, TRUE, TRUE]), 2, 2)
    assert h.v.tolist()[:3] == [FALSE, FALSE, FALSE] and h.v.tolist()[-1] == TRUE
    s2 = make(range(6), [TRUE, TRUE, UNKNOWN, FALSE, FALSE, FALSE])
    assert temporal.transitions(s2, TRUE, FALSE, gap=1) == [3]
    assert temporal.transitions(s2, TRUE, FALSE, gap=0) == []
    assert temporal.persists(s2, FALSE, 3, 3.0) and not temporal.persists(s2, FALSE, 3, 4.0)


def test_interval_operators():
    assert temporal.before((0, 2), (3, 5)) and not temporal.before((0, 4), (3, 5))
    assert temporal.during((3, 4), (2, 6)) and temporal.overlaps((0, 3), (2, 6))
    assert temporal.within((0, 2), (4, 6), 3) and not temporal.within((0, 2), (9, 10), 3)


def test_missing_detection_is_unknown_when_place_is_covered():
    fd = scenario(4, lambda t: [det("cash", CASH)] if t < 2 else [det("person", [300, 400, 380, 460])])
    s = temporal.object_state(fd, "cash", None, 0.25, 0.15)
    assert s.v.tolist() == [TRUE, TRUE, UNKNOWN, UNKNOWN]          # covered by a person: cannot tell
    fd2 = scenario(4, lambda t: [det("cash", CASH)] if t < 2 else [])
    assert temporal.object_state(fd2, "cash", None, 0.25, 0.15).v.tolist() == [TRUE, TRUE, FALSE, FALSE]


# ---------------------------------------------------------------- cases A-D

def _desk(t, cashier=True, cash=True, extra=()):
    ds = []
    if cashier:
        ds.append(det("person", CASHIER, 0.9))
    if cash:
        ds.append(det("cash", CASH, 0.7))
    return ds + list(extra)


def test_case_A_cash_remains_after_cashier_leaves():
    fd = scenario(25, lambda t: _desk(t, cashier=t < 10, cash=True))
    st, z, cands = run(fd)
    cu = [x for x in cands if x["type"] == "cash_unattended"]
    assert len(cu) == 1
    assert 9.0 <= cu[0]["start"] <= 11.0                            # onset = departure, not an earlier frame
    assert all(cd["value"] == TRUE for cd in cu[0]["conditions"])


def test_case_B_cash_removed_before_leaving():
    fd = scenario(25, lambda t: _desk(t, cashier=t < 10, cash=t < 6))
    _, _, cands = run(fd)
    assert "cash_unattended" not in types(cands)


def test_case_C_brief_occlusion_is_not_a_departure():
    fd = scenario(25, lambda t: _desk(t, cashier=t not in (12, 13), cash=True))
    st, _, cands = run(fd)
    assert rules.departures(st, cfg()) == []
    assert "cash_unattended" not in types(cands)


def test_case_D_cash_unknown_after_leaving_needs_review():
    # after the cashier leaves, the cash place is covered by someone walking past: visibility UNKNOWN
    walker = det("person", [300, 400, 380, 470], 0.9)
    fd = scenario(25, lambda t: _desk(t, cashier=t < 16, cash=t < 16, extra=[walker] if t >= 16 else []))
    _, _, fused = run(fd, answers={})
    cu = [x for x in fused if x["type"] == "cash_unattended"]
    assert cu and all(x["status"] == "needs_review" for x in cu)
    assert any(cd["value"] == UNKNOWN for cd in cu[0]["conditions"])


# ---------------------------------------------------------------- cases E-F (internal key transfer)

def test_case_E_key_moves_from_person_a_to_b():
    A, B = [100, 300, 200, 450], [300, 300, 400, 450]

    def fn(t):
        key = [150, 380, 160, 390] if t < 5 else [350, 380, 360, 390]
        return [det("person", A, 0.9), det("person", B, 0.9), det("key", key, 0.6)]
    _, _, cands = run(scenario(10, fn))
    kt = [x for x in cands if x["type"] == "key_transfer"]
    assert kt and kt[0]["internal"] and len(set(kt[0]["persons"])) == 2


def test_case_F_two_people_and_a_key_without_transfer():
    A, B = [100, 300, 200, 450], [300, 300, 400, 450]
    fn = lambda t: [det("person", A, 0.9), det("person", B, 0.9), det("key", [150, 380, 160, 390], 0.6)]  # noqa: E731
    _, _, cands = run(scenario(10, fn))
    assert "key_transfer" not in types(cands)


def test_transfer_not_confirmed_without_visual_evidence():
    A, B = [100, 300, 200, 450], [300, 300, 400, 450]

    def fn(t):
        key = [150, 380, 160, 390] if t < 5 else [350, 380, 360, 390]
        return [det("person", A, 0.9), det("person", B, 0.9), det("key", key, 0.6)]
    _, _, fused = run(scenario(10, fn), answers={"object moving": [0.1, 0.8, 0.1]})
    kt = [x for x in fused if x["type"] == "key_transfer"]
    assert kt and kt[0]["status"] == "rejected"


# ---------------------------------------------------------------- case G (safe duration), H (two events)

def test_case_G_safe_opened_and_closed_quickly_vs_left_open():
    c = cfg()
    base = lambda t, away: [det("safe", SAFE, 0.5)] + ([] if away else [det("person", CASHIER, 0.9)])  # noqa: E731
    short = scenario(20, lambda t: base(t, 10 <= t < 13))           # away 3 s, departure_min 3 -> exactly at limit
    quick = scenario(20, lambda t: base(t, 10 <= t < 11))           # away 1 s: no departure at all
    _, _, c_quick = run(quick, c)
    assert "safe_left_open" not in types(c_quick)
    long_ = scenario(30, lambda t: base(t, t >= 10))
    _, _, fused_yes = run(long_, c, answers={"Is the safe door open": [0.85, 0.1, 0.05]})
    _, _, fused_no = run(long_, c, answers={"Is the safe door open": [0.05, 0.9, 0.05]})
    yes = [x for x in fused_yes if x["type"] == "safe_left_open"][0]
    no = [x for x in fused_no if x["type"] == "safe_left_open"][0]
    assert yes["status"] == "confirmed_candidate" and no["status"] == "rejected"
    assert yes["score"] > no["score"]
    _, _, c_short = run(short, c)
    assert all(x["end"] - x["start"] + 1 >= c["rules"]["safe_left_open"]["min_duration"]
               for x in c_short if x["type"] == "safe_left_open")


def test_case_H_two_events_in_one_video_are_both_kept():
    bag = det("bag", [300, 330, 340, 370], 0.8)
    fd = scenario(30, lambda t: _desk(t, cashier=t < 18, cash=True, extra=[bag] if 2 <= t <= 12 else []))
    _, _, fused = run(fd, answers={"cash remain": [0.8, 0.1, 0.1], "personal item": [0.8, 0.1, 0.1]})
    got = {x["type"] for x in fused if x["status"] != "rejected"}
    assert {"cash_unattended", "personal_items_cash_area"} <= got
    rec = {"candidates": fused}
    ev = calibration.decide(rec, {"global": 0.05, "per_type": {}}, cfg())
    assert {"cash_unattended", "personal_items_cash_area"} <= {e["type"] for e in ev}


def test_high_vlm_score_cannot_override_false_detector_condition():
    fd = scenario(25, lambda t: _desk(t, cashier=t < 10, cash=t < 6))   # cash removed before leaving (case B)
    _, _, fused = run(fd, answers={"cash remain": [0.99, 0.0, 0.01]})
    assert "cash_unattended" not in types(fused)


def test_zones_cashier_is_longest_present_person():
    fd = scenario(20, lambda t: [det("person", CASHIER, 0.9)] + ([det("person", VISITOR, 0.9)] if 5 <= t < 9 else []))
    c = cfg()
    tr = tracking.track_all(fd, c)
    z = zones.find_zones(fd, tr, c, (464, 848))
    cashier = [k for k, v in z["roles"].items() if v == "cashier"]
    visitor = [k for k, v in z["roles"].items() if v == "visitor"]
    assert len(cashier) == 1 and len(visitor) == 1
    st = temporal.build_states(fd, tr, z, c)
    assert st["visitor_present"].v[6] == TRUE and st["visitor_present"].v[15] == FALSE
    assert np.all(st["cashier_present"].v == TRUE)
