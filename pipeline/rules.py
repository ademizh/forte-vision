"""
Violation rules as explainable sequences over observable facts (complex-event style).

    cands = evaluate(states, tracks, zones, cfg)      -> [Candidate dict]

Each rule turns the temporal states (pipeline/temporal.py) into zero or more CANDIDATES. A candidate holds:
    type        violation id from catalog/violations.yaml (internal events have internal=True)
    start, end  seconds (start = first verified onset of ALL conditions, not a stale earlier frame)
    conditions  [{"name", "value": TRUE/FALSE/UNKNOWN, "conf", "source": "detector"|"vlm"}]
    rule_score  min confidence of the detector conditions (UNKNOWN counts 0.5)
    questions   VLM questions that must verify it (pipeline/questions.py), with their frame times and crop
    boxes       {time: [real detector boxes]} for the evidence images

A detector condition that is clearly FALSE stops the rule: no candidate is created, so no VLM answer can
"override" it later (fusion.py). UNKNOWN conditions keep the candidate but mark it for human review.

What can and cannot be observed (464x848 top-down CCTV):
    detector + time  : cash_unattended, previous_cash_on_table, coffee_over_cash, personal_items_cash_area,
                       unauthorized_cash_area_entry, collection_bag_not_inspected / cash_collection_without_checks,
                       collection_bag_unsecured, counting_without_customer (+VLM)
    region close-ups : safe_left_open, safe_key_unattended, service_door_left_open, unlocked_bank_terminal (VLM on
                       the region found automatically; detector gives the departure / presence)
    VLM-only, review : incomplete_cash_payout, acceptance_without_documents, cash_count_interrupted,
                       mixed_denominations_stack (amounts / documents are not readable: never "confirmed")
"""
from __future__ import annotations

import numpy as np

from . import questions as Q
from .temporal import FALSE, TRUE, UNKNOWN, Signal, frames, persists, run_end, transitions
from .zones import center

DETECTOR = "detector"


def _cond(name, value, conf, source=DETECTOR) -> dict:
    return {"name": name, "value": int(value), "conf": round(float(conf), 3), "source": source}


def _score(conds) -> float:
    vals = [c["conf"] if c["value"] == TRUE else 0.5 for c in conds if c["source"] == DETECTOR]
    return round(float(min(vals)) if vals else 0.5, 4)


def _cand(tid, start, end, conds, qs, boxes=None, internal=False, family="", focus=None) -> dict:
    return {"type": tid, "start": round(float(start), 2), "end": round(float(end), 2), "conditions": conds,
            "rule_score": _score(conds), "questions": qs, "boxes": boxes or {}, "internal": internal,
            "family": family, "focus": [round(float(x), 2) for x in (focus or [start])]}


def _boxes(sig: Signal, i0: int, i1: int, k: int = 3) -> dict:
    out = {}
    for i in range(i0, i1 + 1):
        if sig.boxes[i]:
            out[f"{sig.t[i]:.2f}"] = sig.boxes[i]
        if len(out) >= k:
            break
    return out


def _times(sig: Signal, i0: int, i1: int, n: int = 4) -> list[float]:
    idx = np.unique(np.linspace(i0, max(i0, i1), num=n).round().astype(int))
    return [float(sig.t[i]) for i in idx]


# ------------------------------------------------------------------ building blocks

def departures(st, cfg) -> list[tuple[int, int, float]]:
    """(departure frame, last frame of the absence, confidence). Absence = cashier FALSE >= departure_min_sec."""
    cp: Signal = st["cashier_present"]
    out = []
    dmin = float(cfg["temporal"]["departure_min_sec"])
    gap = frames(float(cfg["temporal"]["gap_tolerance_sec"]), cp)
    for i in transitions(cp, TRUE, FALSE, gap):
        if persists(cp, FALSE, i, dmin):
            j = run_end(cp, i)
            out.append((i, j, float(np.mean(cp.c[i:j + 1]))))
    return out


def visits(st, cfg, min_sec: float = 2.0) -> list[tuple[int, int, float]]:
    vp: Signal = st["visitor_present"]
    out = []
    for i in range(vp.n):
        if vp.v[i] == TRUE and (i == 0 or vp.v[i - 1] != TRUE):
            j = run_end(vp, i)
            if vp.t[j] - vp.t[i] + 1 >= min_sec:
                out.append((i, j, float(np.mean(vp.c[i:j + 1]))))
    return out


def _count(sig, i0, i1, value):
    return int(np.sum(sig.v[i0:i1 + 1] == value))


# ------------------------------------------------------------------ rules

def rule_cash_unattended(st, cfg):
    p = cfg["rules"]["cash_unattended"]
    cash: Signal = st["cash_visible"]
    cp = st["cashier_present"]
    out = []
    for i, j, dconf in departures(st, cfg):
        pre = range(max(0, i - 2), i + 1)
        at_dep = [cash.v[k] for k in pre]
        # cash removed before the departure: it was seen, then the visible desk shows none
        last_true = max([k for k in range(0, i + 1) if cash.v[k] == TRUE], default=None)
        removed = last_true is not None and _count(cash, last_true, i, FALSE) >= 2 and cash.v[i] == FALSE
        if removed or (TRUE not in at_dep and last_true is None):
            continue                                            # condition 2 or "not removed" is FALSE
        at_val = TRUE if TRUE in at_dep else UNKNOWN
        n_true, n_false = _count(cash, i, j, TRUE), _count(cash, i, j, FALSE)
        first = next((k for k in range(i, j + 1) if cash.v[k] == TRUE), None)
        stays = first is not None and persists(cash, TRUE, first, float(p["min_duration"]))
        if n_false > (j - i + 1) / 2 and not stays:
            continue                                            # the desk is visibly empty while away
        stay_val = TRUE if stays else UNKNOWN
        conds = [_cond("cashier_was_present", TRUE, float(np.max(cp.c[max(0, i - 3):i + 1]))),
                 _cond("cashier_left", TRUE, dconf),
                 _cond("cash_on_desk_at_departure", at_val, max([cash.c[k] for k in pre if cash.v[k] == TRUE],
                                                                default=0.5)),
                 _cond("cash_remains_while_away", stay_val,
                       float(np.mean(cash.c[i:j + 1][cash.v[i:j + 1] == TRUE])) if n_true else 0.5)]
        onset = first if first is not None else i
        t_q = _times(cash, max(0, i - 2), min(j, onset + frames(p["min_duration"], cash) + 1), 5)
        qs = [Q.spec("q_cash_after_leave", "support", t_q), Q.spec("q_cash_put_away", "contradiction", t_q)]
        out.append(_cand("cash_unattended", cash.t[onset], cash.t[j], conds, qs, _boxes(cash, onset, j),
                         family="departure", focus=[cash.t[i], cash.t[onset]]))
    return out


def rule_previous_cash(st, cfg):
    p = cfg["rules"]["previous_cash_on_table"]
    cash, vp = st["cash_visible"], st["visitor_present"]
    out = []
    k = frames(float(p["cash_before_arrival"]), cash)
    for i in transitions(vp, FALSE, TRUE, frames(cfg["temporal"]["gap_tolerance_sec"], vp)):
        win = range(max(0, i - k), i + 1)
        if i - k < 0 or any(cash.v[w] == FALSE for w in win):
            continue
        val = TRUE if all(cash.v[w] == TRUE for w in win) else UNKNOWN
        if val == UNKNOWN and not any(cash.v[w] == TRUE for w in win):
            continue
        conds = [_cond("client_arrives", TRUE, vp.c[i]),
                 _cond("cash_already_on_desk", val, float(np.mean([cash.c[w] for w in win])))]
        tq = [float(cash.t[max(0, i - k)]), float(cash.t[i]), float(cash.t[min(cash.n - 1, i + 1)])]
        out.append(_cand("previous_cash_on_table", cash.t[i], cash.t[min(cash.n - 1, i + 2)], conds,
                         [Q.spec("q_cash_before_client", "support", tq)], _boxes(cash, max(0, i - k), i),
                         family="visit", focus=[cash.t[i]]))
    return out


def _near(a, b, factor) -> bool:
    ca, cb = center(a), center(b)
    size = max(a[2] - a[0], a[3] - a[1], 1.0)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])) <= factor * size


def rule_coffee(st, cfg):
    p = cfg["rules"]["coffee_over_cash"]
    cup, cash = st.get("cup_visible"), st["cash_visible"]
    if cup is None:
        return []
    near = np.zeros(cup.n, dtype=bool)
    for i in range(cup.n):
        near[i] = any(_near(a["box"], b["box"], float(p["near_factor"])) for a in cup.boxes[i] for b in cash.boxes[i])
    out, i = [], 0
    k = frames(float(p["min_duration"]), cup)
    while i < cup.n:
        if near[i]:
            j = i
            while j + 1 < cup.n and near[j + 1]:
                j += 1
            if j - i + 1 >= k:
                conds = [_cond("cup_on_desk", TRUE, float(np.mean(cup.c[i:j + 1]))),
                         _cond("cash_next_to_cup", TRUE, float(np.mean(cash.c[i:j + 1])))]
                bx = {f"{cup.t[x]:.2f}": cup.boxes[x] + cash.boxes[x] for x in range(i, min(j, i + 2) + 1)}
                out.append(_cand("coffee_over_cash", cup.t[i], cup.t[j], conds,
                                 [Q.spec("q_cup", "support", _times(cup, i, j, 3))], bx, family="object"))
            i = j + 1
        else:
            i += 1
    return out


def rule_personal_items(st, cfg):
    p = cfg["rules"]["personal_items_cash_area"]
    item: Signal = st["item_visible"]
    out = []
    for i in range(item.n):
        if item.v[i] == TRUE and (i == 0 or item.v[i - 1] != TRUE) and persists(item, TRUE, i, float(p["min_duration"])):
            j = run_end(item, i)
            box = item.boxes[i][0]["box"] if item.boxes[i] else None
            conds = [_cond("item_on_desk", TRUE, float(np.mean(item.c[i:j + 1]))),
                     _cond("stays_long", TRUE, 1.0)]
            tq = _times(item, i, j, 3)
            out.append(_cand("personal_items_cash_area", item.t[i], item.t[j], conds,
                             [Q.spec("q_item", "support", tq, crop=box),
                              Q.spec("q_is_collector", "contradiction", tq)], _boxes(item, i, j), family="object"))
    return out


def rule_outsider(st, cfg):
    p = cfg["rules"]["unauthorized_cash_area_entry"]
    s: Signal = st["outsider_in_cashier_zone"]
    out = []
    for i in range(s.n):
        if s.v[i] == TRUE and (i == 0 or s.v[i - 1] != TRUE) and persists(s, TRUE, i, float(p["min_duration"])):
            j = run_end(s, i)
            conds = [_cond("second_person_in_cashier_zone", TRUE, float(np.mean(s.c[i:j + 1])))]
            tq = _times(s, i, j, 4)
            cp = st["cashier_present"]
            out.append(_cand("unauthorized_cash_area_entry", s.t[i], s.t[j], conds,
                             [Q.spec("q_outsider", "mandatory", tq), Q.spec("q_is_collector", "contradiction", tq)],
                             _boxes(cp, i, j), family="person"))
    return out


def rule_collection(st, cfg):
    p = cfg["rules"]["collection_bag_not_inspected"]
    bag = st.get("bag_anywhere")
    out = []
    if bag is None:
        return out
    for i, j, vconf in visits(st, cfg):
        dur = float(st["t"][j] - st["t"][i] + 1)
        nb = _count(bag, i, j, TRUE)
        if nb == 0:
            continue
        short = dur <= float(p["max_visit"])
        conds = [_cond("visitor_with_bag", TRUE, min(vconf, float(np.max(bag.c[i:j + 1])))),
                 _cond("visit_too_short_for_checks", TRUE if short else UNKNOWN, 1.0 if short else 0.5)]
        tq = _times(bag, max(0, i - 1), min(bag.n - 1, j + 1), 5)
        out.append(_cand("collection_bag_not_inspected", st["t"][i], st["t"][j], conds,
                         [Q.spec("q_collector_handover", "mandatory", tq)], _boxes(bag, i, j), family="visit"))
    return out


def rule_bag_unsecured(st, cfg):
    p = cfg["rules"]["collection_bag_unsecured"]
    bag, cp, vp = st.get("bag_anywhere"), st["cashier_present"], st["visitor_present"]
    if bag is None:
        return []
    out = []
    for i, j, _ in visits(st, cfg):
        if _count(bag, i, j, TRUE) == 0:
            continue
        k0 = j + 1
        k = k0
        while k < bag.n and bag.v[k] != FALSE and vp.v[k] != TRUE:
            k += 1
        if k0 >= bag.n:
            continue
        span = [x for x in range(k0, min(k, bag.n)) if bag.v[x] == TRUE and cp.v[x] != TRUE]
        if span and bag.t[span[-1]] - bag.t[span[0]] + 1 >= float(p["min_duration"]):
            conds = [_cond("bag_after_visit", TRUE, float(np.mean([bag.c[x] for x in span]))),
                     _cond("nobody_at_desk", TRUE if all(cp.v[x] == FALSE for x in span) else UNKNOWN, 0.8)]
            out.append(_cand("collection_bag_unsecured", bag.t[span[0]], bag.t[span[-1]], conds,
                             [Q.spec("q_bag_left", "support", _times(bag, span[0], span[-1], 3))],
                             _boxes(bag, span[0], span[-1]), family="visit"))
    return out


def _region(zones, *names):
    for n in names:
        r = zones["rois"].get(n)
        if r:
            return n, r["box"]
    return None, None


def rule_departure_regions(st, zones, cfg):
    """Safe open / key in the safe / screen on while the cashier is away: the detector gives the departure,
    the VLM checks the region close-up (DINO cannot see 'open' or 'key in lock' at this resolution)."""
    out = []
    cp = st["cashier_present"]
    key = st.get("key_anywhere")
    _, safe = _region(zones, "safe")
    _, screen = _region(zones, "monitor", "laptop")
    for i, j, dconf in departures(st, cfg):
        dur = float(cp.t[j] - cp.t[i] + 1)
        tq = _times(cp, i, j, 4)
        left = _cond("cashier_left", TRUE, dconf)
        if dur >= float(cfg["rules"]["safe_left_open"]["min_duration"]):
            out.append(_cand("safe_left_open", cp.t[i], cp.t[j], [left, _cond("safe_open", UNKNOWN, 0.5, "vlm")],
                             [Q.spec("q_safe_open", "mandatory", tq, crop=safe or "frame")], family="departure"))
        if dur >= float(cfg["rules"]["safe_key_unattended"]["min_duration"]):
            near = False
            if key is not None and safe is not None:
                f = float(cfg["rules"]["safe_key_unattended"]["near_safe_factor"])
                near = any(_near(safe, b["box"], f) for x in range(i, j + 1) for b in key.boxes[x])
            kc = _cond("key_at_safe", TRUE if near else UNKNOWN, 0.6 if near else 0.5)
            out.append(_cand("safe_key_unattended", cp.t[i], cp.t[j], [left, kc],
                             [Q.spec("q_key_in_lock", "mandatory", tq, crop=safe or "frame")],
                             _boxes(key, i, j) if key is not None and near else {}, family="departure"))
        if dur >= float(cfg["rules"]["unlocked_bank_terminal"]["min_duration"]) and screen is not None:
            out.append(_cand("unlocked_bank_terminal", cp.t[i], cp.t[j], [left, _cond("screen_on", UNKNOWN, 0.5, "vlm")],
                             [Q.spec("q_screen_on", "mandatory", tq, crop=screen)], family="departure"))
    return out


def rule_door(st, zones, cfg):
    p = cfg["rules"]["service_door_left_open"]
    _, door = _region(zones, "door")
    if door is None:
        return []
    t = st["t"]
    people = st["people"]
    step = float(p["step"])
    times = [float(x) for x in np.arange(t[0], t[-1] + 1e-6, step)
             if people[int(np.argmin(np.abs(t - x)))] >= 1]
    if len(times) < 2:
        return []
    cap = int(p.get("max_samples", 12))
    if len(times) > cap:                                  # bounded VLM cost on long videos
        times = [times[i] for i in np.unique(np.linspace(0, len(times) - 1, num=cap).round().astype(int))]
    c = _cand("service_door_left_open", times[0], times[-1], [_cond("people_present", TRUE, 0.8)],
              [Q.spec("q_door_open", "mandatory", times, crop=door)], family="periodic")
    c["persist_sec"] = float(p["min_duration"])           # fusion: the door must be open on consecutive samples
    return [c]


def rule_counting(st, cfg):
    p = cfg["rules"]["counting_without_customer"]
    vp, cp = st["visitor_present"], st["cashier_present"]
    out = []
    w = frames(float(p["within_sec"]), vp)
    for i in transitions(vp, TRUE, FALSE, frames(cfg["temporal"]["gap_tolerance_sec"], vp)):
        j = min(vp.n - 1, i + w)
        if _count(vp, i, j, TRUE) > 0 or _count(cp, i, j, TRUE) == 0:
            continue
        conds = [_cond("client_left", TRUE, vp.c[i]), _cond("cashier_at_desk", TRUE, float(np.max(cp.c[i:j + 1])))]
        out.append(_cand("counting_without_customer", vp.t[i], vp.t[j], conds,
                         [Q.spec("q_counting", "mandatory", _times(vp, i, j, 5))], family="visit"))
    return out


VLM_ONLY = {"incomplete_cash_payout": "q_partial_payout", "acceptance_without_documents": "q_no_document",
            "cash_count_interrupted": "q_count_interrupted", "mixed_denominations_stack": "q_mixed_stack"}


def rule_vlm_only(st, cfg):
    """Amounts / documents / denominations cannot be measured by the detector: one VLM question per client
    visit; the result is at most 'needs human review' (conditions stay UNKNOWN for the detector)."""
    out = []
    for i, j, vconf in visits(st, cfg, min_sec=4.0):
        tq = _times(st["visitor_present"], i, j, 5)
        for tid, qid in VLM_ONLY.items():
            conds = [_cond("client_visit", TRUE, vconf), _cond(qid[2:], UNKNOWN, 0.5, "vlm")]
            out.append(_cand(tid, st["t"][i], st["t"][j], conds, [Q.spec(qid, "mandatory", tq)], family="vlm_only"))
    return out


def rule_key_transfer(tracks, zones, cfg):
    """Internal event (not a competition label): the key is first next to person A and later next to a
    different person B, each for >= 2 frames. Small keys jump between frames at 1 fps, so the key's own track
    identity is NOT required: all key boxes are used in time order (the best one per frame). Proximity alone is
    not proof: the candidate only exists to be verified by q_transfer (mandatory)."""
    persons = tracks.get("person", [])
    per_frame = {}
    for kt in tracks.get("key", []):
        for o in kt["obs"]:
            if o["i"] not in per_frame or o["score"] > per_frame[o["i"]]["score"]:
                per_frame[o["i"]] = o
    owners = []
    for i in sorted(per_frame):
        o = per_frame[i]
        best, bd = None, 1e9
        for p in persons:
            pb = next((x["box"] for x in p["obs"] if x["i"] == i), None)
            if pb is None:
                continue
            d = float(np.hypot(*(np.subtract(center(pb), center(o["box"])))))
            if d < bd and d <= 0.6 * max(pb[2] - pb[0], pb[3] - pb[1]):
                best, bd = p["id"], d
        if best:
            owners.append((o["t"], best))
    runs = []
    for t, w in owners:
        if runs and runs[-1][1] == w:
            runs[-1][2] = t
            runs[-1][3] += 1
        else:
            runs.append([t, w, t, 1])
    runs = [r for r in runs if r[3] >= 2]
    out = []
    for a, b in zip(runs, runs[1:]):
        if a[1] != b[1]:
            ta, tb = a[2], b[0]
            out.append(_cand("key_transfer", ta, tb, [_cond("key_near_person_a", TRUE, 0.6),
                                                      _cond("key_near_person_b", TRUE, 0.6)],
                             [Q.spec("q_transfer", "mandatory", [ta, (ta + tb) / 2, tb])], internal=True,
                             family="object"))
            out[-1]["persons"] = [a[1], b[1]]
            break
    return out


def evaluate(st: dict, tracks: dict, zones: dict, cfg: dict) -> list[dict]:
    cands = []
    cands += rule_cash_unattended(st, cfg)
    cands += rule_previous_cash(st, cfg)
    cands += rule_coffee(st, cfg)
    cands += rule_personal_items(st, cfg)
    cands += rule_outsider(st, cfg)
    cands += rule_collection(st, cfg)
    cands += rule_bag_unsecured(st, cfg)
    cands += rule_departure_regions(st, zones, cfg)
    cands += rule_door(st, zones, cfg)
    cands += rule_counting(st, cfg)
    cands += rule_vlm_only(st, cfg)
    cands += rule_key_transfer(tracks, zones, cfg)
    for k, c in enumerate(cands):
        c["cid"] = f"c{k:03d}"
    return cands
