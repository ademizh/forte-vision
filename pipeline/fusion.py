"""
Evidence fusion: rule conditions (detector + time) + SmolVLM2 verification -> score, status, explanation.

    fused = fuse(cands, jobs, answers, cfg)

Per candidate
    mandatory question  p = [yes, no, unclear] -> TRUE / FALSE / UNKNOWN (fusion.vlm_yes / vlm_no); a periodic
                        question (door) is TRUE only when 'yes' holds on consecutive samples >= persist_sec
    support question    raises the score, cannot create the event
    contradiction       'yes' lowers the score and sends the event to human review
Score
    rule_score * PROD_mandatory(0.3 + 0.7 * s) * ((1 - w) + w * mean_support) * contradiction * unknown_penalty
    with s = p_yes + 0.5 * p_unclear. A detector condition that was FALSE never reaches this point (rules.py),
    so no VLM answer can override it.
Status
    rejected             a mandatory VLM answer is FALSE
    confirmed_candidate  every condition TRUE (detector and VLM)
    needs_review         anything UNKNOWN, a contradiction, or a VLM-only type (amounts / documents)
The score is not a probability; thresholds come from calibration.py.
"""
from __future__ import annotations

import numpy as np

from . import questions as Q
from .rules import VLM_ONLY
from .vlm_parse import FALSE, TRUE, UNKNOWN, answer_state


def _s(p):
    return float(p[0]) + 0.5 * float(p[2] if len(p) > 2 else 0.0)


def _periodic_state(samples, persist_sec, yes, no):
    """samples: [(t, probs)] in time order -> (state, onset t, mean s)."""
    st = [(t, answer_state(p, yes, no)) for t, p in samples]
    best, run = 0.0, []
    onset = None
    for t, s in st:
        if s == TRUE:
            run.append(t)
            span = run[-1] - run[0]
            if span >= persist_sec - 1e-6 and span >= best:
                best, onset = span, run[0]
        else:
            run = []
    sm = float(np.mean([_s(p) for _, p in samples])) if samples else 0.5
    if onset is not None:
        return TRUE, onset, sm
    if all(s == FALSE for _, s in st):
        return FALSE, None, sm
    return UNKNOWN, None, sm


def fuse(cands: list[dict], jobs: list[dict], answers: list, cfg: dict) -> list[dict]:
    fc = cfg["fusion"]
    yes, no, w, pen = float(fc["vlm_yes"]), float(fc["vlm_no"]), float(fc["vlm_weight"]), float(fc["unknown_penalty"])
    by = {}
    for j, p in zip(jobs, answers):
        by.setdefault(j["cid"], []).append((j, p))
    out = []
    for c in cands:
        c = dict(c)
        c.setdefault("conditions_rule", [dict(x) for x in c["conditions"]])
        conds = [dict(x) for x in c["conditions_rule"]]
        got = by.get(c["cid"], [])
        vlm_rows, mand_factor, support, contra = [], 1.0, [], 1.0
        mand_states = []
        for q in c["questions"]:
            rows = [(j, p) for j, p in got if j["qid"] == q["id"]]
            if not rows:
                vlm_rows.append({"id": q["id"], "role": q["role"], "answered": False, "state": UNKNOWN})
                if q["role"] == "mandatory":
                    mand_states.append(UNKNOWN)
                    mand_factor *= 0.3 + 0.7 * 0.5
                continue
            if c.get("family") == "periodic":
                samples = sorted(((j["times"][0], p) for j, p in rows), key=lambda x: x[0])
                state, onset, s = _periodic_state(samples, float(c.get("persist_sec", 0)), yes, no)
                if onset is not None:
                    c["start"] = round(float(onset), 2)
                probs = [round(float(np.mean([p[k] for _, p in samples])), 4) for k in range(3)]
            else:
                probs = [round(float(np.mean([p[k] for _, p in rows])), 4) for k in range(3)]
                state, s = answer_state(probs, yes, no), _s(probs)
            vlm_rows.append({"id": q["id"], "role": q["role"], "answered": True, "probs": probs, "state": state,
                             "times": sorted({t for j, _ in rows for t in j["times"]}), "question": Q.text(q["id"])})
            if q["role"] == "mandatory":
                mand_states.append(state)
                mand_factor *= 0.3 + 0.7 * s
                for cd in conds:                     # the VLM decides the region condition it was asked about
                    if cd["source"] == "vlm" and cd["value"] == UNKNOWN:
                        cd.update(value=state, conf=round(float(probs[0] if state == TRUE else probs[1]), 3))
                        break
                else:
                    conds.append({"name": q["id"][2:], "value": state, "conf": round(float(max(probs[:2])), 3),
                                  "source": "vlm"})
            elif q["role"] == "support":
                support.append(s)
            else:
                contra *= 1.0 - 0.8 * float(probs[0])
        sup = float(np.mean(support)) if support else 0.5
        any_unknown = any(cd["value"] == UNKNOWN for cd in conds)
        score = c["rule_score"] * mand_factor * ((1 - w) + w * sup) * contra * (pen if any_unknown else 1.0)
        if FALSE in mand_states:
            status = "rejected"
        elif not any_unknown and contra > 0.6 and c["type"] not in VLM_ONLY and c.get("family") != "fallback":
            status = "confirmed_candidate"
        else:
            status = "needs_review"
        c.update(conditions=conds, vlm=vlm_rows, score=round(float(score), 4), status=status,
                 explanation=explain(c["type"], conds, vlm_rows, status))
        out.append(c)
    return out


NAMES_RU = {
    "cashier_was_present": "кассир был на месте", "cashier_left": "кассир ушёл",
    "cash_on_desk_at_departure": "наличные на столе в момент ухода",
    "cash_remains_while_away": "наличные остаются на столе, пока кассира нет",
    "client_arrives": "подходит клиент", "cash_already_on_desk": "наличные уже лежат на столе",
    "cup_on_desk": "чашка на столе", "cash_next_to_cup": "наличные рядом с чашкой",
    "item_on_desk": "посторонний предмет на столе", "stays_long": "лежит долго",
    "second_person_in_cashier_zone": "второй человек в зоне кассира",
    "visitor_with_bag": "посетитель с сумкой", "visit_too_short_for_checks": "визит слишком короткий для проверки",
    "bag_after_visit": "сумка осталась после ухода посетителя", "nobody_at_desk": "за столом никого",
    "safe_open": "сейф открыт", "key_at_safe": "ключ у сейфа", "screen_on": "экран включён",
    "people_present": "в помещении есть люди", "client_left": "клиент ушёл", "cashier_at_desk": "кассир на месте",
    "client_visit": "клиент у кассы", "detector_found_no_event": "детектор не нашёл событий",
}
STATE_RU = {TRUE: "да", FALSE: "нет", UNKNOWN: "не установлено"}


def explain(tid, conds, vlm_rows, status) -> str:
    parts = [f"{NAMES_RU.get(c['name'], c['name'])}: {STATE_RU[c['value']]}"
             f" ({'детектор' if c['source'] == 'detector' else 'SmolVLM2'}, {c['conf']:.2f})" for c in conds]
    qs = [f"«{r['question']}» — {STATE_RU[r['state']]} (да {r['probs'][0]:.2f} / нет {r['probs'][1]:.2f})"
          for r in vlm_rows if r.get("answered")]
    st = {"confirmed_candidate": "все условия подтверждены — подозрение на нарушение",
          "needs_review": "часть условий не установлена — требуется проверка человеком",
          "rejected": "визуальная проверка не подтвердила условие"}[status]
    return "; ".join(parts) + (". Проверка SmolVLM2: " + "; ".join(qs) if qs else "") + f". Итог: {st}."
