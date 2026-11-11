"""
Structured VLM answers: parsing and validation (no torch needed, unit-tested).

    explain_prompt(event_text, n_frames)            -> prompt asking for one JSON object
    parse_explanation(text, n_frames, times)        -> {"ok", "observed", "reason", "supporting_frames",
                                                        "supporting_times", "uncertainty", "errors"}
    answer_state(probs, yes, no)                    -> TRUE / FALSE / UNKNOWN from [p_yes, p_no, p_unclear]

Rules: the JSON is searched in the text (models add words around it); single quotes / trailing commas are
repaired; frame references must point to frames that were actually shown (others are dropped and reported);
a self-reported "confidence" is ignored (it is not a calibrated probability); "observed" must be a boolean
or yes/no, otherwise the answer counts as unclear.
"""
from __future__ import annotations

import json
import re

TRUE, FALSE, UNKNOWN = 1, 0, -1


def explain_prompt(event_text: str, n_frames: int) -> str:
    return (f"The frames (numbered 0 to {n_frames - 1}) come from a bank cash-desk camera. Suspected event: "
            f"{event_text}\nAnswer with ONE JSON object and nothing else, with keys: "
            '"observed" (true or false: is the event visible?), "reason" (one short sentence about what is '
            'visible), "supporting_frames" (list of frame numbers that show it), "uncertainty" (what is unclear).')


def _extract_json(text: str):
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return None
    raw = m.group(0)
    for attempt in (raw, re.sub(r",\s*([}\]])", r"\1", raw.replace("'", '"'))):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def _bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str) and x.strip().lower() in ("true", "yes", "да"):
        return True
    if isinstance(x, str) and x.strip().lower() in ("false", "no", "нет"):
        return False
    return None


def parse_explanation(text: str, n_frames: int, times: list[float] | None = None) -> dict:
    out = {"ok": False, "observed": None, "reason": "", "supporting_frames": [], "supporting_times": [],
           "uncertainty": "", "errors": []}
    obj = _extract_json(text)
    if not isinstance(obj, dict):
        out["errors"].append("no JSON object in the answer")
        return out
    obs = _bool(obj.get("observed"))
    if obs is None:
        out["errors"].append("'observed' missing or not a boolean")
    out["observed"] = obs
    out["reason"] = str(obj.get("reason", ""))[:300]
    out["uncertainty"] = str(obj.get("uncertainty", ""))[:300]
    frames = obj.get("supporting_frames", [])
    if not isinstance(frames, list):
        frames = [frames]
    good = []
    for f in frames:
        try:
            k = int(f)
        except (TypeError, ValueError):
            out["errors"].append(f"frame reference {f!r} is not a number")
            continue
        if 0 <= k < n_frames:
            good.append(k)
        else:
            out["errors"].append(f"frame {k} was not shown (0..{n_frames - 1})")
    out["supporting_frames"] = sorted(set(good))
    if times:
        out["supporting_times"] = [round(float(times[k]), 2) for k in out["supporting_frames"] if k < len(times)]
    if "confidence" in obj:
        out["errors"].append("self-reported confidence ignored")
    out["ok"] = obs is not None
    return out


def answer_state(probs, yes: float, no: float) -> int:
    p_yes, p_no = float(probs[0]), float(probs[1])
    if p_yes >= yes and p_yes > p_no:
        return TRUE
    if p_no >= no and p_no > p_yes:
        return FALSE
    return UNKNOWN
