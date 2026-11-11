"""
Targeted verifier: ONE specific yes/no question to Qwen about ONE specific moment, for the violation types that the
per-frame facts cannot separate from normal work (door, key in the safe, open safe, unlocked computer, personal
belongings, counting after the client left, cash left behind).

Why this works better than the old judge
  * The old judge picked 1 of 18 options from a 4 s clip with a long text: letter-position bias, prior towards
    "normal" (it rejected 13 of the 14 real violations), and it could veto real detections.
  * Here every question is binary, about a moment chosen from the scene state (the cashier just left, a person
    just came in, the client just left, the cashier sits at the desk), and asked twice with the answer letters
    swapped (A=Yes/B=No and A=No/B=Yes). P(yes) is the average, so the letter bias cancels out.
  * Its thresholds are calibrated on CLEAN videos like the state engine (eval/state_eval.py --verify ...), so a
    question that is also answered "yes" on normal videos cannot raise false alarms.
  * It can only ADD events. It never removes what the state engine found.

verify(facts, frames, vlm) -> {"version", "jobs": [{"type", "moment", "t", "frames", "p_yes"}]}   (cached per video)
decide(rec, thresholds) -> events
"""
from __future__ import annotations

import numpy as np

from .state_engine import State, recent_max

VERSION = "verify-v1"

INTRO_AFTER = "These frames were taken right after the cashier walked away from the desk. "
QUESTIONS = {
    # type: (moment kind, frame offsets in seconds from the moment, question)
    "safe_key_unattended": ("departure", (1, 3),
                            INTRO_AFTER + "Look at the safe (the light grey metal cabinet). "
                            "Is a key left inserted in the lock of the safe?"),
    "safe_left_open": ("departure", (1, 3),
                       INTRO_AFTER + "Look at the safe (the light grey metal cabinet). Is the door of the safe open?"),
    "unlocked_bank_terminal": ("departure", (2, 4),
                               INTRO_AFTER + "Look at the laptop on the desk. Is it open with the screen switched on "
                               "and showing a program (not black, not a lock screen, lid not closed)?"),
    "cash_unattended": ("departure", (1, 3),
                        INTRO_AFTER + "Are banknotes or stacks of money left lying on the desk?"),
    "service_door_left_open": ("work", (0, 2),
                               "Look at the door of this room (a glass door in the wall). Is the door standing open, "
                               "so that the doorway is not closed? A closed glass door does not count."),
    "personal_items_cash_area": ("desk", (0,),
                                 "Is there a personal belonging (handbag, backpack, jacket, food, a drink cup) on the "
                                 "cash desk or on the chair next to it? The cash devices, the laptop, the desk phone "
                                 "and a mobile phone are normal and do not count."),
    "counting_without_customer": ("client_left", (1, 3),
                                  "Is the cashier counting or sorting banknotes while no client is in the room?"),
}
VERIFY_TYPES = list(QUESTIONS)
DEFAULT_THRESHOLD = 0.8          # used only until eval/state_eval.py --calibrate --verify writes real thresholds
MAX_MOMENTS = 4


def _merge(times: list[float], gap: float = 4.0) -> list[float]:
    out = []
    for t in sorted(times):
        if not out or t - out[-1] >= gap:
            out.append(t)
    return out


def moments(st: State) -> dict[str, list[float]]:
    """Moments worth asking about, from the scene state."""
    t, n = st.t, st.n
    if n == 0:
        return {}
    # the cashier walks away: at_desk drops below away_th and stays low >= 2 s, or everybody leaves the view
    dep = []
    for i in range(1, n):
        if st.away_s[i] >= 1 and st.away_s[i - 1] < 1:
            dep.append(max(0, i - st.k(2)))                   # away_s is delayed by its 2 s window
        if st.present[i] < 0.3 <= st.present[i - 1]:          # the last person leaves the view (the room)
            dep.append(max(0, i - 1))
    dep.append(max(0, n - 1 - st.k(3)))                       # always: the final state of the video
    dep_t = _merge([float(t[i]) for i in dep])
    dep_t = (dep_t[:MAX_MOMENTS - 1] + dep_t[-1:]) if len(dep_t) > MAX_MOMENTS else dep_t
    # someone works in the room: shortly after a person enters, and the middle of the longest presence
    enter = [i for i in range(1, n) if st.present[i] >= 0.5 and st.present[i - 1] < 0.3]
    work = [float(t[min(n - 1, i + st.k(4))]) for i in enter]
    pres = np.where(st.present >= 0.5)[0]
    if len(pres):
        work.append(float(t[pres[len(pres) // 2]]))
    work_t = _merge(work)[:MAX_MOMENTS]
    # the cashier at the desk: 3 moments spread over the time at the desk
    desk = np.where(st.at_desk >= 0.5)[0]
    if len(desk) == 0:
        desk = np.arange(n)
    desk_t = _merge([float(t[desk[int(q * (len(desk) - 1))]]) for q in (0.2, 0.5, 0.8)])
    # a client (or collector) leaves
    gone = [i for i in range(1, n) if st.client[i] < 0.3 and recent_max(st.client, st.k(10))[i] >= 0.5
            and st.client[i - 1] >= 0.3]
    left_t = _merge([float(t[i]) for i in gone])[:2] or desk_t   # no visible client change: ask at the desk
    return {"departure": dep_t, "work": work_t, "desk": desk_t, "client_left": left_t}


def build_jobs(st: State, types: list[str] | None = None) -> list[dict]:
    mom = moments(st)
    jobs = []
    for tid, (kind, offs, q) in QUESTIONS.items():
        if types is not None and tid not in types:
            continue
        for m in mom.get(kind, []):
            ft = [min(float(st.t[-1]), max(0.0, m + o)) for o in offs]
            jobs.append({"type": tid, "moment": kind, "t": round(m, 2), "frames": ft, "question": q})
    return jobs


def _nearest(frames, t):
    return min(frames, key=lambda f: abs(f.t - t))


def ask(jobs: list[dict], frames, vlm, batch: int = 4) -> list[dict]:
    """Fills p_yes for every job: average over both letter orders (removes the A/B position bias)."""
    from .vlm import SYSTEM
    prompts = []
    for j in jobs:
        fr = [_nearest(frames, x) for x in j["frames"]]
        prompts.append((SYSTEM, fr, j["question"] + "\nA. Yes\nB. No\nAnswer with the letter only."))
        prompts.append((SYSTEM, fr, j["question"] + "\nA. No\nB. Yes\nAnswer with the letter only."))
    probs = []
    for s in range(0, len(prompts), batch):
        probs += vlm.choice_probs(prompts[s:s + batch], 2)
    for k, j in enumerate(jobs):
        p1, p2 = probs[2 * k], probs[2 * k + 1]
        j["p_yes_ab"], j["p_yes_ba"] = round(float(p1[0]), 4), round(float(p2[1]), 4)
        j["p_yes"] = round((j["p_yes_ab"] + j["p_yes_ba"]) / 2, 4)
    return jobs


def verify(facts: dict, frames, vlm, types: list[str] | None = None) -> dict:
    st = State(facts)
    jobs = ask(build_jobs(st, types), frames, vlm) if frames else []
    return {"version": VERSION, "video": facts.get("video"), "jobs": jobs}


def type_max(rec: dict) -> dict[str, dict]:
    """Best job per type."""
    best = {}
    for j in rec.get("jobs", []):
        if "p_yes" in j and (j["type"] not in best or j["p_yes"] > best[j["type"]]["p_yes"]):
            best[j["type"]] = j
    return best


TEXT = {
    "safe_key_unattended": "кассир отходит от рабочего места, ключ остаётся в замке сейфа",
    "safe_left_open": "кассир отходит от рабочего места, дверца сейфа остаётся открытой",
    "unlocked_bank_terminal": "кассир покинул рабочее место, ноутбук открыт и экран с программой включён",
    "cash_unattended": "кассир покинул рабочее место, на столе остаются наличные",
    "service_door_left_open": "дверь кассового помещения остаётся открытой во время работы",
    "personal_items_cash_area": "на кассовом столе или рядом с ним находится личная вещь",
    "counting_without_customer": "кассир пересчитывает наличные, когда клиента в помещении уже нет",
}


def decide(rec: dict, thresholds: dict | None = None) -> list[dict]:
    th = thresholds or {}
    out = []
    for tid, j in type_max(rec).items():
        tau = th.get(tid, DEFAULT_THRESHOLD)
        if j["p_yes"] >= tau:
            out.append({"id": tid, "timecode_sec": j["t"], "start": j["t"], "end": j["frames"][-1],
                        "score": j["p_yes"], "threshold": tau, "source": "verifier",
                        "why": f"Около {j['t']:.0f} с: {TEXT.get(tid, tid)} (проверка по кадрам "
                               f"{', '.join(f'{x:.0f}' for x in j['frames'])} с, P(да)={j['p_yes']:.2f})."})
    return out
