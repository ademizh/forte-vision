"""
Observations (step 1 of the flow): what OWLv2 and Qwen saw, as a timeline of plain facts.

    observe(st, scene) -> {"events": [{"t", "kind", "en", "ru"}...], "moments": {...}, "tags": [...], "text": "..."}

Events: the cashier leaves / returns, a client arrives / leaves, a cash collector is present, an object appears and
stays (with where and how long), the regions found in this video (safe, laptop, door). No judgement here: the
events are the evidence that the norm search (norms.py) and the condition checks (chain.py) work from.
"""
from __future__ import annotations

import numpy as np

from .objects import Scene
from .state_engine import State, recent_max

RU = {"cash": "наличные", "key": "ключ", "cup": "чашка", "phone": "телефон", "bag": "сумка",
      "handbag": "сумка", "backpack": "рюкзак"}


def _merge(times, gap=4.0):
    out = []
    for t in sorted(times):
        if not out or t - out[-1] >= gap:
            out.append(t)
    return out


def moments(st: State) -> dict[str, list[int]]:
    """Frame indices worth checking: departures, presence (work), the cashier at the desk, client left."""
    n = st.n
    if n == 0:
        return {}
    dep = []
    for i in range(1, n):
        if st.away_s[i] >= 1 and st.away_s[i - 1] < 1:
            dep.append(max(0, i - st.k(2)))                   # away_s is delayed by its 2 s window
        if st.present[i] < 0.3 <= st.present[i - 1]:
            dep.append(max(0, i - 1))
    dep = sorted(set(dep))
    merged = []
    for i in dep:
        if not merged or st.t[i] - st.t[merged[-1]] >= 4:
            merged.append(i)
    enter = [i for i in range(1, n) if st.present[i] >= 0.5 and st.present[i - 1] < 0.3]
    work = [min(n - 1, i + st.k(4)) for i in enter]
    pres = np.where(st.present >= 0.5)[0]
    if len(pres):
        work.append(int(pres[len(pres) // 2]))
    desk = np.where(st.at_desk >= 0.5)[0]
    if len(desk) == 0:
        desk = np.arange(n)
    desk_i = sorted({int(desk[int(q * (len(desk) - 1))]) for q in (0.2, 0.5, 0.8)})
    gone = [i for i in range(1, n) if st.client[i] < 0.3 <= st.client[i - 1]
            and recent_max(st.client, st.k(10))[i] >= 0.5]
    return {"departure": merged[:3], "work": sorted(set(work))[:4], "desk": desk_i, "client_left": gone[:2]}


def observe(st: State, scene: Scene) -> dict:
    ev = []
    t = st.t

    def add(i, kind, en, ru):
        ev.append({"t": float(t[i]), "i": int(i), "kind": kind, "en": en, "ru": ru})

    for i in range(1, st.n):
        if st.away_s[i] >= 1 and st.away_s[i - 1] < 1:
            j = max(0, i - st.k(2))
            add(j, "cashier_leaves", f"{t[j]:.0f}s: the cashier walks away from the desk",
                f"{t[j]:.0f} с: кассир отходит от рабочего места")
        if st.away_s[i] < 1 <= st.away_s[i - 1]:
            add(i, "cashier_returns", f"{t[i]:.0f}s: the cashier is back at the desk", f"{t[i]:.0f} с: кассир вернулся")
        if st.client[i] >= 0.5 > st.client[i - 1]:
            add(i, "client_arrives", f"{t[i]:.0f}s: a client comes to the counter", f"{t[i]:.0f} с: подходит клиент")
        if st.client[i] < 0.3 <= st.client[i - 1]:
            add(i, "client_leaves", f"{t[i]:.0f}s: the client leaves", f"{t[i]:.0f} с: клиент уходит")
    coll = np.where(st.q["collector_present"] >= 0.5)[0]
    if len(coll):
        add(int(coll[0]), "collector", f"{t[coll[0]]:.0f}-{t[coll[-1]]:.0f}s: a cash collector is in the room",
            f"{t[coll[0]]:.0f}–{t[coll[-1]]:.0f} с: в помещении инкассатор")
    if scene.n == st.n:
        for obj, trs in scene.tracks.items():
            for tr in trs[:2]:
                if tr["duration"] < 3:
                    continue
                away = float(np.mean(st.away_s[tr["i0"]:tr["i1"] + 1] >= 1))
                add(tr["i0"], f"object_{obj}",
                    f"{tr['t0']:.0f}-{tr['t1']:.0f}s: OWLv2 sees a {obj} at the same place for {tr['duration']:.0f}s"
                    + (f" ({away:.0%} of that time the cashier is away)" if away > 0 else ""),
                    f"{tr['t0']:.0f}–{tr['t1']:.0f} с: детектор видит {RU.get(obj, obj)} на одном месте "
                    f"{tr['duration']:.0f} с" + (f" (кассир отсутствует {away:.0%} этого времени)" if away > 0 else ""))
    ev.sort(key=lambda e: e["t"])
    tags = sorted({e["kind"] for e in ev} | {f"region_{k}" for k, v in scene.rois.items() if v})
    regions = ", ".join(f"{k} (OWLv2 {v['score']:.2f})" for k, v in scene.rois.items() if v) or "none"
    text = "\n".join(f"- {e['en']}" for e in ev) + f"\n- regions found automatically: {regions}"
    return {"events": ev, "moments": moments(st), "tags": tags, "text": text,
            "text_ru": "; ".join(e["ru"] for e in ev)}
