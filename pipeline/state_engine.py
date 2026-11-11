"""Visual facts -> temporal states and context-sensitive evidence.

Persistence = rolling MINIMUM over the window (a state must hold on every frame), cash/cup/personal/outsider
signals are measured as an elevation above the video's own usual level. On the 40 known videos the mean-support /
absolute-signal variant (v2) scored 62% (3 false alarms) against 75% for this one (0 false alarms), so it was
reverted. Thresholds must be recalibrated after changing this module.
Evidence scores are not calibrated probabilities or proof of a legal breach.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

THRESHOLDS_PATH = Path(os.environ.get("STATE_THRESHOLDS",
                                      Path(__file__).parent.parent / "catalog" / "state_thresholds.json"))

P = {
    "smooth": 3,          # median filter (frames at 1 fps)
    "base_q": 20,         # a detector's "usual level" in this video = this percentile of its scores
    "away_th": 0.35,      # cashier_at_desk below this = not at the desk
    "present_th": 0.5,    # person_visible above this = someone in view
    "recent": 30,         # "away" only counts if the cashier was at the desk within this many seconds
    "gray": 0.08,         # score within this distance below the threshold -> candidate for the verifier
    "margin": 0.05,       # default margin over the highest clean score (calibration)
    "personal_sec": 6,    # a personal item must stay this long
    "visit_max": 8,       # collector/client visit shorter than this around a handover = no time for a check
    "collector_min": 0.65,  # Qwen collector_present must reach this at least once near the handover
}

# Automatic decision only for types whose evidence separates violation from clean videos on the cached data
# (eval/state_eval.py --types). The others are decided by the targeted verifier (pipeline/verify.py): their
# per-frame facts are as high in clean videos (Qwen "door open" 0.73-0.77, "key in lock" 0.62, "safe open" 0.73).
AUTO_TYPES = ["cash_unattended", "previous_cash_on_table", "coffee_over_cash", "personal_items_cash_area",
              "collection_bag_not_inspected", "cash_collection_without_checks", "collection_bag_unsecured",
              "unauthorized_cash_area_entry", "counting_without_customer"]

# Types handled by evidence over cached signals. Types not listed here are verifier-only (see verify.py).
TYPES = [
    "cash_unattended", "previous_cash_on_table", "coffee_over_cash", "personal_items_cash_area",
    "collection_bag_not_inspected", "cash_collection_without_checks", "collection_bag_unsecured",
    "unauthorized_cash_area_entry", "service_door_left_open", "counting_without_customer",
    "safe_key_unattended", "safe_left_open", "unlocked_bank_terminal",
]

DET_NAMES = tuple(x for x in os.environ.get("DETS", "owl").split(",") if x)   # OWLv2 only by default
VERSION = "state-min-v3"


# ---------------------------------------------------------------- signal helpers

def _median(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < 2:
        return x.copy()
    h = w // 2
    return np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])


def sustain(x, k):
    x = np.asarray(x, dtype=float)
    k = max(1, int(k))
    return np.array([x[max(0, i - k + 1):i + 1].min() for i in range(len(x))])


def recent_max(x: np.ndarray, k: int) -> np.ndarray:
    """Maximum over the k frames before i (not including i)."""
    return np.array([x[max(0, i - k):i].max() if i > 0 else 0.0 for i in range(len(x))])


def _clip(x):
    return np.clip(x, 0.0, 1.0)


class State:
    """All signals of one video on a common time axis, plus derived states."""

    def __init__(self, facts: dict, prm: dict | None = None, use_qwen: bool = True, use_det: bool = True):
        self.prm = {**P, **(prm or {})}
        self.t = np.asarray(facts["t"], dtype=float)
        self.n = len(self.t)
        if not np.all(np.isfinite(self.t)) or (self.n > 1 and np.any(np.diff(self.t) <= 0)):
            raise ValueError("Fact timestamps must be finite and strictly increasing")
        self.dt = float(np.median(np.diff(self.t))) if self.n > 1 else 1.0 / float(facts.get("fps", 1.0))
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Invalid fact sampling interval")
        raw = facts["p"]
        for key, values in raw.items():
            a = np.asarray(values, dtype=float)
            if a.shape != (self.n,) or not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1)):
                raise ValueError(f"Invalid fact signal: {key}")
        w = self.prm["smooth"]
        self.use_qwen, self.use_det = use_qwen, use_det

        def q(k):                                    # Qwen fact, smoothed (0 if missing / disabled)
            if not use_qwen or k not in raw:
                return np.zeros(self.n)
            return _median(np.asarray(raw[k], dtype=float), w)

        self.qraw = {k: (np.asarray(raw[k], dtype=float) if use_qwen and k in raw else np.zeros(self.n))
                     for k in ("collector_present",)}
        self.q = {k: q(k) for k in ("person_visible", "cashier_at_desk", "client_at_desk",
                                    "outsider_behind_counter", "collector_present", "cash_on_desk",
                                    "cash_in_hands", "collection_bag_visible", "safe_open", "key_in_safe_lock",
                                    "personal_item_on_desk", "service_door_open", "computer_screen_on",
                                    "drink_over_cash")}
        if use_qwen and "person_visible" not in raw:
            self.q["person_visible"] = np.maximum.reduce([self.q["cashier_at_desk"], self.q["client_at_desk"],
                                                          self.q["outsider_behind_counter"]])
        # detectors: elevation above this video's usual level, averaged over the detectors present
        self.dets = [d for d in DET_NAMES if use_det and f"{d}_cash" in raw]
        self.det = {}
        self.det_absolute = {}
        self.det_raw = {}
        for obj in ("cash", "key", "person", "cup", "phone", "bag"):
            elev = []
            for d in self.dets:
                x = _median(np.asarray(raw.get(f"{d}_{obj}", np.zeros(self.n)), dtype=float), w)
                self.det_raw[f"{d}_{obj}"] = x
                elev.append(np.clip(x - np.percentile(x, self.prm["base_q"]), 0, None) if self.n else x)
            self.det[obj] = np.mean(elev, axis=0) if elev else np.zeros(self.n)
            raw_objects = [self.det_raw[f"{d}_{obj}"] for d in self.dets]
            self.det_absolute[obj] = np.mean(raw_objects, axis=0) if raw_objects else np.zeros(self.n)
        self._derive()

    def k(self, sec: float) -> int:
        return max(1, int(round(sec / self.dt)))

    def qelev(self, key: str) -> np.ndarray:
        """Qwen fact above this video's usual level (for facts whose clean level is not 0, e.g. door)."""
        x = self.q[key]
        return np.clip(x - np.percentile(x, self.prm["base_q"]), 0, None) if self.n else x

    def _derive(self):
        q, prm = self.q, self.prm
        if not self.use_qwen:                        # detector-only ablation: person from detectors
            person = _clip(self.det["person"] * 3)
            self.present = person
            self.at_desk = person
        else:
            self.present = q["person_visible"]
            self.at_desk = q["cashier_at_desk"]
        was_at_desk = recent_max(self.at_desk, self.k(prm["recent"])) >= 0.5
        was_present = recent_max(self.present, self.k(prm["recent"])) >= prm["present_th"]
        # away: the cashier left the desk (was there recently, now isn't) or nobody is in view any more
        self.away = (((self.at_desk < prm["away_th"]) & was_at_desk) |
                     ((self.present < 0.3) & was_present)).astype(float)
        self.away_s = sustain(self.away, self.k(2))
        # cash on the desk: DINO+OWL agreement, Qwen fact as a third vote
        # A dangerous object can be visible for the ENTIRE video. Subtracting
        # the video's own background would erase it. Frozen calibration on
        # labelled normal videos, rather than test-video centering, handles noise.
        self.cash = np.maximum(self.det["cash"], q["cash_on_desk"] if self.use_qwen else 0)
        self.cup = np.maximum(self.det["cup"], q["drink_over_cash"] if self.use_qwen else 0)
        # client present = Qwen client fact, or the collector (also a visitor)
        self.client = np.maximum(q["client_at_desk"], q["collector_present"])


# ---------------------------------------------------------------- evidence per type
# Each returns (score array s(t) in ~[0,1], text template). Score = how strongly the state holds at t,
# sustained over a few seconds. The threshold per type comes from calibration on clean videos.

def ev_cash_unattended(s: State):
    sc = np.minimum(sustain(s.cash, s.k(3)), s.away_s)
    return sc, "кассир уходит от рабочего места {who}, наличные остаются на столе без присмотра (детекторы объектов: {det})"


def ev_previous_cash(s: State):
    # cash already lying on the desk when a (new) client arrives: cash sustained BEFORE the arrival
    arrive = np.zeros(s.n)
    if not s.n:
        return arrive, "к столу подходит следующий клиент, ранее принятые наличные остаются на столе ({det})"
    c = s.client
    for i in range(1, s.n):
        # A first arrival while the cashier prepares cash is a normal operation.
        # Require an observed earlier visit, departure, and then a new arrival.
        earlier = c[max(0, i - s.k(30)):max(0, i - s.k(3))]
        gap = c[max(0, i - s.k(3)):i]
        if c[i] >= 0.5 and c[i - 1] < 0.5 and np.any(earlier >= 0.5) and np.any(gap < 0.3):
            arrive[i] = 1.0
    arrive = np.maximum(arrive, np.r_[arrive[1:], 0])                       # +-1 frame tolerance
    before = sustain(s.cash, s.k(4))
    return np.minimum(before, arrive), "к столу подходит клиент, а на столе уже лежат наличные ({det})"


def ev_coffee(s: State):
    return sustain(s.cup, s.k(2)), "на кассовом столе находится чашка/напиток{cash_note} ({det})"


def ev_personal(s: State):
    # a personal item is STORED at the desk: it must stay for >= 6 s (a phone put down for a moment is normal)
    sc = sustain(s.qelev("personal_item_on_desk"), s.k(s.prm["personal_sec"]))
    bag = sustain(s.det["bag"], s.k(s.prm["personal_sec"])) * (1 - s.q["collector_present"])
    return np.maximum(sc, bag), "на кассовом столе или рядом с ним находится личная вещь (сумка, телефон, одежда)"


def _collector(s: State):
    bag = np.maximum(s.q["collection_bag_visible"], _clip(s.det["bag"] * 2))
    return sustain(np.minimum(s.q["collector_present"], bag), s.k(2))


def visit_seconds(s: State, i: int, span: float = 20.0) -> float:
    """How long a visitor (client or collector) is in the room around frame i."""
    a, b = max(0, i - s.k(span)), min(s.n, i + s.k(span) + 1)
    return float(np.sum(s.client[a:b] >= 0.5)) * s.dt


def ev_collection(s: State):
    # Rules No. 231 p. 24: the bag, seal and document are checked BEFORE the collector leaves. A handover where
    # the visitor is gone after a few seconds leaves no time for that check (X-17: ~3 s; clean X-16: 11 s).
    sc = _collector(s)
    short = np.array([1.0 if visit_seconds(s, i) < s.prm["visit_max"] else 0.0 for i in range(s.n)])
    # a real collector is seen clearly at least once (Qwen 0.7-0.8); a handbag on the desk gives <= 0.5
    k = s.k(3)
    strong = np.array([1.0 if s.qraw["collector_present"][max(0, i - k):i + k + 1].max() >= s.prm["collector_min"]
                       else 0.0 for i in range(s.n)])
    return sc * short * strong, "инкассатор передаёт сумку/ценности и уходит{visit}"


def ev_bag_unsecured(s: State):
    # collection bag still visible after the collector left, nobody at the desk
    bag = np.maximum(s.q["collection_bag_visible"], _clip(s.det["bag"] * 2))
    left = (recent_max(s.q["collector_present"], s.k(30)) >= 0.5) & (s.q["collector_present"] < 0.3)
    return sustain(np.minimum(bag, left * s.away_s), s.k(3)), "инкассаторская сумка остаётся лежать без присмотра"


def ev_outsider(s: State):
    return sustain(s.qelev("outsider_behind_counter"), s.k(2)), "есть признаки нахождения постороннего на стороне кассира"


def ev_door(s: State):
    # door stays open while someone works in the room: mean door score over 6 s while a person is present
    k = s.k(6)
    d = s.q["service_door_open"]
    m = np.array([d[max(0, i - k + 1):i + 1].mean() for i in range(s.n)])
    pres = sustain(s.present, k)
    return np.minimum(m, pres), "дверь кассового помещения остаётся открытой, пока кассир работает"


def ev_count_no_client(s: State):
    gone = (recent_max(s.client, s.k(10)) >= 0.5) & (s.client < 0.3)
    sc = np.minimum(sustain(s.q["cash_in_hands"], s.k(2)), gone.astype(float))
    return sc, "кассир пересчитывает наличные после ухода клиента"


def ev_key(s: State):
    return np.minimum(sustain(s.q["key_in_safe_lock"], s.k(2)), s.away_s), \
        "кассир отходит, ключ остаётся в замке сейфа"


def ev_safe(s: State):
    return np.minimum(sustain(s.q["safe_open"], s.k(2)), s.away_s), "кассир отходит, сейф остаётся открытым"


def ev_terminal(s: State):
    long_away = sustain(s.away, s.k(5))
    return np.minimum(sustain(s.q["computer_screen_on"], s.k(5)), long_away), \
        "кассир покинул рабочее место, экран компьютера остаётся включённым"


EVIDENCE = {
    "cash_unattended": ev_cash_unattended,
    "previous_cash_on_table": ev_previous_cash,
    "coffee_over_cash": ev_coffee,
    "personal_items_cash_area": ev_personal,
    "collection_bag_not_inspected": ev_collection,     # same evidence; the visit length decides the subtype
    "cash_collection_without_checks": ev_collection,
    "collection_bag_unsecured": ev_bag_unsecured,
    "unauthorized_cash_area_entry": ev_outsider,
    "service_door_left_open": ev_door,
    "counting_without_customer": ev_count_no_client,
    "safe_key_unattended": ev_key,
    "safe_left_open": ev_safe,
    "unlocked_bank_terminal": ev_terminal,
}
COLLECTOR_TYPES = ("collection_bag_not_inspected", "cash_collection_without_checks")


# ---------------------------------------------------------------- decision

def load_config(path: Path | str | None = None) -> dict:
    """catalog/state_thresholds.json written by eval/state_eval.py --calibrate."""
    p = Path(path or THRESHOLDS_PATH)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_thresholds(path: Path | str | None = None) -> dict:
    return load_config(path).get("thresholds", {})


def type_scores(facts: dict, prm: dict | None = None, use_qwen=True, use_det=True):
    st = State(facts, prm, use_qwen, use_det)
    out = {}
    for tid, fn in EVIDENCE.items():
        if tid == "cash_collection_without_checks":
            out[tid] = out["collection_bag_not_inspected"]
            continue
        out[tid] = fn(st)
    return st, out


def _onset(s: np.ndarray, th: float, peak: int) -> int:
    """First frame of the run above th that contains the peak (the moment the state began)."""
    i = peak
    while i > 0 and s[i - 1] >= th:
        i -= 1
    return i


def rise_onset(x: np.ndarray, i: int, back: int = 8, frac: float = 0.5) -> int:
    """Walk back from i while the signal stays above frac * x[i]: the moment the object appeared."""
    lvl, j = frac * x[i], i
    while j > 0 and i - j < back and x[j - 1] >= lvl and x[j - 1] > 0:
        j -= 1
    return j


ONSET_SIGNAL = {"cash_unattended": "cash", "previous_cash_on_table": None, "coffee_over_cash": "cup"}


def _det_text(st: State, obj: str, i: int) -> str:
    parts = []
    for d in st.dets:
        x = st.det_raw.get(f"{d}_{obj}")
        if x is not None:
            parts.append(f"{'Grounding DINO' if d == 'dino' else 'OWLv2'} {x[i]:.2f} (обычно {np.percentile(x, 20):.2f})")
    return ", ".join(parts) or "детекторы не подключены"


def evaluate(facts: dict, thresholds: dict | None = None, prm: dict | None = None,
             use_qwen=True, use_det=True, types: list[str] | None = None) -> dict:
    """Events above the calibrated threshold + candidates in the gray zone below it (for the verifier)."""
    prm = {**P, **(prm or {})}
    if thresholds is None:
        cfg = load_config()
        th = cfg.get("thresholds", {})
        if types is None:
            types = cfg.get("enabled", AUTO_TYPES)
    else:
        th = thresholds
    st, scores = type_scores(facts, prm, use_qwen, use_det)
    events, cands = [], []
    if st.n == 0:
        return {"events": [], "candidates": [], "scores": {}, "state": st}
    for tid, (sc, tpl) in scores.items():
        if types is not None and tid not in types:
            continue
        if tid in COLLECTOR_TYPES:
            continue                                   # handled together below
        tau = th.get(tid)
        peak = int(np.argmax(sc))
        top = float(sc[peak])
        rec = {"id": tid, "score": round(top, 3), "threshold": tau, "peak_sec": float(st.t[peak])}
        if tau is not None and top >= tau:
            i0 = _onset(sc, tau, peak)
            sig = ONSET_SIGNAL.get(tid)
            if sig:                                   # timecode = when the object state began, not when
                i0 = rise_onset(getattr(st, sig), i0)  # the evidence became sustained
                if tid == "cash_unattended":
                    i0 = max(i0, rise_onset(st.away, peak))
            events.append({**rec, "timecode_sec": float(st.t[i0]), "start": float(st.t[i0]),
                           "end": float(st.t[peak]),
                           "why": _describe(tid, tpl, st, i0, peak=peak), "margin": round(top - tau, 3)})
        elif tau is None or top >= tau - prm["gray"]:
            cands.append({**rec, "timecode_sec": float(st.t[peak])})
    # collector: one evidence, subtype by visit length (short visit = no inspection of the bag)
    sc, tpl = scores["collection_bag_not_inspected"]
    tau = th.get("collection_bag_not_inspected")
    if (types is None or set(types) & set(COLLECTOR_TYPES)) and st.n:
        peak = int(np.argmax(sc))
        top = float(sc[peak])
        if tau is not None and top >= tau:
            i0 = _onset(sc, tau, peak)
            visit = visit_seconds(st, peak)
            bag_seen = float(np.max(st.q["collection_bag_visible"])) >= 0.5 or float(np.max(st.det["bag"])) >= 0.1
            tid = "collection_bag_not_inspected" if bag_seen else "cash_collection_without_checks"
            events.append({"id": tid, "score": round(top, 3), "threshold": tau, "peak_sec": float(st.t[peak]),
                           "timecode_sec": float(st.t[i0]), "start": float(st.t[i0]), "end": float(st.t[peak]),
                           "why": _describe(tid, tpl, st, i0, visit=visit), "margin": round(top - tau, 3)})
        elif tau is None or top >= tau - prm["gray"]:
            cands.append({"id": "collection_bag_not_inspected", "score": round(top, 3), "threshold": tau,
                          "peak_sec": float(st.t[peak]), "timecode_sec": float(st.t[peak])})
    # a collector visit explains an "outsider" and a "bag" in the same seconds: keep one event
    coll = [e["timecode_sec"] for e in events if e["id"] in COLLECTOR_TYPES]
    events = [e for e in events if not (e["id"] in ("unauthorized_cash_area_entry", "personal_items_cash_area")
                                        and any(abs(e["timecode_sec"] - c) <= 8 for c in coll))]
    events.sort(key=lambda e: e["timecode_sec"])
    return {"events": events, "candidates": cands, "scores": {k: v[0] for k, v in scores.items()}, "state": st}


def _describe(tid: str, tpl: str, st: State, i: int, visit: float | None = None, peak: int | None = None) -> str:
    """Text for the reviewer: what each source measured. i = onset (timecode), peak = strongest moment."""
    j = i if peak is None else peak
    obj = {"cash_unattended": "cash", "previous_cash_on_table": "cash", "coffee_over_cash": "cup"}.get(tid, "cash")
    cash_note = " рядом с наличными" if st.cash[j] > 0.1 else ""
    if tid == "cash_unattended":
        leave = next((k for k in range(i, st.n) if st.away_s[k] >= 1), j)
        who = f"около {st.t[leave]:.0f} с"
        head = f"С {st.t[i]:.0f} с на столе лежат наличные"
    else:
        who, head = "", f"Около {st.t[i]:.0f} с"
    text = tpl.format(who=who, det=_det_text(st, obj, j), cash_note=cash_note,
                      visit=f" (посетитель в помещении около {visit:.0f} с; короткий визит сам по себе "
                            f"не доказывает отсутствие проверки)" if visit is not None else "")
    return f"{head}: {text}."
