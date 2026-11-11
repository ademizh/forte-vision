"""
Detector-guided VLM questions (stage 4 of the judge flow): Qwen looks where DINO and OWLv2 point.

Why
  * The source video is small (464x848). A key in the lock, the safe door or the laptop screen is a few dozen
    pixels in a full frame, and Qwen cuts the image into 28 px patches, so full-frame questions about them
    were answered at chance (door: clean X-23 0.85 > violation X-32 0.72; screen 0.75-0.99 everywhere).
  * Here the detectors find the region first (pipeline/objects.py: safe, laptop, door, cash/key/bag tracks),
    the region is cut out and ENLARGED (long side 560 px), and Qwen gets the close-up plus what the
    detectors and the scene state say (which object, how sure each detector is, the cashier has left,
    how many people are in the room). The question is a single fact about that close-up.
  * Every question is asked twice with the answer letters swapped; P(yes) is the average (no letter bias).
  * One pixel measurement needs no model: does the laptop screen look the same after the cashier leaves as
    while she works? (a locked screen / closed lid changes, an unlocked program does not).

guided(st, scene, frames, vlm) -> {"version", "jobs": [...], "features": {name: {type, score, t, why}}}
Every feature is calibrated on CLEAN videos by the judge (pipeline/judge.py), never used raw.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .frames import Frame
from .objects import Scene
from .state_engine import State, recent_max
from .verify import moments

VERSION = "guided-v1"
CROP_LONG = 560
MAX_AWAY = 5           # away moments asked per video
MAX_TRACKS = 3         # tracks asked per object kind

INTRO = ("This image is an ENLARGED close-up cut from a top-down CCTV frame of a bank cash room "
         "(low resolution, so it can be blurry). ")


# ---------------------------------------------------------------- geometry

def crop(img: Image.Image, box, pad: float = 0.5, min_side: float = 48, out_long: int = CROP_LONG) -> Image.Image:
    w, h = img.size
    if box is None:
        return img
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, min_side), max(y1 - y0, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
    a = (max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)), min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2)))
    if a[2] - a[0] < 4 or a[3] - a[1] < 4:
        return img
    c = img.crop(a)
    s = out_long / max(c.size)
    return c.resize((max(1, int(c.size[0] * s)), max(1, int(c.size[1] * s))), Image.LANCZOS)


def _gray(img: Image.Image, size=(64, 64)) -> np.ndarray:
    return np.asarray(img.convert("L").resize(size, Image.BILINEAR), dtype=float) / 255.0


def _idx(st: State, t: float) -> int:
    return int(np.argmin(np.abs(st.t - t)))


# ---------------------------------------------------------------- moments from the scene state

def away_moments(st: State) -> list[int]:
    """Frames where the cashier is away: the departure itself + samples every 3 s while away."""
    idx = [i for i in range(st.n) if st.away_s[i] >= 1]
    out = []
    for i in idx:
        if not out or st.t[i] - st.t[out[-1]] >= 3:
            out.append(i)
    if len(out) > MAX_AWAY:
        pick = np.linspace(0, len(out) - 1, MAX_AWAY).round().astype(int)
        out = [out[k] for k in pick]
    return out


def client_gone(st: State, scene: Scene) -> np.ndarray:
    """1 where the client has left: Qwen client fact falls, or the detectors' person count drops to 1."""
    q_gone = (recent_max(st.client, st.k(15)) >= 0.5) & (st.client < 0.3)
    if scene.n == st.n and scene.n:
        p = scene.persons
        d_gone = (recent_max(p, st.k(15)) >= 2) & (p <= 1)
        alone = (p <= 1) & (st.client < 0.5)
        return (q_gone | (d_gone & alone)).astype(float)
    return q_gone.astype(float)


# ---------------------------------------------------------------- jobs

def build_jobs(st: State, scene: Scene, frames) -> list[dict]:
    jobs = []
    n = min(st.n, len(frames))
    if n == 0:
        return jobs
    away = [i for i in away_moments(st) if i < n]
    mom = moments(st)
    rois = scene.rois if scene.n == st.n else {}

    def add(feat, tid, i, box, ctx, q, weight=1.0, extra=None):
        jobs.append({"feature": feat, "type": tid, "i": int(i), "t": float(st.t[i]), "box": box,
                     "prompt": INTRO + ctx + " " + q, "weight": float(weight), **(extra or {})})

    # safe: key in the lock / door open, while the cashier is away
    safe = rois.get("safe")
    sctx = (f"Object detectors located the safe here ({safe['prompt']}, score {safe['score']:.2f}"
            f"{', both detectors agree' if safe.get('agree') else ''}). " if safe else
            "The whole frame is shown because the detectors did not locate the safe. ")
    for i in away:
        i2 = min(n - 1, i + st.k(1))
        ctx = sctx + f"At this moment ({st.t[i2]:.0f} s) the cashier has walked away from the desk."
        add("g_key_in_lock", "safe_key_unattended", i2, safe["box"] if safe else None, ctx,
            "Is a key inserted in the lock of the safe (a key or key ring sticking out of the safe door)?")
        add("g_safe_open", "safe_left_open", i2, safe["box"] if safe else None, ctx,
            "Is the door of the safe open, so that the inside of the safe is visible?")
    # key tracks seen while the cashier is away
    for tr in [x for x in scene.tracks.get("key", []) if x["i1"] < n][:MAX_TRACKS]:
        i = max(range(tr["i0"], tr["i1"] + 1), key=lambda k: st.away_s[k])
        w = float(st.away_s[i])
        if w <= 0:
            continue
        add("g_key_box", "safe_key_unattended", i, tr["best_box"],
            f"Object detectors marked a possible key here ({scene.det_text('key', i)}); it is seen for "
            f"{tr['duration']:.0f} s. The cashier is away from the desk at this moment.",
            "Is this a key left inserted in a lock (of a safe or a cabinet)?", w, {"track": tr})
    # door: while someone works in the room
    door = rois.get("door")
    work = sorted({_idx(st, x) for x in mom.get("work", []) + mom.get("desk", [])})[:4]
    for i in work:
        if i >= n:
            continue
        ctx = (f"Object detectors located the door of the room here (score {door['score']:.2f}"
               f"{', both detectors agree' if door.get('agree') else ''})." if door else
               "The whole frame is shown because the detectors did not locate the door.")
        add("g_door_open", "service_door_left_open", i, door["box"] if door else None, ctx,
            "Is this door standing open (a gap or the open doorway is visible), rather than closed?")
    # laptop screen after the cashier left
    lap = rois.get("laptop")
    deps = sorted({_idx(st, x) for x in mom.get("departure", [])})
    for i in deps:
        i2 = min(n - 1, i + st.k(3))
        if st.away_s[i2] < 1 and st.away[i2] < 1:
            continue
        ctx = (f"Object detectors located the laptop here (score {lap['score']:.2f}). " if lap else
               "The whole frame is shown because the detectors did not locate the laptop. ")
        add("g_screen_on", "unlocked_bank_terminal", i2, lap["box"] if lap else None,
            ctx + f"The cashier left the workplace about 3 s before this moment ({st.t[i2]:.0f} s).",
            "Is the laptop open with the screen switched on and showing a working program (windows, text or "
            "tables), not black, not a lock screen, lid not closed?")
    # cash still on the desk while the cashier is away
    for tr in [x for x in scene.tracks.get("cash", []) if x["i1"] < n][:MAX_TRACKS]:
        ks = [k for k in range(tr["i0"], tr["i1"] + 1) if st.away_s[k] >= 1]
        if not ks:
            continue
        i = ks[len(ks) // 2]
        add("g_cash_left", "cash_unattended", i, tr["box"],
            f"Object detectors marked banknotes here ({scene.det_text('cash', i)}), seen for {tr['duration']:.0f} s; "
            f"the cashier is away from the desk at this moment.",
            "Are there banknotes (money) lying here?", 1.0, {"track": tr})
    # personal belongings that stay (not the collector's bag)
    items = []
    for obj in ("handbag", "backpack", "bag", "cup"):
        items += [x for x in scene.tracks.get(obj, []) if x["duration"] >= 5 and x["i1"] < n]
    items.sort(key=lambda x: -x["mean"] * x["frames"])
    for tr in items[:MAX_TRACKS + 1]:
        i = tr["best_i"]
        coll = float(np.max(st.q["collector_present"][tr["i0"]:tr["i1"] + 1]))
        add("g_personal", "personal_items_cash_area", i, tr["box"],
            f"Object detectors marked a {tr['obj']} here ({scene.det_text(tr['obj'], i)}); it stays in place for "
            f"{tr['duration']:.0f} s.",
            "Is this a personal belonging (handbag, backpack, jacket, food or a drink)? Cash, documents, the "
            "laptop, cash devices and a mobile phone do not count.", 1.0 - coll, {"track": tr})
        if tr["obj"] in ("bag", "handbag", "backpack"):
            left = (recent_max(st.q["collector_present"], st.k(30)) >= 0.5) & (st.q["collector_present"] < 0.3)
            w = float(np.max(left[tr["i0"]:tr["i1"] + 1] * st.away_s[tr["i0"]:tr["i1"] + 1]))
            if w > 0:
                add("g_bag_left", "collection_bag_unsecured", i, tr["box"],
                    "Object detectors marked a bag here; the cash collector has left and the cashier is away.",
                    "Is this a cash-collection bag (a sealed bag for transporting money)?", w, {"track": tr})
    # counting after the client left: people count from the detectors, counting from Qwen (full frame)
    gone = client_gone(st, scene)
    gi = [i for i in range(n) if gone[i] >= 1 and st.at_desk[i] >= 0.5]
    picks = []
    for i in gi:
        if not picks or st.t[i] - st.t[picks[-1]] >= 4:
            picks.append(i)
    for i in picks[:3]:
        np_ = int(scene.persons[i]) if scene.n == st.n else -1
        ctx = (f"Object detectors count {np_} person(s) in the room at this moment ({st.t[i]:.0f} s); the client "
               f"has already left." if np_ >= 0 else "The client has already left at this moment.")
        jobs.append({"feature": "g_count_alone", "type": "counting_without_customer", "i": i, "t": float(st.t[i]),
                     "box": None, "weight": 1.0,
                     "prompt": "This is a top-down CCTV frame of a bank cash room. " + ctx +
                               " Is the cashier counting or sorting banknotes in their hands?"})
    return jobs


def ask(jobs: list[dict], frames, vlm, batch: int = 8) -> list[dict]:
    from .vlm import SYSTEM
    prompts = []
    for j in jobs:
        img = crop(frames[j["i"]].image, j["box"]) if j["box"] is not None else frames[j["i"]].image
        fr = [Frame(j["t"], img)]
        prompts.append((SYSTEM, fr, j["prompt"] + "\nA. Yes\nB. No\nAnswer with the letter only."))
        prompts.append((SYSTEM, fr, j["prompt"] + "\nA. No\nB. Yes\nAnswer with the letter only."))
    probs = []
    for s in range(0, len(prompts), batch):
        probs += vlm.choice_probs(prompts[s:s + batch], 2)
    for k, j in enumerate(jobs):
        a, b = probs[2 * k], probs[2 * k + 1]
        j["p_yes"] = round((float(a[0]) + float(b[1])) / 2, 4)
        j["score"] = round(j["p_yes"] * j["weight"], 4)
    return jobs


# ---------------------------------------------------------------- pixel measurement (no model)

def screen_static(st: State, scene: Scene, frames) -> dict | None:
    """1 = the laptop screen after the departure looks like while the cashier worked (not locked)."""
    lap = scene.rois.get("laptop") if scene.n == st.n else None
    n = min(st.n, len(frames))
    if not lap or n < 4:
        return None
    desk = [i for i in range(n) if st.at_desk[i] >= 0.5]
    best = None
    for d in sorted({_idx(st, x) for x in moments(st).get("departure", [])}):
        post = [k for k in range(d + st.k(2), min(n, d + st.k(8))) if st.away_s[k] >= 1 or st.away[k] >= 1]
        ref = [k for k in desk if k < d][-6:]
        if not post or not ref:
            continue
        p = np.mean([_gray(frames[k].image.crop(tuple(int(v) for v in lap["box"]))) for k in post], axis=0)
        change = min(float(np.mean(np.abs(_gray(frames[k].image.crop(tuple(int(v) for v in lap["box"]))) - p)))
                     for k in ref)
        sc = float(np.clip(1 - change / 0.2, 0, 1))
        if best is None or sc > best["score"]:
            best = {"score": round(sc, 4), "t": float(st.t[post[0]]), "change": round(change, 4)}
    return best


# ---------------------------------------------------------------- features

TEXT = {
    "g_key_in_lock": "кассир отошёл, на увеличенном фрагменте сейфа (найден детекторами) виден ключ в замке",
    "g_key_box": "детекторы отмечают ключ, кассир отошёл; на увеличенном фрагменте ключ в замке",
    "g_safe_open": "кассир отошёл, на увеличенном фрагменте сейфа дверца открыта",
    "g_door_open": "на увеличенном фрагменте двери (найдена детекторами) дверь открыта во время работы",
    "g_screen_on": "кассир покинул рабочее место, экран ноутбука включён и показывает программу",
    "c_terminal": "кассир покинул рабочее место, экран ноутбука не заблокирован (изображение не изменилось)",
    "g_cash_left": "кассир отошёл, детекторы и VLM подтверждают наличные на столе",
    "g_personal": "личная вещь (найдена детекторами) остаётся у кассового стола",
    "g_bag_left": "инкассаторская сумка остаётся без присмотра после ухода инкассатора",
    "g_count_alone": "кассир пересчитывает наличные, клиента в помещении уже нет (по детекторам и VLM)",
}
FEATURE_TYPE = {
    "g_key_in_lock": "safe_key_unattended", "g_key_box": "safe_key_unattended", "g_safe_open": "safe_left_open",
    "g_door_open": "service_door_left_open", "g_screen_on": "unlocked_bank_terminal",
    "px_screen_static": "unlocked_bank_terminal", "c_terminal": "unlocked_bank_terminal",
    "g_cash_left": "cash_unattended", "g_personal": "personal_items_cash_area",
    "g_bag_left": "collection_bag_unsecured", "g_count_alone": "counting_without_customer",
}


def features(jobs: list[dict], px: dict | None) -> dict:
    out = {}
    door = [j for j in jobs if j["feature"] == "g_door_open" and "score" in j]
    for j in jobs:
        if "score" not in j or j["feature"] == "g_door_open":
            continue
        f = j["feature"]
        if f not in out or j["score"] > out[f]["score"]:
            out[f] = {"type": j["type"], "score": j["score"], "t": j["t"], "p_yes": j["p_yes"], "box": j["box"]}
    if door:                                   # a door LEFT open: it must look open at most moments
        top = sorted(door, key=lambda j: -j["score"])[:2]
        out["g_door_open"] = {"type": "service_door_left_open", "score": round(float(np.mean([j["score"] for j in top])), 4),
                              "t": min(j["t"] for j in top), "p_yes": top[0]["p_yes"], "box": top[0]["box"]}
    if px:
        out["px_screen_static"] = {"type": "unlocked_bank_terminal", "score": px["score"], "t": px["t"]}
        if "g_screen_on" in out:
            out["c_terminal"] = {"type": "unlocked_bank_terminal", "t": px["t"],
                                 "score": round(min(px["score"], out["g_screen_on"]["score"]), 4)}
    return out


def guided(st: State, scene: Scene, frames, vlm) -> dict:
    jobs = build_jobs(st, scene, frames)
    if jobs:
        ask(jobs, frames, vlm)
    px = screen_static(st, scene, frames)
    for j in jobs:
        j.pop("track", None)
    return {"version": VERSION, "jobs": jobs, "pixel": px, "features": features(jobs, px)}
