"""
Condition chains (step 3): Qwen checks, condition by condition, whether a found rule really applies.

For every rule returned by the norm search (norms.py) that this video gives a reason to check, a chain of
conditions is built from the observations. Example, key left in the safe:

    the object is a key          Qwen: sequence of enlarged crops of the tracked object / the safe region
    the key is in the lock       Qwen: the same crops after the departure
    the cashier walked away      scene state (Qwen per-frame facts + OWLv2 people), no extra question
    the key stays in the lock    Qwen: crop at the last moment the cashier is still away
    the rule applies             Qwen: two full frames + the observation timeline + the rule text with its
                                 conditions and exceptions -> "are all conditions met and no exception?"

Chain score = the weakest condition (fuzzy AND): one unconfirmed condition is enough to reject the chain.
Every Qwen question is asked with both answer orders (A=Yes/B=No and A=No/B=Yes) and averaged.
The judge (judge.py) calibrates every chain on CLEAN videos and switches off a chain that also fires there.

run(st, scene, obs, frames, vlm) -> {"version", "retrieved": [...], "chains": [...], "features": {...}}
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

from . import norms
from .frames import Frame
from .objects import Scene
from .state_engine import State, recent_max

VERSION = "chain-v1"
CROP_LONG = 448
MAX_DEP = int(os.environ.get("CHAIN_MAX_DEPARTURES", "2"))
CHAIN_TYPES = ["safe_key_unattended", "safe_left_open", "unlocked_bank_terminal", "cash_unattended",
               "personal_items_cash_area", "counting_without_customer", "service_door_left_open",
               "collection_bag_unsecured"]
INTRO = "These images are ENLARGED close-ups cut from a low-resolution top-down CCTV frame of a bank cash room. "


# ---------------------------------------------------------------- images

def crop(img: Image.Image, box, pad: float = 0.6, min_side: float = 40, out_long: int = CROP_LONG) -> Image.Image:
    if box is None:
        return img
    w, h = img.size
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, min_side) * (1 + 2 * pad), max(y1 - y0, min_side) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a = (max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)), min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2)))
    if a[2] - a[0] < 4 or a[3] - a[1] < 4:
        return img
    c = img.crop(a)
    s = out_long / max(c.size)
    return c.resize((max(1, int(c.size[0] * s)), max(1, int(c.size[1] * s))), Image.LANCZOS)


def marked(img: Image.Image, box) -> Image.Image:
    if box is None:
        return img
    im = img.copy()
    ImageDraw.Draw(im).rectangle([int(v) for v in box], outline=(255, 40, 40), width=3)
    return im


def _gray(img, size=(64, 64)):
    return np.asarray(img.convert("L").resize(size, Image.BILINEAR), dtype=float) / 255.0


# ---------------------------------------------------------------- helpers over the scene

class Ctx:
    def __init__(self, st: State, scene: Scene, obs: dict, n: int):
        self.st, self.scene, self.obs, self.n = st, scene, obs, n
        self.rois = scene.rois if scene.n == st.n else {}

    def c(self, i):
        return int(min(self.n - 1, max(0, i)))

    def away_frac(self, a, b):
        a, b = self.c(a), self.c(b)
        return float(np.mean(self.st.away_s[a:b + 1] >= 1)) if b >= a else 0.0

    def track_near(self, objs, a, b, region=None):
        """Best track of these objects seen between frames a..b (inside the region, if given)."""
        best, bs = None, 0.0
        for obj in objs:
            for tr in self.scene.tracks.get(obj, []):
                pts = [p for p in tr["path"] if a <= p[0] <= b]
                if not pts:
                    continue
                if region is not None:
                    rx0, ry0, rx1, ry1 = region
                    pw, ph = (rx1 - rx0) * 0.5, (ry1 - ry0) * 0.5
                    cx, cy = (pts[0][1][0] + pts[0][1][2]) / 2, (pts[0][1][1] + pts[0][1][3]) / 2
                    if not (rx0 - pw <= cx <= rx1 + pw and ry0 - ph <= cy <= ry1 + ph):
                        continue
                s = tr["mean"] * len(pts)
                if s > bs:
                    best, bs = tr, s
        return best

    def target(self, tr, roi_name, i):
        if tr is not None:
            return self.scene.box_at(tr, i)
        r = self.rois.get(roi_name)
        return r["box"] if r else None


def V(name, ru, images, question):
    return {"name": name, "ru": ru, "kind": "vlm", "images": images, "question": question}


def S(name, ru, p):
    return {"name": name, "ru": ru, "kind": "state", "p": round(float(np.clip(p, 0, 1)), 4)}


def R(norm, images):
    return {"name": "rule_applies", "ru": "норма применима к ситуации", "kind": "rule", "norm": norm["id"],
            "images": images}


# ---------------------------------------------------------------- chains per rule

def _where(ctx, tr, roi):
    if tr is not None:
        return f" (OWLv2 tracked a '{tr['obj']}' there for {tr['duration']:.0f} s)"
    if ctx.rois.get(roi):
        return f" (OWLv2 located the {roi} there, score {ctx.rois[roi]['score']:.2f})"
    return " (the region was not located, the whole frame is shown)"


def chains_key(ctx, norm):
    out = []
    safe = ctx.rois.get("safe")
    for d in ctx.obs["moments"].get("departure", [])[:MAX_DEP]:
        tr = ctx.track_near(["key"], d - 4, d + 8, safe["box"] if safe else None)
        seq = [(ctx.c(d - 1), ctx.target(tr, "safe", d - 1), "crop"), (ctx.c(d + 2), ctx.target(tr, "safe", d + 2), "crop"),
               (ctx.c(d + 5), ctx.target(tr, "safe", d + 5), "crop")]
        last = max([k for k in range(d, ctx.c(d + 10) + 1) if ctx.st.away_s[k] >= 1] or [ctx.c(d + 3)])
        w = _where(ctx, tr, "safe")
        out.append({"type": "safe_key_unattended", "i": d, "box": seq[1][1], "steps": [
            V("is_key", "предмет — ключ", seq, INTRO + f"They show the same place at three moments{w}. "
              "Is a key (a small metal key or a key ring) visible?"),
            V("in_lock", "ключ в замке сейфа", seq[1:], INTRO + f"The cashier has just walked away{w}. "
              "Is a key inserted in the lock of the safe, sticking out of the safe door?"),
            S("away", "кассир ушёл от рабочего места", ctx.away_frac(d + 1, d + 6)),
            V("stays", "ключ остаётся в замке без присмотра", [(last, ctx.target(tr, "safe", last), "crop")],
              INTRO + "The cashier is still away from the desk at this moment. Is a key still in the lock of the safe?"),
            R(norm, [(ctx.c(d - 1), None, "full"), (last, seq[1][1], "mark")])]})
    return out


def chains_safe_open(ctx, norm):
    out = []
    for d in ctx.obs["moments"].get("departure", [])[:MAX_DEP]:
        box = ctx.target(None, "safe", d)
        seq = [(ctx.c(d + 2), box, "crop"), (ctx.c(d + 5), box, "crop")]
        w = _where(ctx, None, "safe")
        out.append({"type": "safe_left_open", "i": d, "box": box, "steps": [
            V("open", "дверца сейфа открыта", seq, INTRO + f"The cashier has walked away{w}. "
              "Is the door of the safe open, so that the inside of the safe is visible?"),
            S("away", "кассир ушёл от сейфа", ctx.away_frac(d + 1, d + 6)),
            R(norm, [(ctx.c(d - 1), None, "full"), (ctx.c(d + 4), box, "mark")])]})
    return out


def screen_unchanged(ctx, frames, d, box):
    """1 = the screen region after the departure looks like while the cashier worked (not locked / closed)."""
    if box is None:
        return 0.5
    st, n = ctx.st, ctx.n
    post = [k for k in range(d + st.k(2), min(n, d + st.k(8))) if st.away_s[k] >= 1 or st.away[k] >= 1]
    ref = [k for k in range(n) if st.at_desk[k] >= 0.5 and k < d][-6:]
    if not post or not ref:
        return 0.5
    b = tuple(int(v) for v in box)
    p = np.mean([_gray(frames[k].image.crop(b)) for k in post], axis=0)
    change = min(float(np.mean(np.abs(_gray(frames[k].image.crop(b)) - p))) for k in ref)
    return float(np.clip(1 - change / 0.2, 0, 1))


def chains_terminal(ctx, norm, frames):
    out = []
    for d in ctx.obs["moments"].get("departure", [])[:MAX_DEP]:
        box = ctx.target(None, "laptop", d)
        seq = [(ctx.c(d + 3), box, "crop"), (ctx.c(d + 6), box, "crop")]
        w = _where(ctx, None, "laptop")
        out.append({"type": "unlocked_bank_terminal", "i": d, "box": box, "steps": [
            V("screen_on", "экран ноутбука включён, открыта программа", seq, INTRO + f"The cashier left the desk "
              f"a few seconds earlier{w}. Is the laptop open with the screen switched on and showing a working "
              "program (windows, text or tables), not black, not a lock screen, lid not closed?"),
            S("not_locked", "изображение экрана не изменилось после ухода (не заблокирован)",
              screen_unchanged(ctx, frames, d, box)),
            S("away", "кассир отсутствует", ctx.away_frac(d + 2, d + 8)),
            R(norm, [(ctx.c(d - 1), box, "mark"), (ctx.c(d + 4), box, "mark")])]})
    return out


def chains_cash(ctx, norm):
    out = []
    for d in ctx.obs["moments"].get("departure", [])[:MAX_DEP]:
        tr = ctx.track_near(["cash"], d, d + 10)
        if tr is None:
            continue
        ks = [p[0] for p in tr["path"] if d <= p[0] <= d + 10 and ctx.st.away_s[p[0]] >= 1]
        if not ks:
            continue
        seq = [(k, ctx.scene.box_at(tr, k), "crop") for k in (ks[0], ks[len(ks) // 2], ks[-1])]
        out.append({"type": "cash_unattended", "i": d, "box": seq[0][1], "steps": [
            V("is_cash", "на столе лежат наличные", seq, INTRO + f"OWLv2 marked banknotes here for {tr['duration']:.0f} s. "
              "Are there banknotes (money) lying here?"),
            S("away", "кассир ушёл", ctx.away_frac(d + 1, d + 6)),
            S("stays", "наличные остаются без присмотра", len(ks) / max(1, sum(ctx.st.away_s[d:d + 11] >= 1))),
            R(norm, [(ctx.c(d - 1), None, "full"), (ks[-1], seq[-1][1], "mark")])]})
    return out


def chains_personal(ctx, norm):
    items = []
    for obj in ("handbag", "backpack", "bag", "cup"):
        items += [x for x in ctx.scene.tracks.get(obj, []) if x["duration"] >= 5 and x["i1"] < ctx.n]
    items.sort(key=lambda x: -x["mean"] * x["frames"])
    out = []
    for tr in items[:2]:
        mid = tr["path"][len(tr["path"]) // 2][0]
        seq = [(tr["i0"], ctx.scene.box_at(tr, tr["i0"]), "crop"), (mid, ctx.scene.box_at(tr, mid), "crop"),
               (tr["i1"], ctx.scene.box_at(tr, tr["i1"]), "crop")]
        coll = float(np.max(ctx.st.q["collector_present"][tr["i0"]:tr["i1"] + 1]))
        out.append({"type": "personal_items_cash_area", "i": tr["i0"], "box": tr["box"], "steps": [
            V("is_personal", "предмет — личная вещь", seq, INTRO + f"OWLv2 tracked a '{tr['obj']}' here for "
              f"{tr['duration']:.0f} s. Is it a personal belonging (handbag, backpack, jacket, food or a drink)? "
              "Cash, documents, the laptop, cash devices and a mobile phone do not count."),
            V("at_desk", "вещь на кассовом столе или рядом", [(mid, tr["box"], "mark")],
              "Top-down CCTV frame of a bank cash room. Is the object in the red box on the cash desk or right "
              "next to it?"),
            S("stays", "вещь остаётся на месте", min(1.0, tr["duration"] / 8.0)),
            S("not_collector", "это не сумка инкассатора", 1.0 - coll),
            R(norm, [(mid, tr["box"], "mark")])]})
    return out


def chains_counting(ctx, norm):
    st, sc = ctx.st, ctx.scene
    gone_q = (recent_max(st.client, st.k(15)) >= 0.5) & (st.client < 0.3)
    if sc.n == st.n:
        gone_d = (recent_max(sc.persons, st.k(15)) >= 2) & (sc.persons <= 1) & (st.client < 0.5)
        gone = gone_q | gone_d
    else:
        gone = gone_q
    starts = [i for i in range(1, ctx.n) if gone[i] and not gone[i - 1] and st.at_desk[i] >= 0.5]
    out = []
    for g in starts[:2]:
        np_ = int(sc.persons[g]) if sc.n == st.n else -1
        out.append({"type": "counting_without_customer", "i": g, "box": None, "steps": [
            S("client_gone", "клиент ушёл", float(np.mean(gone[g:ctx.c(g + 4) + 1]))),
            V("counting", "кассир пересчитывает наличные", [(ctx.c(g + 1), None, "full"), (ctx.c(g + 3), None, "full")],
              f"Top-down CCTV frames of a bank cash room. OWLv2 counts {np_} person(s) in the room; the client has "
              "already left. Is the cashier counting or sorting banknotes in their hands or on the counting machine?"),
            R(norm, [(ctx.c(max(0, g - 3)), None, "full"), (ctx.c(g + 2), None, "full")])]})
    return out


def chains_door(ctx, norm):
    door = ctx.rois.get("door")
    work = ctx.obs["moments"].get("work", [])[:3] or ctx.obs["moments"].get("desk", [])[:3]
    if not work:
        return []
    box = door["box"] if door else None
    seq = [(ctx.c(i), box, "crop") for i in work]
    pres = float(np.mean([ctx.st.present[ctx.c(i)] >= 0.5 for i in work]))
    return [{"type": "service_door_left_open", "i": work[0], "box": box, "steps": [
        V("open", "дверь кассового помещения открыта", seq, INTRO + f"They show the door of the room at "
          f"{len(seq)} different moments{_where(ctx, None, 'door')}. Is the door standing open in these images "
          "(a gap or the open doorway is visible), rather than closed?"),
        S("someone_works", "в помещении работают", pres),
        R(norm, [(ctx.c(work[0]), box, "mark")])]}]


def chains_bag(ctx, norm):
    st = ctx.st
    left = (recent_max(st.q["collector_present"], st.k(30)) >= 0.5) & (st.q["collector_present"] < 0.3)
    alone = left & (st.away_s >= 1)
    out = []
    for obj in ("bag", "handbag", "backpack"):
        for tr in ctx.scene.tracks.get(obj, [])[:2]:
            ks = [p[0] for p in tr["path"] if alone[p[0]]]
            if not ks:
                continue
            seq = [(k, ctx.scene.box_at(tr, k), "crop") for k in (ks[0], ks[-1])]
            out.append({"type": "collection_bag_unsecured", "i": ks[0], "box": seq[0][1], "steps": [
                V("is_bag", "предмет — инкассаторская сумка", seq, INTRO + "Is this a cash-collection bag "
                  "(a sealed bag for transporting money)?"),
                S("left_alone", "инкассатор ушёл, кассира нет рядом", len(ks) / max(1, len(tr["path"]))),
                R(norm, [(ks[-1], seq[-1][1], "mark")])]})
    return out[:2]


BUILDERS = {"safe_key_unattended": chains_key, "safe_left_open": chains_safe_open, "cash_unattended": chains_cash,
            "personal_items_cash_area": chains_personal, "counting_without_customer": chains_counting,
            "service_door_left_open": chains_door, "collection_bag_unsecured": chains_bag}


# ---------------------------------------------------------------- Qwen

def _images(spec, frames):
    out = []
    for i, box, how in spec:
        img = frames[i].image
        img = crop(img, box) if how == "crop" else (marked(img, box) if how == "mark" else img)
        out.append(Frame(frames[i].t, img))
    return out


def rule_prompt(norm: dict, obs_text: str) -> str:
    cond = "\n".join(f"{k}. {c}" for k, c in enumerate(norm.get("conditions", []), 1))
    exc = "\n".join(f"- {e}" for e in norm.get("exceptions", [])) or "- none"
    official = norms.official_text(norm)
    text = official[:600] if official else norm["text"]
    return ("Top-down CCTV frames of a bank cash room (a red box marks the object in question, if any).\n"
            f"Observations from the object detector (OWLv2) and the scene analysis:\n{obs_text}\n\n"
            f"Rule ({norm['document_name']}, {norm['clause']}): {text}\n"
            f"The rule applies only if ALL of these are visible:\n{cond}\n"
            f"It does NOT apply in these normal situations:\n{exc}\n\n"
            "Looking at the frames and the observations, are all the conditions met, with none of the normal "
            "situations?")


def ask(chains, frames, vlm, obs_text, batch: int = 4):
    from .vlm import SYSTEM
    jobs = []
    for ch in chains:
        for s in ch["steps"]:
            if s["kind"] == "state":
                continue
            q = rule_prompt(norms.for_type(ch["type"]) or {}, obs_text) if s["kind"] == "rule" else s["question"]
            imgs = _images(s["images"], frames)
            jobs.append((s, imgs, q))
    prompts = []
    for _, imgs, q in jobs:
        prompts.append((SYSTEM, imgs, q + "\nA. Yes\nB. No\nAnswer with the letter only."))
        prompts.append((SYSTEM, imgs, q + "\nA. No\nB. Yes\nAnswer with the letter only."))
    probs = []
    for k in range(0, len(prompts), batch):
        probs += vlm.choice_probs(prompts[k:k + batch], 2)
    for k, (s, _, _) in enumerate(jobs):
        s["p"] = round((float(probs[2 * k][0]) + float(probs[2 * k + 1][1])) / 2, 4)
    return len(jobs)


def run(st: State, scene: Scene, obs: dict, frames, vlm) -> dict:
    n = min(st.n, len(frames))
    if n == 0:
        return {"version": VERSION, "retrieved": [], "chains": [], "features": {}}
    ctx = Ctx(st, scene, obs, n)
    words = " ".join(e["en"] for e in obs["events"])
    retrieved = norms.retrieve(obs["tags"], words, k=50)          # every rule whose tags match the observations
    to_check = [t for t in CHAIN_TYPES if any(r.get("applies_to") == t for r in retrieved)]
    chains = []
    for tid in to_check:
        norm = norms.for_type(tid)
        if tid == "unlocked_bank_terminal":
            chains += chains_terminal(ctx, norm, frames)
        elif tid in BUILDERS:
            chains += BUILDERS[tid](ctx, norm)
    nq = ask(chains, frames, vlm, obs["text"]) if chains else 0
    feats = {}
    for ch in chains:
        ps = [s["p"] for s in ch["steps"]]
        ch["score"] = round(float(min(ps)), 4) if ps else 0.0
        ch["t"] = float(st.t[ch["i"]])
        f = f"ch_{ch['type']}"
        if f not in feats or ch["score"] > feats[f]["score"]:
            feats[f] = {"type": ch["type"], "score": ch["score"], "t": ch["t"], "box": ch["box"],
                        "norm": (norms.for_type(ch["type"]) or {}).get("id"),
                        "steps": [{"name": s["name"], "ru": s["ru"], "p": s["p"]} for s in ch["steps"]]}
    for ch in chains:                               # images are rebuilt from the frames when needed
        for s in ch["steps"]:
            s.pop("question", None)
    return {"version": VERSION, "retrieved": [{"id": r["id"], "applies_to": r.get("applies_to"), "score": r["score"]}
                                              for r in retrieved],
            "checked": to_check, "n_questions": nq, "chains": chains, "features": feats}
