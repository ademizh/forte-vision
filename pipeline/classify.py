"""Local normal-vs-abnormal flow analysis over overlapping video fragments.

The risk score is 1-P(normal), including visually unresolved cases as candidates
for human review. It is an answer-option score, not a calibrated probability of
an actual violation. All thresholds must be fitted on known videos before use.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

VERSION = "classify-flow-v2"
N_FRAMES = int(os.environ.get("CLS_FRAMES", "8"))
if not 3 <= N_FRAMES <= 26:
    raise ValueError("CLS_FRAMES must be between 3 and 26")
BATCH = int(os.environ.get("CLS_BATCH", "1"))           # each prompt holds N_FRAMES images: keep it small
CATALOG = Path(os.environ.get("CATALOG", Path(__file__).parent.parent / "catalog" / "violations.yaml"))
TABLE_FACTS = [("cashier_at_desk", "cashier"), ("client_at_desk", "client"), ("collector_present", "collector"),
               ("outsider_behind_counter", "outsider"), ("cash_on_desk", "cash_desk"),
               ("cash_in_hands", "cash_hands"), ("collection_bag_visible", "bag")]
_CAT = None


def catalog() -> dict:
    global _CAT
    if _CAT is None:
        _CAT = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    return _CAT


def keyframes(st, obs, n_frames: int, k: int = N_FRAMES) -> list[int]:
    """Frame indices: the moments the observations point at, then evenly spread to fill k."""
    n = min(st.n, n_frames)
    if n == 0:
        return []
    m = obs.get("moments", {})
    cand = [i for key in ("departure", "client_left", "work", "desk") for i in m.get(key, [])]
    cand += [e["i"] for e in obs.get("events", []) if "i" in e]
    picked = []
    for i in cand:
        if 0 <= i < n and all(abs(st.t[i] - st.t[j]) >= 2.0 for j in picked):
            picked.append(int(i))
        if len(picked) >= k - 2:
            break
    for i in np.linspace(0, n - 1, num=k).round().astype(int):
        if len(picked) >= k:
            break
        if all(abs(st.t[i] - st.t[j]) >= 2.0 for j in picked):
            picked.append(int(i))
    return sorted(picked)[:k]


def table(st, scene=None, step: float = 2.0) -> str:
    """Compact state table every `step` seconds (Qwen per-frame yes-probabilities, OWLv2 people count)."""
    rows, last = [], -1e9
    head = "t(s) | " + " | ".join(short for _, short in TABLE_FACTS) + " | people(OWLv2)"
    for i in range(st.n):
        if st.t[i] - last < step - 1e-6:
            continue
        last = st.t[i]
        vals = " | ".join(f"{st.q[k][i]:.1f}" for k, _ in TABLE_FACTS)
        ppl = f"{scene.persons[i]:.0f}" if scene is not None and getattr(scene, "n", 0) == st.n else "-"
        rows.append(f"{st.t[i]:.0f} | {vals} | {ppl}")
    return head + "\n" + "\n".join(rows)


def context(st, scene, obs) -> str:
    cat = catalog()
    normal = str(cat.get("normal_routine", "")).strip()
    viol = "\n".join(f"- {v['type']}: {str(v.get('definition', '')).strip()}" for v in cat["violations"])
    return (
        "These are frames from a silent top-down CCTV video of a bank cash room, in time order.\n\n"
        f"NORMAL ROUTINE (this is NOT a violation):\n{normal[:2500]}\n\n"
        f"VIOLATIONS the bank must flag:\n{viol}\n\n"
        "WHAT THE DETECTORS SAW IN THE WHOLE VIDEO (may be incomplete; small objects are often missed):\n"
        f"{obs.get('text', '')}\n\n"
        "State over time (0 = no, 1 = yes; probabilities from a per-frame model):\n"
        f"{table(st, scene)}\n\n")


def _two_orders(vlm, system, imgs, ctx, question, options):
    """Probability of each option, averaged over the given order and the reversed order."""
    from .vlm import LETTERS
    n = len(options)

    def prompt(order):
        lines = "\n".join(f"{LETTERS[j]}. {options[o]}" for j, o in enumerate(order))
        return (system, imgs, ctx + question + "\n" + lines + "\nAnswer with the letter only.")
    fwd, rev = list(range(n)), list(range(n))[::-1]
    prompts = [prompt(fwd), prompt(rev)]
    probs = []
    for s in range(0, len(prompts), BATCH):
        probs += vlm.choice_probs(prompts[s:s + BATCH], n)
    p = np.zeros(n)
    for order, pr in zip((fwd, rev), probs):
        for j, o in enumerate(order):
            p[o] += float(pr[j]) / 2
    return p


def flow_windows(st, n_frames, width=12.0, stride=6.0, budget=N_FRAMES):
    """Cover the whole timeline; retain before/during/after context in each clip.

    At most `budget` images per question. Sampled visual changes augment uniform
    key frames. No hard cutoff silently discards the end of a long video.
    """
    n = min(st.n, n_frames)
    if not n:
        return []
    out, start = [], float(st.t[0])
    while True:
        idx = np.flatnonzero((st.t[:n] >= start) & (st.t[:n] <= start + width))
        if len(idx):
            uniform = idx[np.linspace(0, len(idx)-1, min(len(idx), max(2,budget-2))).round().astype(int)]
            movement = np.zeros(len(idx))
            for key in ("cashier_at_desk", "client_at_desk", "cash_in_hands", "safe_open", "service_door_open"):
                movement += np.abs(np.diff(st.q[key][idx], prepend=st.q[key][idx[0]]))
            picks = set(int(i) for i in uniform)
            for j in np.argsort(-movement, kind="stable"):
                if len(picks) >= budget:
                    break
                picks.add(int(idx[j]))
            out.append(sorted(picks))
        if start + width >= st.t[n-1]:
            break
        start += stride
    return out


def run(st, scene, obs, frames, vlm) -> dict:
    """Contrast normal and irregular flows locally, rather than eight images per video."""
    from .vlm import SYSTEM
    windows = flow_windows(st, len(frames))
    if not windows:
        raise ValueError("No frames for whole-flow analysis")
    cat = catalog()
    normal = str(cat.get("normal_routine", "")).strip()
    definitions = "\n".join(f"- {v['type']}: {v.get('definition','')}" for v in cat['violations'])
    records = []
    for idx in windows:
        imgs = [frames[i] for i in idx]
        a, b = imgs[0].t, imgs[-1].t
        nearby = [e.get('en','') for e in obs.get('events',[]) if a-3 <= e.get('t',0) <= b+3]
        history = [e.get('en','') for e in obs.get('events',[]) if e.get('t',0) < a-3][-12:]
        ctx = ("Silent CCTV frames in chronological order. Compare the visible sequence, including its context.\n"
               f"NORMAL WORKFLOW (alternatives, not one mandatory order):\n{normal}\n\n"
               f"Examples of visible deviations (not an exhaustive list):\n{definitions}\n\n"
               "Normal alternatives: an employee working alone may sort/count cash; going to the safe, "
               "opening it while attending it, a legitimate collector visit, temporarily opening a door "
               "for passage, and switching off equipment during closing are normal. A phone or counting "
               "device by itself does not establish a violation.\n"
               "Do not assume an action was omitted because it is outside this clip or obscured. "
               "Do not infer amounts, bank-system entries, identity, or an unlocked screen from an illuminated screen. "
               "A short visit alone does not prove that a check was omitted.\n"
               "Earlier timeline (uncertain detector hints; use only for context, not as proof of a skipped step):\n"
               + "\n".join(history) + "\n"
               + "Nearby observations (also uncertain):\n" + "\n".join(nearby) + "\n")
        p = _two_orders(vlm, SYSTEM, imgs, ctx,
                        "Which explanation best matches the VISIBLE sequence?",
                        ["Normal work, including one of the normal alternatives",
                         "A visible action or persistent state suggests a safety/procedure deviation",
                         "The relevant action/state is unclear or not visually verifiable"])
        # Uncertainty is recorded separately; it is not silently treated as a normal video.
        records.append({'start':float(a),'end':float(b),'indices':idx,
                        'score':float(1.0-p[0]), 'p_deviation':float(p[1]),
                        'p_normal':float(p[0]), 'p_unclear':float(p[2])})
    best = max(records, key=lambda x:x['score'])
    idx, score = best['indices'], best['score']
    # The frame is a candidate for review, not a proven event localization.
    imgs = [frames[i] for i in idx]
    t = float(imgs[len(imgs)//2].t)
    names = [v['type'] for v in cat['violations']] + ['The specific type is not visually established']
    ids = [v['id'] for v in cat['violations']] + ['other']
    pt = _two_orders(vlm, SYSTEM, imgs,
                     "Use only visible actions in these chronological frames.\n",
                     "If this sequence contains a deviation, which type is supported by the visible evidence?", names)
    k = int(np.argmax(pt))
    pf = _two_orders(vlm, SYSTEM, imgs, '',
                     'Which frame is the most useful for a human reviewing this possible deviation?',
                     [f'Frame {j} at {f.t:.1f} seconds' for j,f in enumerate(imgs)])
    t = float(imgs[int(np.argmax(pf))].t)
    feat = {'type':ids[k], 'score':score, 't':t, 'start':best['start'], 'end':best['end'],
            'generic':True, 'classification_score':float(pt[k]),
            'type_supported': bool(ids[k] != 'other' and pt[k] >= 0.65 and
                                   best['p_deviation'] >= 0.5 and best['p_unclear'] <= 0.25),
            'why':f"Фрагмент {best['start']:.1f}–{best['end']:.1f} с требует проверки на отклонение от нормального порядка."}
    return {'version':VERSION,'features':{'v_judge':feat},'p_violation':score,
            'keyframes':[float(f.t) for f in imgs], 'p_types':dict(zip(ids,map(float,pt))),
            'p_frames':list(map(float,pf)), 'windows':records, 'n_questions':2*len(records)+4}
