"""Decision from frozen, versioned calibration.

Default CLI policy: recall_policy.py. Strong physical events can add evidence;
no source vetoes another. The global normal-vs-abnormal score gets the highest
threshold with zero misses on the calibration videos not already covered by
strong events. This is an empirical objective, not an unseen-video guarantee.
Legacy strict/headroom/support policies are available for comparisons only.
"""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

import numpy as np

from . import norms
from . import state_engine as se

JUDGE_PATH = Path(os.environ.get("JUDGE_THRESHOLDS", Path(__file__).parent.parent / "catalog" / "judge_thresholds.json"))

ENGINE_TYPES = [t for t in se.TYPES if t not in (
    "cash_collection_without_checks", "collection_bag_not_inspected", "unlocked_bank_terminal",
    "safe_left_open", "safe_key_unattended")]
FLOOR = {"engine": 0.10, "chain": 0.20, "change": 0.15, "vlm": 0.30, "focus": 0.15, "door": 0.15}
CAP = {"engine": 10.0, "chain": 0.98, "change": 0.98, "vlm": 0.995, "focus": 0.995, "door": 2.5}
MARGIN = {"engine": 0.05, "chain": 0.05, "change": 0.05, "vlm": 0.03, "focus": 0.05, "door": 0.08}
SOURCES = ("engine", "chain", "change", "vlm", "focus", "door")
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}       # identical files: scored once


def same_type(a: str, types) -> bool:
    if a in types:
        return True
    return a in se.COLLECTOR_TYPES and bool(set(se.COLLECTOR_TYPES) & set(types))


def source(feat: str) -> str:
    if feat.startswith("e_"):
        return "engine"
    return {"c_": "change", "v_": "vlm", "f_": "focus", "d_": "door"}.get(feat[:2], "chain")


def engine_features(facts: dict) -> dict:
    st, scores = se.type_scores(facts)
    out = {}
    for tid in ENGINE_TYPES:
        sc = scores[tid][0]
        if not len(sc):
            continue
        i = int(np.argmax(sc))
        out[f"e_{tid}"] = {"type": tid, "score": float(sc[i]), "t": float(st.t[i])}
    return out


def video_features(facts: dict, chain_rec: dict | None, change_rec: dict | None = None,
                   vlm_rec: dict | None = None, focus_rec: dict | None = None, door_rec: dict | None = None) -> dict:
    f = engine_features(facts)
    for rec in (chain_rec, change_rec, vlm_rec, focus_rec, door_rec):
        if rec:
            f.update(rec.get("features", {}))
    return f


# ---------------------------------------------------------------- calibration

def inference_signature(include_focus=True):
    """Settings that change the scale of visual scores must match calibration."""
    from .classify import VERSION as cv, N_FRAMES
    from .vlm import MAX_PIXELS, MODEL_PATH, QUANT
    from . import detect
    def config_hash(folder):
        p = Path(folder) / "config.json"
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    sig = {"state_version": se.VERSION, "classifier_version": cv,
            "fact_fps": float(os.environ.get("FACT_FPS", "1")),
            "fact_width": int(os.environ.get("FACT_WIDTH", "1280")),
            "tile_every": int(os.environ.get("TILE_EVERY", "2")),
            "vlm_max_pixels": MAX_PIXELS, "vlm_quant": QUANT,
            "cls_frames": N_FRAMES, "detectors": list(detect.wanted()),
            "vlm_config": config_hash(MODEL_PATH),
            "detector_configs": {d: config_hash(detect.PATHS[d]) for d in detect.wanted() if d in detect.PATHS}}
    if include_focus:
        from .focus import signature
        sig["focus"] = signature()
    return sig

def _thr(feats, clean, f, src, excl, margin_scale=1.0, allow_fa=0):
    """Highest clean value + margin. allow_fa=k (chains only): just above the (k+1)-th highest clean value."""
    vals = sorted((feats[v].get(f, {}).get("score", 0.0) for v in clean if v not in excl), reverse=True)
    top = vals[0] if vals else 0.0
    if allow_fa and src != "engine":
        base = vals[allow_fa] if len(vals) > allow_fa else 0.0
        return round(max(base + 0.01, FLOOR[src]), 4), top
    return round(max(top + MARGIN[src] * margin_scale, FLOOR[src]), 4), top


def calibrate(feats: dict, truth: dict, exclude: set = frozenset(), rule: str = "strict",
              sources=SOURCES, margin_scale: float = 1.0, allow_fa: int = 0) -> dict:
    if rule == "recall":
        from .recall_policy import fit
        return fit(feats, truth, source, exclude, sources)
    names = sorted({k for v, fv in feats.items() if v in truth and v not in exclude for k in fv})
    clean = [v for v in feats if v in truth and truth[v]["label"] == "no_violation" and v not in exclude]
    viol = [v for v in feats if v in truth and truth[v]["label"] == "violation" and v not in exclude]
    out = {}
    for f in names:
        src = source(f)
        ftype = next((fv[f]["type"] for fv in feats.values() if f in fv), "")
        generic = any(fv[f].get("generic") for fv in feats.values() if f in fv)
        thr, cmax = _thr(feats, clean, f, src, set(), margin_scale, allow_fa)
        on = src in sources and thr <= CAP[src]
        if src == "engine" and ftype not in ENGINE_TYPES:
            on = False
        hits = [v for v in viol if feats[v].get(f, {}).get("score", 0.0) >= thr and
                (generic or same_type(ftype, truth[v]["types"]))]
        fires = []
        if src != "engine" and rule in ("support", "strict") and not hits:
            on = False
        if src != "engine" and rule == "strict" and on:
            fires = [c for c in clean if feats[c].get(f, {}).get("score", 0.0) >=
                     _thr(feats, clean, f, src, {c}, margin_scale, allow_fa)[0]]
            on = (len(hits) > len(fires)) if allow_fa else not fires
        out[f] = {"type": ftype, "threshold": thr, "enabled": bool(on), "source": src, "generic": generic,
                  "clean_max": round(cmax, 4), "own_hits": hits, "clean_firing_loo": fires}
    return out


def save(cal: dict, meta: dict, path: Path = JUDGE_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    sig = inference_signature()
    tmp.write_text(json.dumps({**meta, "state_version": se.VERSION,
                              "classifier_version": sig["classifier_version"],
                              "inference_signature": sig, "features": cal},
                             ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def load(path: Path = JUDGE_PATH) -> dict:
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("requires_calibration"):
        raise RuntimeError("Updated pipeline requires calibration on YOUR labelled videos: "
                           "python forte.py predict --extract-only; python forte.py accuracy --write")
    if data.get("state_version") != se.VERSION:
        raise RuntimeError("Thresholds belong to an older state engine. Run python forte.py accuracy --write")
    from .classify import VERSION as classifier_version
    if data.get("classifier_version") != classifier_version:
        raise RuntimeError("Whole-flow classifier changed. Re-extract training scores and recalibrate before inference")
    if data.get("inference_signature") != inference_signature():
        raise RuntimeError("Visual inference settings differ from calibration. Restore the frozen settings or recalibrate on known videos")
    return data.get("features", {})


def chains_needed(cal: dict) -> bool:
    """The service runs the condition chains only if calibration switched at least one of them on."""
    return needed(cal, "chain")


def needed(cal: dict, src: str) -> bool:
    return any(c["enabled"] and c["source"] == src for c in cal.values())


# ---------------------------------------------------------------- decision

def _chain_text(x: dict) -> str:
    steps = "; ".join(f"{s['ru']} — {'да' if s['p'] >= 0.5 else 'не подтверждено'} ({s['p']:.2f})"
                      for s in x.get("steps", []))
    return f"Проверка условий нормы: {steps}."


def decide(feats: dict, facts: dict | None, cal: dict) -> list[dict]:
    from .recall_policy import META
    if META in cal:
        if not facts or not facts.get("t"):
            raise ValueError("Analysis incomplete: no visual facts; cannot return no_violation")
        if needed(cal, "vlm") and "v_judge" not in feats:
            raise ValueError("Analysis incomplete: whole-flow classifier did not run")
        if needed(cal, "focus") and "f_any" not in feats:
            raise ValueError("Analysis incomplete: focused visual-state branch did not run")
        required=set(cal[META].get("required_fact_keys",[]))
        if required-set(facts.get("p",{})):
            raise ValueError("Analysis incomplete: missing calibrated visual signals: "+
                             ", ".join(sorted(required-set(facts.get("p",{})))))
        se.State(facts)
    events=[]
    for f,feature in feats.items():
        c=cal.get(f)
        if not c or not c.get("enabled"):continue
        observations=feature.get("episodes",[feature]) if c["source"]=="focus" else [feature]
        for x in observations:
            if x["score"]<=0 or x["score"]<c["threshold"]:continue
            ev={"id":x["type"],"feature":f,"source":c["source"],"score":float(x["score"]),
                "threshold":c["threshold"],"margin":float(x["score"]-c["threshold"]),
                "rank":float((x["score"]-c["threshold"])/max(1e-9,1-c["threshold"])),
                "timecode_sec":float(x["t"]),"start":float(x.get("start",x["t"])),
                "end":float(x.get("end",x["t"])),"box":x.get("box"),
                "type_supported":bool(x.get("type_supported")),"persistent":bool(x.get("persistent"))}
            for key in ("evidence_times","box_timecode","candidate_type","support_count"):
                if key in x:ev[key]=x[key]
            if c["source"]=="engine" and facts is not None:
                r=se.evaluate(facts,{x["type"]:c["threshold"]},types=[x["type"]])
                if r["events"]:
                    e=max(r["events"],key=lambda e:e["score"])
                    ev.update({"id":e["id"],"timecode_sec":e["timecode_sec"],"start":e["start"],
                               "end":e["end"],"why":e.get("why","")})
            if "why" not in ev:
                ev["why"]=x.get("why") or (f"Около {ev['timecode_sec']:.1f} с. "+_chain_text(x))
            if ((c.get("generic") or c["source"]=="focus") and not x.get("type_supported")) or x["score"]<=c.get("clean_max",0.):
                ev["candidate_type"]=x.get("candidate_type",ev["id"])
                ev["id"]="other"
                ev["type_supported"]=False
                ev["why"]+=" Конкретный тип и применимость нормативного требования требуют проверки."
            norm=norms.for_type(ev["id"])
            if norm:
                ev["norm"]={"id":norm["id"],"document":norm["document_name"],"clause":norm["clause"],
                            "text":norms.official_text(norm) or norm["text"],"text_status":norm.get("text_status")}
            events.append(ev)
    def overlaps(a,b):
        return max(a["start"],b["start"])<=min(a["end"],b["end"])+1.5
    # Prefer localized, persistent type evidence when duplicate sources describe
    # the same episode. Two separated incidents of one type remain two events.
    ordered=sorted(events,key=lambda e:(e["id"]!="other",e["type_supported"],
                                        e["persistent"],e["source"]=="focus",e["rank"]),reverse=True)
    kept=[]
    for e in ordered:
        if any(e["id"]==p["id"] and overlaps(e,p) for p in kept):continue
        if e["id"]=="other" and any(e.get("candidate_type")==p["id"] and overlaps(e,p) for p in kept):continue
        kept.append(e)
    # A collector visit is context, not a veto of a distinct outsider/bag event.
    return sorted(kept,key=lambda e:(e["id"]!="other",e["rank"]),reverse=True)


def verdict(events: list[dict]) -> dict:
    if not events:
        return {"verdict": "no_violation", "type": "", "timecode_sec": None, "score": 0.0}
    e = events[0]
    return {"verdict": "violation", "type": e["id"], "timecode_sec": round(e["timecode_sec"], 1),
            "score": e["score"], "threshold": e["threshold"], "feature": e["feature"]}
