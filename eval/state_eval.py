"""
Evaluate / calibrate the state engine (pipeline/state_engine.py) on the cached facts. No GPU, seconds.

  python eval/state_eval.py                    # honest leave-one-video-out result + per-video lines
  python eval/state_eval.py --types            # per-type table: fires on its own videos? on clean videos?
  python eval/state_eval.py --ablation         # Qwen only / detectors only / both
  python eval/state_eval.py --margins          # how the calibration margin trades recall vs false alarms
  python eval/state_eval.py --calibrate        # write catalog/state_thresholds.json from ALL clean videos
  python eval/state_eval.py --verify eval/verify   # include cached verifier answers (scripts/run_vlm.py verify-dataset)

Honest protocol
  * Thresholds are learned from CLEAN videos only: tau(type) = highest score of that type in any clean video
    + margin. In leave-one-out, the tested video is removed from calibration (with its duplicate).
  * X-47 is the same file as X-4 (labels 57 s and 68 s -> one video with two labels), X-48 the same as X-5.
    Duplicates are scored once, otherwise they count twice and leak between train and test.
  * A detection counts as LOCALIZED only if an event is within --tol seconds of a label (criterion 2).
    "Any event" recall is printed too, for comparison with the old scripts.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.state_engine import (AUTO_TYPES, COLLECTOR_TYPES, EVIDENCE, P, THRESHOLDS_PATH,  # noqa: E402
                                   evaluate, type_scores)
from pipeline.verify import VERIFY_TYPES, type_max  # noqa: E402
from pipeline.verify import decide as verify_decide  # noqa: E402

DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}     # copy -> original (check_data.py section 6, cosine 1.0000)
FLOOR = 0.10                                    # a threshold is never below this (noise floor)


def load(a):
    labels = {r["video"]: r for r in csv.DictReader(open(a.labels, encoding="utf-8"))}
    facts = {}
    for v in labels:
        p = Path(a.facts) / f"{v}.json"
        if p.exists():
            facts[v] = json.loads(p.read_text(encoding="utf-8"))
    cat = yaml.safe_load(open(a.catalog, encoding="utf-8"))
    expected = {}
    for t in cat["violations"]:
        for vid in t.get("videos", []):
            expected.setdefault(vid, set()).add(t["id"])
    # merge duplicates: one video, all labels
    truth = {}
    for v, r in labels.items():
        if v not in facts:
            continue
        key = DUPLICATES.get(v, v)
        e = truth.setdefault(key, {"label": "no_violation", "times": [], "types": set()})
        if r["label"] == "violation":
            e["label"] = "violation"
            if r["timecode_sec"].strip():
                e["times"].append(float(r["timecode_sec"]))
            e["types"] |= expected.get(v, set())
    return truth, facts, cat


def max_scores(facts, use_qwen=True, use_det=True):
    _, sc = type_scores(facts, P, use_qwen, use_det)
    return {k: float(v[0].max()) if len(v[0]) else 0.0 for k, v in sc.items()}


def calibrate(clean_max: dict, exclude: set, margin: float) -> dict:
    th = {}
    for tid in EVIDENCE:
        vals = [m[tid] for v, m in clean_max.items() if v not in exclude]
        th[tid] = round(max(max(vals, default=0.0) + margin, FLOOR), 4)
    return th


VFLOOR = 0.5          # a verifier threshold is never below P(yes) = 0.5


def calibrate_verify(vmax: dict, clean: list, exclude: set, margin: float) -> dict:
    """tau(type) = highest debiased P(yes) of that question in any clean video + margin."""
    th = {}
    for tid in VERIFY_TYPES:
        vals = [vmax[c][tid]["p_yes"] for c in clean if c not in exclude and c in vmax and tid in vmax[c]]
        th[tid] = round(min(max(max(vals, default=0.0) + margin, VFLOOR), 0.99), 4)
    return th


def run(truth, facts, margin, use_qwen=True, use_det=True, enabled=None, verify=None, loo=True,
        vmargin=0.02, use_engine=True):
    clean = [v for v in truth if truth[v]["label"] == "no_violation"]
    cmax = {v: max_scores(facts[v], use_qwen, use_det) for v in clean}
    full_th = calibrate(cmax, set(), margin)
    vmax = {v: type_max(r) for v, r in (verify or {}).items()}
    rows = []
    for v, tr in truth.items():
        th = calibrate(cmax, {v}, margin) if loo else full_th
        res = evaluate(facts[v], th, use_qwen=use_qwen, use_det=use_det, types=enabled)
        evs = res["events"] if use_engine else []
        if verify is not None and v in verify:            # verifier can ADD events, never remove
            vth = calibrate_verify(vmax, clean, {v} if loo else set(), vmargin)
            have = {(e["id"], round(e["timecode_sec"])) for e in evs}
            evs = evs + [e for e in verify_decide(verify[v], vth) if (e["id"], round(e["timecode_sec"])) not in have]
        rows.append({"video": v, "truth": tr, "events": sorted(evs, key=lambda e: e["timecode_sec"]),
                     "cands": res["candidates"]})
    return rows, full_th


def metrics(rows, tol):
    tp = fn = fp = tn = loc = typ = 0
    for r in rows:
        tr, ev = r["truth"], r["events"]
        if tr["label"] == "violation":
            if ev:
                tp += 1
                near = [e for e in ev if any(abs(e["timecode_sec"] - t) <= tol for t in tr["times"])]
                if near:
                    loc += 1
                    if tr["types"] and any(e["id"] in tr["types"] for e in near):
                        typ += 1
            else:
                fn += 1
        else:
            fp += bool(ev)
            tn += not ev
    n = tp + fn + fp + tn
    return {"n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn, "acc": (tp + tn) / n if n else 0, "loc": loc, "typ": typ}


def fmt(m, tol):
    nv, nc = m["tp"] + m["fn"], m["fp"] + m["tn"]
    return (f"videos {m['n']} | accuracy {m['acc']:.0%} | recall {m['tp']}/{nv} | false alarms {m['fp']}/{nc} | "
            f"localized (+-{tol:.0f}s) {m['loc']}/{nv} | right type at the label {m['typ']}/{nv}")


def per_type(truth, facts, margin, tol):
    clean = [v for v in truth if truth[v]["label"] == "no_violation"]
    print(f"{'type':<32} {'own videos: score / tau (LOO)':<44} {'clean videos firing (LOO)':<28}")
    for tid in EVIDENCE:
        if tid == "cash_collection_without_checks":
            continue
        cmax = {v: max_scores(facts[v])[tid] for v in clean}
        own = [v for v in truth if tid in truth[v]["types"] or
               (tid in COLLECTOR_TYPES and set(COLLECTOR_TYPES) & truth[v]["types"])]
        cells = []
        for v in own:
            tau = max(max((cmax[c] for c in clean), default=0) + margin, FLOOR)
            res = evaluate(facts[v], {tid: tau}, types=[tid] if tid not in COLLECTOR_TYPES else list(COLLECTOR_TYPES))
            hit = any(any(abs(e["timecode_sec"] - t) <= tol for t in truth[v]["times"]) for e in res["events"])
            s = max_scores(facts[v])[tid]
            cells.append(f"{v} {s:.2f}/{tau:.2f}{' HIT' if hit else (' far' if res['events'] else '')}")
        fires = []
        for c in clean:
            others = [cmax[x] for x in clean if x != c]
            if cmax[c] >= max(max(others, default=0) + margin, FLOOR):
                fires.append(f"{c}({cmax[c]:.2f})")
        top = sorted(cmax.values(), reverse=True)[:3]
        print(f"{tid:<32} {', '.join(cells) or '-':<44} {', '.join(fires) or 'none':<16} clean top {top[0]:.2f} {top[1]:.2f} {top[2]:.2f}")


def load_verify(path):
    out = {}
    for p in sorted(Path(path).glob("*.json")):
        if p.stem in DUPLICATES:
            continue
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    if not out:
        sys.exit(f"no verifier results in {path} (run: python scripts/run_vlm.py verify-dataset)")
    return out


def verify_table(truth, verify, vmargin):
    clean = [v for v in truth if truth[v]["label"] == "no_violation"]
    vmax = {v: type_max(r) for v, r in verify.items()}
    print(f"\nVERIFIER per question: own videos P(yes) vs threshold (LOO), clean videos above their LOO threshold")
    for tid in VERIFY_TYPES:
        own = [v for v in truth if tid in truth[v]["types"] and v in vmax]
        cells = []
        for v in own:
            tau = calibrate_verify(vmax, clean, {v}, vmargin)[tid]
            p = vmax[v].get(tid, {}).get("p_yes", 0)
            cells.append(f"{v} {p:.2f}/{tau:.2f}{' HIT' if p >= tau else ''}")
        fires = [c for c in clean if c in vmax and vmax[c].get(tid, {}).get("p_yes", 0) >=
                 calibrate_verify(vmax, clean, {c}, vmargin)[tid]]
        vals = sorted((vmax[c].get(tid, {}).get("p_yes", 0) for c in clean if c in vmax), reverse=True)[:3]
        print(f"  {tid:<28} {', '.join(cells) or '-':<40} clean firing: {', '.join(fires) or 'none'}  "
              f"clean top {' '.join(f'{x:.2f}' for x in vals)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--facts", default="eval/facts")
    ap.add_argument("--catalog", default="catalog/violations.yaml")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--types", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--margins", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--enable", default="", help="comma list of types to use (default: thresholds file / all)")
    ap.add_argument("--verify", default="", help="folder with cached verifier results (eval/verify)")
    ap.add_argument("--vmargin", type=float, default=0.02, help="verifier margin over the clean maximum")
    ap.add_argument("--verify-enable", default="", help="comma list of verifier questions to use (default all)")
    a = ap.parse_args()
    truth, facts, cat = load(a)
    print(f"{len(truth)} unique videos ({sum(t['label'] == 'violation' for t in truth.values())} violation, "
          f"{sum(t['label'] == 'no_violation' for t in truth.values())} clean); duplicates merged: {DUPLICATES}")
    enabled = [x for x in a.enable.split(",") if x] or None
    if a.enable == "all":
        enabled = list(EVIDENCE)
    elif enabled is None:
        enabled = list(AUTO_TYPES)

    verify = load_verify(a.verify) if a.verify else None
    if a.calibrate:
        clean = [v for v in truth if truth[v]["label"] == "no_violation"]
        th = calibrate({v: max_scores(facts[v]) for v in clean}, set(), a.margin)
        en = enabled or list(EVIDENCE)
        out = {"note": "tau = highest score in any clean video + margin (eval/state_eval.py --calibrate)",
               "margin": a.margin, "clean_videos": sorted(clean), "enabled": en, "thresholds": th}
        if verify:
            vmax = {v: type_max(r) for v, r in verify.items()}
            out["verify_margin"] = a.vmargin
            out["verify_thresholds"] = calibrate_verify(vmax, clean, set(), a.vmargin)
            if a.verify_enable:
                out["verify_enabled"] = a.verify_enable.split(",")
            else:                                   # keep a question only if, leave-one-out, it fires on no clean
                ok = []                             # video and confirms at least one of its own videos
                for tid in VERIFY_TYPES:
                    fires = [c for c in clean if c in vmax and vmax[c].get(tid, {}).get("p_yes", 0) >=
                             calibrate_verify(vmax, clean, {c}, a.vmargin)[tid]]
                    hits = [v for v in truth if tid in truth[v]["types"] and v in vmax and
                            vmax[v].get(tid, {}).get("p_yes", 0) >= calibrate_verify(vmax, clean, {v}, a.vmargin)[tid]]
                    print(f"  verifier {tid:<28} own hits {len(hits)}  clean false alarms {len(fires)}"
                          f"  -> {'ON' if hits and not fires else 'off'}")
                    if hits and not fires:
                        ok.append(tid)
                out["verify_enabled"] = ok
        elif THRESHOLDS_PATH.exists():                      # keep verifier thresholds from an earlier run
            old = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
            for k in ("verify_margin", "verify_thresholds", "verify_enabled"):
                if k in old:
                    out[k] = old[k]
        THRESHOLDS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {THRESHOLDS_PATH}")
        for k, v in th.items():
            print(f"  {k:<32} {v:.3f}{'' if k in en else '   (not automatic -> verifier)'}")
        for k, v in out.get("verify_thresholds", {}).items():
            print(f"  verifier {k:<23} {v:.3f}{'' if k in out.get('verify_enabled', []) else '   (disabled)'}")
        return
    if a.types:
        per_type(truth, facts, a.margin, a.tol)
        return
    if a.margins:
        for m in (0.0, 0.02, 0.03, 0.05, 0.08, 0.12):
            print(f"  margin {m:.2f}: " + fmt(metrics(run(truth, facts, m, enabled=enabled)[0], a.tol), a.tol))
        return
    if a.ablation:
        for name, q, d in (("Qwen facts only", True, False), ("DINO+OWL only", False, True), ("Qwen + DINO + OWL", True, True)):
            print(f"  {name:<18} " + fmt(metrics(run(truth, facts, a.margin, q, d, enabled)[0], a.tol), a.tol))
        return
    if verify:
        venable = [x for x in a.verify_enable.split(",") if x] or VERIFY_TYPES
        verify = {v: {**r, "jobs": [j for j in r["jobs"] if j["type"] in venable]} for v, r in verify.items()}
        verify_table(truth, verify, a.vmargin)
        print("\nSOURCES (leave-one-out):")
        print("  state engine only   " + fmt(metrics(run(truth, facts, a.margin, enabled=enabled)[0], a.tol), a.tol))
        print("  verifier only       " + fmt(metrics(run(truth, facts, a.margin, enabled=enabled, verify=verify,
                                                          vmargin=a.vmargin, use_engine=False)[0], a.tol), a.tol))
        print("  engine + verifier   " + fmt(metrics(run(truth, facts, a.margin, enabled=enabled, verify=verify,
                                                          vmargin=a.vmargin)[0], a.tol), a.tol))
        print()
    rows, _ = run(truth, facts, a.margin, enabled=enabled, verify=verify, vmargin=a.vmargin)
    m = metrics(rows, a.tol)
    print("LEAVE-ONE-OUT: " + fmt(m, a.tol) + "   <- report this one")
    for r in rows:
        tr = r["truth"]
        lab = ("VIOL " + "/".join(f"{t:.0f}s" for t in tr["times"]) + " " + "/".join(sorted(tr["types"]))) \
            if tr["label"] == "violation" else "clean"
        ev = "; ".join(f"{e['id']}@{e['timecode_sec']:.0f}s ({e['score']:.2f}>{e['threshold']:.2f})"
                       for e in r["events"]) or "-"
        if tr["label"] == "violation":
            tag = "MISS " if not r["events"] else (
                "OK   " if any(any(abs(e["timecode_sec"] - t) <= a.tol for t in tr["times"]) for e in r["events"])
                else "FAR  ")
        else:
            tag = "FA   " if r["events"] else "ok   "
        cands = ", ".join(f"{c['id']}@{c['timecode_sec']:.0f}s" for c in r["cands"][:3])
        print(f"  {tag}{r['video']:5} {lab:<52} -> {ev}" + (f"   [verifier candidates: {cands}]" if cands and tag in ("MISS ", "FAR  ") else ""))


if __name__ == "__main__":
    main()
