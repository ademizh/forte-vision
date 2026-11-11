"""
Video-level and event-level metrics for the verified engine (no GPU; works on the JSONL records).

    m = summarize(rows, records, truth, tol=3.0)
rows: [{"video", "truth", "events"}] (calibration.loo or frozen thresholds)
Video level : accuracy, precision, recall, F1, F2, false-positive rate, false negatives, confusion matrix
Event level : per-violation recall / precision (right type predicted), candidate-generation recall (a candidate
              of the true type existed before thresholds), temporal error |predicted start - labelled time|
              for right-type events (labels.csv has one time per violation)
"""
from __future__ import annotations

import numpy as np

from .calibration import same_type


def _div(a, b):
    return a / b if b else 0.0


def summarize(rows, records, truth, tol: float = 3.0) -> dict:
    tp = fp = fn = tn = 0
    per = {}
    terr, loc = [], 0
    gen_hit, gen_n = 0, 0
    for r in rows:
        v, tr, ev = r["video"], r["truth"], r["events"]
        if tr["label"] == "violation":
            tp += bool(ev)
            fn += not ev
            for t in tr["types"]:
                d = per.setdefault(t, {"pos": 0, "hit": 0, "pred": 0, "pred_ok": 0})
                d["pos"] += 1
                d["hit"] += any(same_type(e["type"], [t]) for e in ev)
            gen_n += 1
            gen_hit += any(same_type(c["type"], tr["types"]) for c in records[v].get("candidates", [])
                           if not c.get("internal"))
            right = [e for e in ev if same_type(e["type"], tr["types"])]
            if right and tr["times"]:
                err = min(abs(e["start"] - t) for e in right for t in tr["times"])
                terr.append(err)
                loc += err <= tol
        else:
            fp += bool(ev)
            tn += not ev
        for e in ev:
            d = per.setdefault(e["type"], {"pos": 0, "hit": 0, "pred": 0, "pred_ok": 0})
            d["pred"] += 1
            d["pred_ok"] += tr["label"] == "violation" and same_type(e["type"], tr["types"])
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = _div(2 * prec * rec, prec + rec)
    f2 = _div(5 * prec * rec, 4 * prec + rec)
    runtimes = [records[r["video"]]["timing"].get("total", 0) for r in rows if r["video"] in records]
    peaks = [records[r["video"]].get("gpu", {}).get("peak_gb") for r in rows if r["video"] in records]
    peaks = [p for p in peaks if p]
    return {
        "videos": len(rows), "accuracy": _div(tp + tn, len(rows)), "precision": prec, "recall": rec, "f1": f1,
        "f2": f2, "fpr": _div(fp, fp + tn), "false_negatives": fn,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "per_violation": {t: {"recall": _div(d["hit"], d["pos"]), "precision": _div(d["pred_ok"], d["pred"]), **d}
                          for t, d in sorted(per.items())},
        "candidate_recall": _div(gen_hit, gen_n),
        "temporal_error_median_sec": float(np.median(terr)) if terr else None, f"within_{tol:.0f}s": loc,
        "runtime_sec": {"mean": float(np.mean(runtimes)) if runtimes else None,
                        "max": float(np.max(runtimes)) if runtimes else None},
        "gpu_peak_gb": max(peaks) if peaks else None,
    }


def line(m: dict) -> str:
    c = m["confusion"]
    return (f"videos {m['videos']} | accuracy {m['accuracy']:.0%} | precision {m['precision']:.2f} | "
            f"recall {m['recall']:.2f} ({c['tp']}/{c['tp'] + c['fn']}) | F1 {m['f1']:.2f} | F2 {m['f2']:.2f} | "
            f"FPR {m['fpr']:.2f} ({c['fp']}/{c['fp'] + c['tn']})")
