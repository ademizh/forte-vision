"""
Score the fusion method (VLM facts + DINO + OWLv2 + Qwen judge) on the cached judge results. No GPU, seconds.

  python eval/fusion_eval.py                          # default params, per-video lines
  python eval/fusion_eval.py --ablation               # what each source adds (judge / +rules / +detectors)
  python eval/fusion_eval.py --sweep                  # table over th x w_judge
  python eval/fusion_eval.py --cv                     # honest number: tune on half the videos, test on the other half
  python eval/fusion_eval.py --judge eval/judge_noev  # judge run that saw only the frames (--no-evidence)
  python eval/fusion_eval.py --th 0.6 --w-rule 0.4
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.fusion import PARAMS, decide  # noqa: E402

TOL = 2.0


def load(a):
    with open(a.labels, encoding="utf-8") as f:
        labels = {r["video"]: r for r in csv.DictReader(f)}
    recs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(a.judge).glob("*.json"))}
    with open(a.catalog, encoding="utf-8") as f:
        cat = yaml.safe_load(f)
    expected = {}
    for v in cat["violations"]:
        for vid in v.get("videos", []):
            expected.setdefault(vid, set()).add(v["id"])
    return labels, recs, expected


def score(labels, recs, expected, prm, only=None):
    tp = fn = fp = tn = 0
    errs, type_ok, lines, fires = [], [], [], {}
    for vid, lab in labels.items():
        if vid not in recs or (only is not None and vid not in only):
            continue
        ev = decide(recs[vid], prm)
        truth = lab["label"] == "violation"
        if truth and ev:
            tp += 1
            tc = float(lab["timecode_sec"])
            best = min(ev, key=lambda e: abs(e["timecode_sec"] - tc))
            errs.append(abs(best["timecode_sec"] - tc))
            ok = best["id"] in expected[vid] if vid in expected else None
            if ok is not None:
                type_ok.append(ok)
            lines.append(f"  TP    {vid:5} label {tc:4.0f}s  pred {best['timecode_sec']:5.1f}s {best['id']:<30}"
                         f" score {best['score']:.2f}  err {errs[-1]:4.1f}s  type "
                         f"{'OK' if ok else '-' if ok is None else 'WRONG'}  ({len(ev)} event(s))")
        elif truth:
            fn += 1
            exp = "/".join(sorted(expected.get(vid, {"?"})))
            seen = [c for c in recs[vid]["cands"] if abs(c["t"] - float(lab["timecode_sec"])) <= 4]
            why = "no candidate near the label" if not seen else \
                f"candidate at {seen[0]['t']:.0f}s judged normal (P(normal) {seen[0]['probs'][-1]:.2f})"
            lines.append(f"  MISS  {vid:5} label {lab['timecode_sec']:>4}s  expected {exp}  [{why}]")
        elif ev:
            fp += 1
            for e in ev:
                fires.setdefault(e["id"], []).append(vid)
            lines.append(f"  FA    {vid:5} " + "; ".join(f"{e['id']} {e['timecode_sec']:.0f}s ({e['score']:.2f})"
                                                       for e in ev[:3]))
        else:
            tn += 1
    n = tp + fn + fp + tn
    return {"n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn, "acc": (tp + tn) / n if n else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "tc_mean": sum(errs) / len(errs) if errs else None,
            "tc_hit": sum(e <= TOL for e in errs) / len(errs) if errs else None,
            "type_acc": sum(type_ok) / len(type_ok) if type_ok else None, "lines": lines, "fires": fires}


def line(m, label=""):
    tc = f"tc err {m['tc_mean']:.1f}s ({m['tc_hit']:.0%} within {TOL:.0f}s)" if m["tc_mean"] is not None else "tc n/a"
    ty = f"type {m['type_acc']:.0%}" if m["type_acc"] is not None else "type n/a"
    return (f"{label}videos {m['n']} | accuracy {m['acc']:.0%} | recall {m['recall']:.0%} ({m['tp']}/{m['tp'] + m['fn']})"
            f" | false alarms {m['fp']}/{m['fp'] + m['tn']} | {tc} | {ty}")


GRID = {"th": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "w_judge": [0.0, 0.3, 0.6, 1.0], "rule_bonus": [0.0, 0.15, 0.4]}


def grid():
    for th, wj, rb in itertools.product(GRID["th"], GRID["w_judge"], GRID["rule_bonus"]):
        yield {"th": th, "w_judge": wj, "w_rule": 1 - wj, "rule_bonus": rb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--judge", default="eval/judge")
    ap.add_argument("--catalog", default="catalog/violations.yaml")
    for k in ("th", "w_judge", "w_rule", "w_det", "rule_bonus"):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=PARAMS[k])
    ap.add_argument("--holdout", default="", help="report only on these videos: X-3,X-22,...")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--cv", action="store_true")
    a = ap.parse_args()

    labels, recs, expected = load(a)
    if not recs:
        sys.exit(f"No judge results in {a.judge}. Run: python scripts/run_vlm.py judge-dataset")
    prm = {k: getattr(a, k) for k in ("th", "w_judge", "w_rule", "w_det", "rule_bonus")}
    only = set(a.holdout.split(",")) if a.holdout else None
    ev_on = sum(r.get("with_evidence", True) for r in recs.values())
    dets = sorted({d for r in recs.values() for d in r.get("detectors", [])})
    print(f"{len(recs)} videos judged ({ev_on} with evidence text; detectors in facts: {dets or 'none'})")

    if a.ablation:
        print("What each source adds (same cached judge answers, th=%.2f):" % a.th)
        rows = [("rules only (judge ignored)", {"w_judge": 0.0, "w_rule": 1.0, "w_det": 0.0, "rule_bonus": 1.0}),
                ("judge only", {"w_judge": 1.0, "w_rule": 0.0, "w_det": 0.0, "rule_bonus": 0.0}),
                ("judge + rules", {"w_det": 0.0}),
                ("judge + rules + detectors", {})]
        for name, over in rows:
            print(line(score(labels, recs, expected, {**prm, **over}, only), f"  {name:<27} "))
        print("Detectors also feed the judge's evidence text: compare with a --no-evidence judge run:\n"
              "  python eval/fusion_eval.py --judge eval/judge_noev --ablation")
        return
    if a.sweep:
        print(f"{'th':>5} {'w_judge':>7} {'bonus':>5} {'recall':>7} {'FA':>4} {'acc':>5} {'type':>5}")
        res = []
        for g in grid():
            m = score(labels, recs, expected, {**prm, **g}, only)
            res.append((m["acc"], g, m))
        for acc, g, m in sorted(res, key=lambda x: -x[0])[:15]:
            ty = f"{m['type_acc']:.0%}" if m["type_acc"] is not None else "-"
            print(f"{g['th']:5.2f} {g['w_judge']:7.2f} {g['rule_bonus']:5.2f} {m['recall']:7.0%} {m['fp']:4d}"
                  f" {acc:5.0%} {ty:>5}")
        print("(top 15 of the grid; w_rule = 1 - w_judge. These numbers are tuned ON the same videos: use --cv)")
        return
    if a.cv:
        folds = [set(), set()]                      # stratified: violations and clean videos split evenly
        for lab in ("violation", "no_violation"):
            vids = sorted(v for v in labels if v in recs and labels[v]["label"] == lab)
            folds[0] |= set(vids[0::2])
            folds[1] |= set(vids[1::2])
        tot = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
        for i, test in enumerate(folds):
            train = folds[1 - i]
            best = max(grid(), key=lambda g: score(labels, recs, expected, {**prm, **g}, train)["acc"])
            m = score(labels, recs, expected, {**prm, **best}, test)
            print(line(m, f"  fold {i + 1} (params {best}) "))
            for k in tot:
                tot[k] += m[k]
        n = sum(tot.values())
        print(f"CROSS-VALIDATED: accuracy {(tot['tp'] + tot['tn']) / n:.0%} | recall "
              f"{tot['tp'] / max(tot['tp'] + tot['fn'], 1):.0%} | false alarms {tot['fp']}/{tot['fp'] + tot['tn']}"
              "  <- report this one")
        return
    m = score(labels, recs, expected, prm, only)
    print(line(m))
    print("\n".join(m["lines"]))
    if m["fires"]:
        print("\nTypes that fire on CLEAN videos:")
        for rid, vs in sorted(m["fires"].items(), key=lambda x: -len(x[1])):
            print(f"  {rid:<30} {len(vs)}: {', '.join(vs)}")


if __name__ == "__main__":
    main()
