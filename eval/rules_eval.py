"""
Score the facts -> rules method on the cached facts (eval/facts/<video>.json). No GPU, runs in seconds,
so you can change rules.py or thresholds and re-run immediately.

  python eval/rules_eval.py                         # default thresholds, per-video lines
  python eval/rules_eval.py --sweep                 # table over fact threshold x duration scale
  python eval/rules_eval.py --th 0.4 --dur 1.5
  python eval/rules_eval.py --th-fact cash_on_desk=0.3,key_in_safe_lock=0.3
  python eval/rules_eval.py --dump X-26             # print the fact timeline of one video (P(yes) x10)
  python eval/rules_eval.py --holdout X-38,X-22,... # report only on these videos
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.rules import COVERED, DEFAULTS, detect, prepare  # noqa: E402

TOL = 2.0


def load(a):
    with open(a.labels, encoding="utf-8") as f:
        labels = {r["video"]: r for r in csv.DictReader(f)}
    facts = {}
    for p in sorted(Path(a.facts).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        facts[p.stem] = d
    with open(a.catalog, encoding="utf-8") as f:
        cat = yaml.safe_load(f)
    expected = {}
    for v in cat["violations"]:
        for vid in v.get("videos", []):
            expected.setdefault(vid, set()).add(v["id"])
    return labels, facts, expected


def score(labels, facts, expected, params, only=None, verbose=False):
    tp = fn = fp = tn = 0
    errs, type_ok, lines = [], [], []
    fires_clean, covered_tp, covered_n = {}, 0, 0
    for vid, lab in labels.items():
        if vid not in facts or (only and vid not in only):
            continue
        ev = detect(facts[vid], params)
        truth = lab["label"] == "violation"
        is_cov = bool(expected.get(vid, set()) & set(COVERED))
        if truth and is_cov:
            covered_n += 1
        if truth and ev:
            tp += 1
            covered_tp += is_cov
            tc = float(lab["timecode_sec"])
            best = min(ev, key=lambda e: abs(e["timecode_sec"] - tc))
            errs.append(abs(best["timecode_sec"] - tc))
            ok = best["id"] in expected.get(vid, set()) if vid in expected else None
            if ok is not None:
                type_ok.append(ok)
            lines.append(f"  TP    {vid:5} label {tc:4.0f}s  pred {best['timecode_sec']:5.1f}s {best['id']:<30}"
                         f" err {errs[-1]:4.1f}s  type {'OK' if ok else '-' if ok is None else 'WRONG'}"
                         f"  ({len(ev)} event(s))")
        elif truth:
            fn += 1
            exp = "/".join(sorted(expected.get(vid, {"?"})))
            lines.append(f"  MISS  {vid:5} label {lab['timecode_sec']:>4}s  expected {exp}"
                         f"{'' if is_cov else '  (type not covered by rules)'}")
        elif ev:
            fp += 1
            for e in ev:
                fires_clean.setdefault(e["id"], []).append(vid)
            lines.append(f"  FA    {vid:5} " + "; ".join(f"{e['id']} {e['start']:.0f}-{e['end']:.0f}s"
                                                       for e in ev[:3]))
        else:
            tn += 1
    n = tp + fn + fp + tn
    m = {"n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
         "acc": (tp + tn) / n if n else 0, "recall": tp / (tp + fn) if tp + fn else 0,
         "tc_mean": sum(errs) / len(errs) if errs else None,
         "tc_hit": sum(e <= TOL for e in errs) / len(errs) if errs else None,
         "type_acc": sum(type_ok) / len(type_ok) if type_ok else None,
         "covered": (covered_tp, covered_n), "fires_clean": fires_clean, "lines": lines}
    return m


def show(m):
    tc = f"timecode err {m['tc_mean']:.1f}s, within {TOL:.0f}s {m['tc_hit']:.0%}" if m["tc_mean"] is not None else "timecode n/a"
    ty = f" | type correct {m['type_acc']:.0%}" if m["type_acc"] is not None else ""
    print(f"videos {m['n']} | accuracy {m['acc']:.0%} | recall {m['recall']:.0%} ({m['tp']}/{m['tp'] + m['fn']}) | "
          f"false alarms {m['fp']}/{m['fp'] + m['tn']} | {tc}{ty}")
    print(f"recall on types the rules cover: {m['covered'][0]}/{m['covered'][1]}")


def dump(facts, vid, params):
    f = facts[vid]
    c = prepare(f, params)
    keys = list(f["p"])
    print(f"{vid}: P(yes) x10 after smoothing; * = ON")
    for i, k in enumerate(keys):
        print(f"  {i:>2} = {k}")
    print(" " * 9 + " ".join(f"{i:>2}" for i in range(len(keys))))
    for j, t in enumerate(c.t):
        print(f"{t:7.1f}s " + " ".join(
            f"{min(int(c.p[k][j] * 10), 9)}{'*' if c.on[k][j] else ' '}" for k in keys))
    ev = detect(f, params)
    print("\nEVENTS:" if ev else "\nEVENTS: none")
    for e in ev:
        print(f"  {e['id']} {e['start']:.0f}-{e['end']:.0f}s score {e['score']:.2f}: {e['why']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--facts", default="eval/facts")
    ap.add_argument("--catalog", default="catalog/violations.yaml")
    ap.add_argument("--th", type=float, default=DEFAULTS["th"])
    ap.add_argument("--dur", type=float, default=DEFAULTS["dur"])
    ap.add_argument("--smooth", type=int, default=DEFAULTS["smooth"])
    ap.add_argument("--th-fact", default="", help="per-fact thresholds: fact=0.3,fact2=0.6")
    ap.add_argument("--no-det", action="store_true", help="ignore detector scores (det_*), Qwen facts only")
    ap.add_argument("--th-det", type=float, default=DEFAULTS["th_det"])
    ap.add_argument("--det-use", default=",".join(DEFAULTS["det_use"]),
                    help="detector keys allowed to switch facts ON, e.g. dino_person,owl_cash,dino_cup")
    ap.add_argument("--holdout", default="")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-det", action="store_true", help="table over the detector threshold")
    ap.add_argument("--dump", default="")
    a = ap.parse_args()

    labels, facts, expected = load(a)
    if not facts:
        sys.exit("No cached facts. Run first: python scripts/run_vlm.py facts-dataset")
    th_fact = {k: float(v) for k, v in (x.split("=") for x in a.th_fact.split(",") if x)}
    params = {"th": a.th, "dur": a.dur, "smooth": a.smooth, "th_fact": th_fact,
              "use_det": not a.no_det, "th_det": a.th_det, "det_use": [x for x in a.det_use.split(",") if x]}
    only = set(a.holdout.split(",")) if a.holdout else None

    if a.dump:
        return dump(facts, a.dump, params)
    print(f"{len(facts)} videos with facts, {len(labels)} labeled")
    if a.sweep_det:
        print(f"{'th_det':>6} {'recall':>7} {'FA':>4} {'acc':>5} {'tc_err':>7} {'type':>5}")
        for td in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6):
            m = score(labels, facts, expected, {**params, "th_det": td}, only)
            tce = f"{m['tc_mean']:.1f}" if m["tc_mean"] is not None else "-"
            ty = f"{m['type_acc']:.0%}" if m["type_acc"] is not None else "-"
            print(f"{td:6.2f} {m['recall']:7.0%} {m['fp']:4d} {m['acc']:5.0%} {tce:>7} {ty:>5}")
        return
    if a.sweep:
        print(f"{'th':>5} {'dur':>5} {'recall':>7} {'FA':>4} {'acc':>5} {'tc_err':>7} {'type':>5}")
        for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for dur in (0.5, 1.0, 1.5, 2.0):
                m = score(labels, facts, expected, {**params, "th": th, "dur": dur}, only)
                tce = f"{m['tc_mean']:.1f}" if m["tc_mean"] is not None else "-"
                ty = f"{m['type_acc']:.0%}" if m["type_acc"] is not None else "-"
                print(f"{th:5.2f} {dur:5.1f} {m['recall']:7.0%} {m['fp']:4d} {m['acc']:5.0%} {tce:>7} {ty:>5}")
        return
    m = score(labels, facts, expected, params, only, verbose=True)
    show(m)
    print("\n".join(m["lines"]))
    if m["fires_clean"]:
        print("\nRules that fire on CLEAN videos (tighten these first):")
        for rid, vids in sorted(m["fires_clean"].items(), key=lambda x: -len(x[1])):
            print(f"  {rid:<30} {len(vids)}: {', '.join(vids)}")


if __name__ == "__main__":
    main()
