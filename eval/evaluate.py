"""
Score results.jsonl against labels.csv. No GPU needed: thresholds are re-applied offline
to the saved per-window answers, so you can tune MIN_CONF / MIN_WINDOWS without re-running the VLM.

  python eval/evaluate.py                              # default thresholds
  python eval/evaluate.py --min-conf 0.3 --min-windows 2
  python eval/evaluate.py --sweep                      # table over thresholds
  python eval/evaluate.py --holdout X-40,X-43,...      # report only on held-out videos
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.analyze import MIN_CONF, MIN_WINDOWS, load_catalog, merge_events  # noqa: E402

TOL = 2.0  # seconds; our own target for "timecode hit"


def load(labels_path, results_path):
    with open(labels_path, encoding="utf-8") as f:
        labels = {r["video"]: r for r in csv.DictReader(f)}
    results = {}
    for line in Path(results_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            results[r["video_id"]] = r
    return labels, results


def score(labels, results, catalog, min_conf, min_windows, only=None, verbose=False):
    tp = fn = fp = tn = 0
    errs, problems, slow, type_ok = [], [], [], []
    failed = []
    expected = {}                                   # video -> type ids from the catalog's "videos" lists
    for v in catalog["violations"]:
        for vid in v.get("videos", []):
            expected.setdefault(vid, set()).add(v["id"])
    for vid, lab in labels.items():
        if only and vid not in only:
            continue
        if vid not in results:
            continue
        r = results[vid]
        if r.get("complete") is False:
            failed.append(vid)
            problems.append(f"  FAILED {vid}: {r.get('error', 'incomplete')}")
            continue
        events = merge_events(r["windows"], catalog, min_conf, min_windows)
        pred = bool(events)
        truth = lab["label"] == "violation"
        if r["sec_total"] > 240:
            slow.append((vid, r["sec_total"]))
        if truth and pred:
            tp += 1
            tc = float(lab["timecode_sec"])
            best = min(events, key=lambda e: abs(e["timecode_sec"] - tc))
            errs.append(abs(best["timecode_sec"] - tc))
            if vid in expected:
                type_ok.append(best["type"] in expected[vid])
            if verbose:
                print(f"  TP {vid}: label {tc:.0f}s, pred {best['timecode_sec']}s ({best['type']}), "
                      f"err {errs[-1]:.1f}s, {len(events)} event(s)"
                      + ("" if vid not in expected else
                         f", type {'OK' if type_ok[-1] else 'WRONG (expected ' + '/'.join(expected[vid]) + ')'}"))
        elif truth:
            fn += 1
            problems.append(f"  MISS {vid}: label {lab['timecode_sec']}s, nothing found")
        elif pred:
            fp += 1
            e = events[0]
            problems.append(f"  FALSE ALARM {vid}: {e['type']} @ {e['timecode_sec']}s - {e['observation'][:80]}")
        else:
            tn += 1
    n = tp + fn + fp + tn
    return {
        "n": n, "failed": failed, "total_expected": sum(1 for v in labels if not only or v in only), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "acc": (tp + tn) / n if n else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "fp_rate": fp / (fp + tn) if fp + tn else 0,
        "tc_mean": sum(errs) / len(errs) if errs else None,
        "tc_hit": sum(e <= TOL for e in errs) / len(errs) if errs else None,
        "type_acc": sum(type_ok) / len(type_ok) if type_ok else None,
        "problems": problems, "slow": slow,
    }


def show(m):
    tc = (f"timecode err mean {m['tc_mean']:.1f}s, within {TOL:.0f}s: {m['tc_hit']:.0%}"
          if m["tc_mean"] is not None else "timecode: n/a")
    print(f"completed {m['n']}/{m['total_expected']} | failed {len(m['failed'])} | accuracy {m['acc']:.0%} | recall {m['recall']:.0%} ({m['tp']}/{m['tp'] + m['fn']}) | "
          f"false alarms {m['fp']}/{m['fp'] + m['tn']} | {tc}"
          + (f" | type correct {m['type_acc']:.0%}" if m["type_acc"] is not None else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--results", default="eval/results.jsonl")
    ap.add_argument("--min-conf", type=float, default=None)
    ap.add_argument("--min-windows", type=int, default=MIN_WINDOWS)
    ap.add_argument("--holdout", default="", help="comma-separated video ids to score on")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    labels, results = load(a.labels, a.results)
    hashes={r.get('config_hash','legacy') for r in results.values()}
    if len(hashes)>1: raise ValueError('Mixed configurations in results; evaluate separate files')
    if a.min_conf is None:
        a.min_conf=next((r.get('settings',{}).get('threshold',MIN_CONF) for r in results.values()
                         if r.get('complete',True)),MIN_CONF)
    catalog = load_catalog()
    only = set(a.holdout.split(",")) if a.holdout else None
    print(f"{len(results)} videos scanned, {len(labels)} labeled")

    if a.sweep:
        print(f"{'min_conf':>8} {'min_win':>7} {'recall':>7} {'FA':>5} {'acc':>5} {'tc_err':>7}")
        for c in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
            for w in (1, 2, 3):
                m = score(labels, results, catalog, c, w, only)
                tce = f"{m['tc_mean']:.1f}" if m["tc_mean"] is not None else "-"
                print(f"{c:8.1f} {w:7d} {m['recall']:7.0%} {m['fp']:5d} {m['acc']:5.0%} {tce:>7}")
        return

    m = score(labels, results, catalog, a.min_conf, a.min_windows, only, verbose=True)
    show(m)
    print("\n".join(m["problems"]) or "  no misses, no false alarms")
    for vid, s in m["slow"]:
        print(f"  SLOW {vid}: {s:.0f}s (limit 300s incl. upload)")


if __name__ == "__main__":
    main()
