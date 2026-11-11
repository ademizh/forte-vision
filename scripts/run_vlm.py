"""
CLI of the verified engine (Grounding DINO + ByteTrack + temporal rules + SmolVLM2 + calibrated thresholds).

  python scripts/run_vlm.py dataset --engine verified --only X-1,X-2,X-11,X-22 --out eval/final_smoke.jsonl --overwrite
  python scripts/run_vlm.py dataset --engine verified --out eval/verified.jsonl            # all labelled videos
  python scripts/run_vlm.py evaluate --pred eval/verified.jsonl --report eval/report_verified.md
  python scripts/run_vlm.py calibrate --pred eval/verified.jsonl                           # -> configs/thresholds_verified.json
  python scripts/run_vlm.py video --path data/videos/X-26.mov --save-evidence outputs/evidence
  python scripts/run_vlm.py benchmark --path data/videos/X-14.mov

dataset writes one JSON line per video (resumes: videos already in --out are skipped unless --overwrite).
evaluate / calibrate only read that file (CPU, seconds). --engine legacy is the old Qwen + OWLv2 engine and
is run with forte.py (it is only used as baseline A in the report).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import calibration, config, engine, metrics  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # Russian text in a Windows console
except Exception:  # noqa: BLE001
    pass

VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv"}
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}


def load_truth(labels: str, catalog: str) -> dict:
    import yaml
    if not labels or labels == "none" or not Path(labels).exists():
        return {}
    cat = yaml.safe_load(open(catalog, encoding="utf-8"))
    expected = {}
    for t in cat["violations"]:
        for vid in t.get("videos", []):
            expected.setdefault(vid, set()).add(t["id"])
    truth = {}
    for r in csv.DictReader(open(labels, encoding="utf-8")):
        key = DUPLICATES.get(r["video"], r["video"])
        e = truth.setdefault(key, {"label": "no_violation", "times": [], "types": set()})
        if r["label"] == "violation":
            e["label"] = "violation"
            if r.get("timecode_sec", "").strip():
                e["times"].append(float(r["timecode_sec"]))
            e["types"] |= expected.get(r["video"], set())
    return truth


def read_jsonl(p) -> dict:
    out = {}
    if Path(p).exists():
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["video"]] = r
    return out


def _cfg(a):
    ov = {"models": {}}
    if getattr(a, "device", None):
        ov["models"]["device"] = a.device
    if getattr(a, "model_path", None):
        ov["models"]["vlm_path"] = a.model_path
    if getattr(a, "detector_path", None):
        ov["models"]["detector_path"] = a.detector_path
    for ab in (getattr(a, "ablate", "") or "").split(","):
        if ab == "novlm":
            ov["vlm"] = {"enabled": False}
        elif ab == "nofocus":
            ov["sampling"] = {"focus_pre_sec": 0.0, "focus_post_sec": 0.0, "fallback_windows": 0}
        elif ab == "notiles":
            ov["detection"] = {"tiles": {"enabled": False}}
    return config.load(getattr(a, "config", None), ov)


def _videos(src, only):
    src = Path(src)
    vids = [src] if src.is_file() else sorted((p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXT),
                                              key=lambda p: (len(p.stem), p.stem))
    if only:
        want = {x.strip() for x in only.split(",") if x.strip()}
        vids = [v for v in vids if v.stem in want]
    return [v for v in vids if v.stem not in DUPLICATES]


def cmd_dataset(a):
    if a.engine == "legacy":
        sys.exit("legacy engine (Qwen2.5-VL + OWLv2): run  python forte.py predict --videos data/videos  and "
                 "python forte.py accuracy  (baseline A is read from its cache by `evaluate`)")
    cfg = _cfg(a)
    vids = _videos(a.videos, a.only)
    if not vids:
        sys.exit(f"no videos in {a.videos}" + (f" matching {a.only}" if a.only else ""))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {} if a.overwrite else read_jsonl(out)
    if a.overwrite and out.exists():
        out.unlink()
    models = engine.Models(cfg, mock=a.mock)
    th = engine.load_thresholds(a.thresholds, cfg)
    print(f"{len(vids)} video(s), engine verified, config key {config.stage_key(cfg, 'vlm')}, "
          f"tracker {models.names()['tracker']}{'  (MOCK models: numbers are meaningless)' if a.mock else ''}")
    for k, p in enumerate(vids, 1):
        if p.stem in done:
            print(f"[{k}/{len(vids)}] {p.name}: already in {out} (use --overwrite to redo)")
            continue
        t0 = time.time()
        try:
            rec = engine.run_video(str(p), models, cfg, cache_dir=a.cache, overwrite=a.overwrite_cache,
                                   max_seconds=a.max_seconds)
        except Exception as e:  # noqa: BLE001  one broken file must not stop the batch
            print(f"[{k}/{len(vids)}] {p.name}: ERROR {type(e).__name__}: {e}", flush=True)
            if a.debug:
                raise
            continue
        res = engine.result(rec, th, cfg)
        rec["decision_with_saved_thresholds"] = {kk: v for kk, v in res.items() if kk != "events"}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_c = sum(not c.get("internal") for c in rec["candidates"])
        print(f"[{k}/{len(vids)}] {p.name:9} {rec['duration']:5.0f}s video {time.time() - t0:6.1f}s | "
              f"cashier {rec['zones']['cashier'] or '-'} | regions "
              f"{','.join(n for n, r in rec['zones']['rois'].items() if r) or '-'} | {n_c} candidates | "
              f"{rec['vlm_calls']} VLM passes{' | SKIPPED ' + str(len(rec['vlm_skipped'])) if rec['vlm_skipped'] else ''}"
              f" | {'VIOLATION ' + res['type'] + ' @' + str(res['start']) + 's [' + res['status'] + ']' if res['violation'] else 'no violation'}"
              f" | {rec['timing']} {rec['gpu'] or ''}", flush=True)
        if a.save_evidence and res["events"]:
            from pipeline import evidence
            for e in res["events"]:
                evidence.save(evidence.evidence(str(p), e, cfg), a.save_evidence, f"{p.stem}_{e['type']}")
    print(f"\nwrote {out}.  Next:  python scripts/run_vlm.py evaluate --pred {out}")


def cmd_video(a):
    a.videos, a.only, a.engine = a.path, "", "verified"
    a.out = a.out or str(ROOT / "outputs" / f"{Path(a.path).stem}.jsonl")
    a.overwrite = True
    cmd_dataset(a)
    rec = read_jsonl(a.out)[Path(a.path).stem]
    print(json.dumps(rec["decision_with_saved_thresholds"], ensure_ascii=False, indent=1))


def _rows_fixed(records, truth, th, cfg):
    return [{"video": v, "truth": truth[v], "events": calibration.decide(records[v], th, cfg)}
            for v in records if v in truth]


def _no_vlm(records, cfg):
    from pipeline import fusion
    out = {}
    for v, r in records.items():
        cands = []
        for c in r["candidates"]:
            c = dict(c)
            c["conditions"] = c.get("conditions_rule", c["conditions"])
            c.pop("conditions_rule", None)
            cands.append(c)
        out[v] = {**r, "candidates": fusion.fuse(cands, [], [], cfg)}
    return out


def _legacy_baseline(truth):
    try:
        import forte
        feats, facts, kind = forte.load_features(ROOT / "eval" / "cache", truth, ROOT / "eval" / "facts")
        if not feats:
            return None
        rows = forte.loo(feats, facts, truth, "strict", ("engine",))
        return forte.score_rows(rows, 3.0)["text"]
    except Exception as e:  # noqa: BLE001
        return f"not available ({type(e).__name__}: {e})"


def cmd_evaluate(a):
    cfg = _cfg(a)
    truth = load_truth(a.labels, a.catalog)
    records = {v: r for v, r in read_jsonl(a.pred).items() if v in truth}
    if not records:
        sys.exit(f"no labelled videos in {a.pred}")
    missing = sorted(set(truth) - set(records), key=lambda x: (len(x), x))
    print(f"{len(records)} labelled videos in {a.pred}"
          f" ({sum(truth[v]['label'] == 'violation' for v in records)} violation, "
          f"{sum(truth[v]['label'] != 'violation' for v in records)} clean)"
          + (f"; not processed: {', '.join(missing)}" if missing else ""))
    print(f"Decision: maximise {cfg['decision']['objective'].upper()} with false-alarm rate <= "
          f"{cfg['decision']['max_fpr']:.0%} on the training videos; each video judged by thresholds fitted "
          f"WITHOUT it (leave-one-video-out). 'needs review' counts as reported: "
          f"{cfg['decision']['report_review_as_violation']}\n")
    rows = calibration.loo(records, truth, cfg)
    m = metrics.summarize(rows, records, truth, a.tol)
    for r in sorted(rows, key=lambda r: (len(r["video"]), r["video"])):
        tr, ev = r["truth"], r["events"]
        lab = ("VIOL " + "/".join(f"{t:.0f}s" for t in tr["times"]) + " " + "/".join(sorted(tr["types"]))
               if tr["label"] == "violation" else "clean")
        if tr["label"] != "violation":
            tag = "FA  " if ev else "ok  "
        elif not ev:
            tag = "MISS"
        else:
            tag = "OK  " if any(calibration.same_type(e["type"], tr["types"]) for e in ev) else "WRONG-TYPE"
        txt = "; ".join(f"{e['type']}@{e['start']:.0f}s {e['score']:.2f}>={e['threshold']:.2f} [{e['status']}]"
                        for e in ev) or "-"
        print(f"  {tag:4} {r['video']:5} {lab:<52} -> {txt}")
    print(f"\nVERIFIED ENGINE, leave-one-video-out:  {metrics.line(m)}   <- report this one")
    print("  per violation recall: " + ", ".join(f"{t} {d['hit']}/{d['pos']}" for t, d in m["per_violation"].items()
                                                  if d["pos"]))
    print(f"  candidate generation recall (a candidate of the true type existed): {m['candidate_recall']:.0%}")
    print(f"  temporal error (median, right type): {m['temporal_error_median_sec']}  "
          f"runtime mean/max {m['runtime_sec']}  GPU peak {m['gpu_peak_gb']} GB")
    # ---- ablation on the same videos (CPU)
    print("\nABLATION (same videos, same protocol unless stated)")
    abl = {}
    base = _legacy_baseline(truth)
    abl["A legacy Qwen2.5-VL + OWLv2 state engine (forte.py)"] = base or "no legacy cache"
    nv = _no_vlm(records, cfg)
    abl["D detector + ByteTrack + temporal rules, no VLM"] = metrics.line(metrics.summarize(
        calibration.loo(nv, truth, cfg), nv, truth, a.tol))
    fixed = {"global": float(cfg["decision"]["default_threshold"]), "per_type": {}}
    abl[f"F + SmolVLM2 questions, fixed threshold {fixed['global']}"] = metrics.line(metrics.summarize(
        _rows_fixed(records, truth, fixed, cfg), records, truth, a.tol))
    abl["G + calibrated thresholds (LOO)"] = metrics.line(m)
    for k, v in abl.items():
        print(f"  {k:<62} {v}")
    print("  B, C, E (without tracking / adaptive sampling) need a GPU re-run: dataset --ablate nofocus|notiles")
    conf = calibration.conformal_loo(records, truth, float(cfg["decision"]["conformal_alpha"]))
    flagged = [r for r in conf["rows"] if r["flag"]]
    print(f"\nCONFORMAL (alpha {conf['alpha']}): flags {sum(r['truth'] == 'violation' for r in flagged)} violations, "
          f"{sum(r['truth'] != 'violation' for r in flagged)} clean.  " + " | ".join(conf["warnings"]))
    if a.report:
        _report(a.report, m, abl, conf, rows, cfg)
        print(f"\nreport: {a.report}")


def _report(path, m, abl, conf, rows, cfg):
    c = m["confusion"]
    lines = ["# Verified engine — evaluation", "",
             f"Protocol: leave-one-video-out; thresholds maximise {cfg['decision']['objective'].upper()} with "
             f"FPR <= {cfg['decision']['max_fpr']:.0%} on the training videos.", "",
             "| metric | value |", "|---|---|"]
    for k in ("accuracy", "precision", "recall", "f1", "f2", "fpr"):
        lines.append(f"| {k} | {m[k]:.3f} |")
    lines += [f"| confusion | TP {c['tp']} FP {c['fp']} FN {c['fn']} TN {c['tn']} |",
              f"| candidate generation recall | {m['candidate_recall']:.3f} |",
              f"| temporal error median (s) | {m['temporal_error_median_sec']} |",
              f"| runtime mean / max (s) | {m['runtime_sec']['mean']} / {m['runtime_sec']['max']} |",
              f"| GPU peak (GB) | {m['gpu_peak_gb']} |", "", "## Per violation", "",
              "| type | positives | found | recall | predicted | precision |", "|---|---|---|---|---|---|"]
    for t, d in m["per_violation"].items():
        lines.append(f"| {t} | {d['pos']} | {d['hit']} | {d['recall']:.2f} | {d['pred']} | {d['precision']:.2f} |")
    lines += ["", "## Ablation", "", "| variant | result |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in abl.items()]
    lines += ["", "## Conformal (descriptive)", ""] + [f"- {w}" for w in conf["warnings"]]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_calibrate(a):
    cfg = _cfg(a)
    truth = load_truth(a.labels, a.catalog)
    records = {v: r for v, r in read_jsonl(a.pred).items() if v in truth}
    if not records:
        sys.exit(f"no labelled videos in {a.pred}")
    th = calibration.fit(records, truth, cfg)
    th["fitted_on"] = sorted(records)
    th["config_key"] = config.stage_key(cfg, "vlm")
    Path(a.write).parent.mkdir(parents=True, exist_ok=True)
    Path(a.write).write_text(json.dumps(th, indent=1), encoding="utf-8")
    print(f"thresholds fitted on {len(records)} TRAINING videos -> {a.write}\n{json.dumps(th, indent=1)}")
    print("They are frozen: new (test) videos are only compared with them, never used to change them.")


def cmd_benchmark(a):
    cfg = _cfg(a)
    models = engine.Models(cfg, mock=a.mock)
    t0 = time.time()
    _ = models.detector
    t_det = time.time() - t0
    t0 = time.time()
    _ = models.vlm
    t_vlm = time.time() - t0
    rec = engine.run_video(a.path, models, cfg, cache_dir=None, max_seconds=a.max_seconds)
    print(json.dumps({"load_detector_sec": round(t_det, 1), "load_vlm_sec": round(t_vlm, 1),
                      "video_sec": rec["duration"], "timing": rec["timing"], "vlm_passes": rec["vlm_calls"],
                      "vlm_skipped": len(rec["vlm_skipped"]), "gpu": rec["gpu"],
                      "within_limit": rec["timing"]["total"] + t_det + t_vlm <= 300}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", default=None)
        p.add_argument("--labels", default=str(ROOT / "data" / "labels.csv"))
        p.add_argument("--catalog", default=str(ROOT / "catalog" / "violations.yaml"))
        p.add_argument("--device", default=None)
        p.add_argument("--model-path", default=None, help="SmolVLM2 folder")
        p.add_argument("--detector-path", default=None, help="Grounding DINO folder")
        p.add_argument("--ablate", default="", help="novlm,nofocus,notiles")

    def run(p):
        p.add_argument("--cache", default=str(ROOT / "eval" / "cache_verified"))
        p.add_argument("--overwrite-cache", action="store_true")
        p.add_argument("--thresholds", default=None)
        p.add_argument("--max-seconds", type=float, default=None)
        p.add_argument("--save-evidence", default="")
        p.add_argument("--mock", action="store_true", help="fake models (plumbing test without GPU)")
        p.add_argument("--debug", action="store_true")

    p = sub.add_parser("dataset")
    common(p)
    run(p)
    p.add_argument("--engine", default="verified", choices=("verified", "legacy"))
    p.add_argument("--videos", default=str(ROOT / "data" / "videos"))
    p.add_argument("--only", default="")
    p.add_argument("--out", default=str(ROOT / "eval" / "verified.jsonl"))
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_dataset)
    p = sub.add_parser("video")
    common(p)
    run(p)
    p.add_argument("--path", required=True)
    p.add_argument("--out", default="")
    p.set_defaults(fn=cmd_video)
    p = sub.add_parser("evaluate")
    common(p)
    p.add_argument("--pred", default=str(ROOT / "eval" / "verified.jsonl"))
    p.add_argument("--tol", type=float, default=3.0)
    p.add_argument("--report", default=str(ROOT / "eval" / "report_verified.md"))
    p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("calibrate")
    common(p)
    p.add_argument("--pred", default=str(ROOT / "eval" / "verified.jsonl"))
    p.add_argument("--write", default=str(ROOT / "configs" / "thresholds_verified.json"))
    p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("benchmark")
    common(p)
    run(p)
    p.add_argument("--path", required=True)
    p.set_defaults(fn=cmd_benchmark)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
