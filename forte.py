"""ForteBank: frozen inference and recall-first threshold calibration.

Prepare known training videos once:
  python forte.py predict --videos data/videos --extract-only
  python forte.py accuracy --write
  python forte.py accuracy --archive-labels --write  # only if original labels.csv is unavailable

New unknown videos (never changes thresholds):
  python forte.py predict --videos D:/new --labels none

API: uvicorn app.main:app --port 8000
See README_RECALL_PATCH.md for the scope of the calibration and GPU validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}


def load_env(path: Path = ROOT / ".env"):
    """KEY=VALUE lines from .env (does not override variables already set in the shell)."""
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():   # utf-8-sig: Notepad BOM
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))
    os.environ.setdefault("VLM_MAX_PIXELS", "921600")
    os.environ.setdefault("FACT_FPS", "1")
    os.environ.setdefault("DETS", "owl")


load_env()
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------- labels and cached evidence

def load_truth(labels: str, catalog: str) -> dict:
    """{video: {"label", "times", "types"}} with duplicate files merged (X-47 = X-4, X-48 = X-5)."""
    import yaml
    from pipeline.judge import DUPLICATES
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


def load_features(cache: Path, truth: dict, facts_dir: Path):
    """Cached evidence per labelled video; a video without a cache falls back to the state engine on the old
    Qwen facts (eval/facts), so the baseline can always be measured."""
    from pipeline import judge
    from pipeline.flow import load_cached
    feats, facts, kind = {}, {}, {}
    for v in truth:
        rec, fa = load_cached(cache, v)
        if fa is not None:
            def stage(name):
                p = cache / name / f"{v}.json"
                return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
            # Rebuild from completed stages. A later inference run may have
            # intentionally skipped a branch without deleting its training cache.
            feats[v] = judge.video_features(fa, stage("chain"), stage("change"), stage("vlm"), stage("focus"), stage("door"))
            facts[v], kind[v] = fa, "full" if rec is not None else "engine-only"
            continue
        p = facts_dir / f"{v}.json"
        if p.exists():
            fa = json.loads(p.read_text(encoding="utf-8"))
            feats[v], facts[v], kind[v] = judge.engine_features(fa), fa, "engine-only"
    return feats, facts, kind


# ---------------------------------------------------------------- metrics

def score_rows(rows, tol):
    from pipeline.judge import same_type
    m = dict(n=0, tp=0, fn=0, fp=0, tn=0, loc=0, typ=0)
    for r in rows:
        tr, ev = r["truth"], r["events"]
        m["n"] += 1
        if tr["label"] == "violation":
            if ev:
                m["tp"] += 1
                near = [e for e in ev if any(abs(e["timecode_sec"] - t) <= tol for t in tr["times"])]
                if near:
                    m["loc"] += 1
                    m["typ"] += any(same_type(e["id"], tr["types"]) for e in near)
            else:
                m["fn"] += 1
        else:
            m["fp"] += bool(ev)
            m["tn"] += not ev
    m["acc"] = (m["tp"] + m["tn"]) / m["n"] if m["n"] else 0.0
    nv, nc = m["tp"] + m["fn"], m["fp"] + m["tn"]
    prec = m["tp"] / (m["tp"] + m["fp"]) if m["tp"] + m["fp"] else 0.0
    rec = m["tp"] / nv if nv else 0.0
    m["f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    m["text"] = (f"videos {m['n']} | accuracy {m['acc']:.0%} | F1 {m['f1']:.2f} | found {m['tp']}/{nv} violations | "
                 f"false alarms {m['fp']}/{nc} clean | timecode within {tol:.0f}s {m['loc']}/{nv} | "
                 f"right type there {m['typ']}/{nv}")
    if not any(r["truth"]["times"] for r in rows):
        m["text"] = m["text"].split(" | timecode within")[0] + " | localization NOT evaluated"
        m["loc"] = m["typ"] = None
    return m


def loo(feats, facts, truth, rule="strict", sources=None, margin_scale=1.0, allow_fa=0):
    from pipeline import judge
    rows = []
    for v in feats:
        cal = judge.calibrate(feats, truth, {v}, rule, sources or judge.SOURCES, margin_scale, allow_fa)
        rows.append({"video": v, "truth": truth[v], "events": judge.decide(feats[v], facts.get(v), cal)})
    return rows


def tag(r, tol):
    tr, ev = r["truth"], r["events"]
    if tr["label"] != "violation":
        return "FA  " if ev else "ok  "
    if not ev:
        return "MISS"
    return "OK  " if not tr["times"] or any(any(abs(e["timecode_sec"] - t) <= tol for t in tr["times"]) for e in ev) else "FAR "


def ev_text(ev):
    return "; ".join(f"{e['id']}@{e['timecode_sec']:.0f}s [{e['feature']} {e['score']:.2f}>{e['threshold']:.2f}]"
                     for e in ev) or "-"


# ---------------------------------------------------------------- commands

def cmd_predict(a):
    if a.mock:
        os.environ["FORTE_MOCK"] = "1"
    from pipeline import judge
    from pipeline.flow import Models, detect_stage, process, expand_force
    src = Path(a.video) if a.video else Path(a.videos)
    vids = [src] if src.is_file() else sorted((p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXT),
                                              key=lambda p: (len(p.stem), p.stem))
    if not vids:
        sys.exit(f"no videos found in {src}")
    if a.mock and Path(a.cache) == ROOT / "eval" / "cache":
        a.cache = str(ROOT / "eval" / "cache_mock")
    cache = Path(a.cache)
    frozen = None if a.extract_only else judge.load()
    if not a.extract_only and not frozen:
        sys.exit("No frozen policy. First run predict --extract-only on labelled training videos, then accuracy --write")
    force = expand_force({x for x in a.force.split(",") if x})
    models = Models(mock=a.mock)
    print(f"{len(vids)} video(s) -> cache {cache}{'   (MOCK models: numbers are meaningless)' if a.mock else ''}")

    print("pass 1/2: OWLv2 (full frame + SAHI tiles), Qwen not loaded", flush=True)
    for k, p in enumerate(vids, 1):
        try:
            new, sec = detect_stage(str(p), models, cache, force)
        except Exception as e:  # noqa: BLE001  one broken file must not stop the batch
            print(f"  [{k}/{len(vids)}] {p.name}: detector ERROR {e}", flush=True)
            continue
        print(f"  [{k}/{len(vids)}] {p.name:10} {'detected' if new else 'cached  '} {sec:5.1f}s", flush=True)
    models.release_detectors()
    force = force - {"det"}

    print("pass 2/2: Qwen facts -> observations -> norm search -> condition chains -> judge", flush=True)
    done = []
    for k, p in enumerate(vids, 1):
        t0 = time.time()
        try:
            r = process(str(p), models, cache, force, log=print,
                        run_chains=a.extract_only or judge.needed(frozen, "chain"),
                        run_vlm=a.extract_only or judge.needed(frozen, "vlm") or judge.needed(frozen, "focus"),
                        run_focus=a.extract_only or judge.needed(frozen, "focus"),
                        run_change=not a.extract_only and judge.needed(frozen, "change"))
        except Exception as e:  # noqa: BLE001
            print(f"[{k}/{len(vids)}] {p.name}: ERROR {e}", flush=True)
            continue
        rec = r["rec"]
        rois = ", ".join(n for n, x in rec["scene"]["rois"].items() if x) or "none"
        pv = rec.get("p_violation")
        pv = "-" if pv is None else f"{pv:.2f}"
        print(f"[{k}/{len(vids)}] {p.name:10} {rec['duration']:5.0f}s video {time.time() - t0:5.1f}s | "
              f"flow risk {pv} | focus windows {(rec.get('focus_coverage') or {}).get('checked', 0)} | regions {rois} | "
              f"{len(rec['observations'])} observations | rules checked: {', '.join(rec['checked']) or '-'} | "
              f"{rec['n_questions']} sequence/chain checks + {rec.get('focus_state_queries', 0)} focus state queries | {rec['timing']}", flush=True)
        done.append((p, r))

    if a.extract_only:
        if len(done) != len(vids):
            sys.exit("Extraction incomplete. Fix failed videos before calibration.")
        print("Training evidence extracted. No threshold file was modified. Next: python forte.py accuracy --write")
        return
    truth = load_truth(a.labels, a.catalog)
    note = f"Frozen thresholds: {judge.JUDGE_PATH}; prediction never updates them."
    cal = judge.load()
    if not cal:
        sys.exit("no thresholds: run with --labels data/labels.csv once (or python forte.py accuracy --write)")
    print(f"\n{note}\n\n{'video':10} {'verdict':13} {'type':32} {'time':>6}  evidence")
    out = []
    for p, r in done:
        events = judge.decide(r["rec"]["features"], r["facts"], cal)
        vd = judge.verdict(events)
        out.append({"video": p.stem, "file": p.name, **vd, "events": events})
        print(f"{p.stem:10} {vd['verdict']:13} {vd['type'] or '-':32} "
              f"{'' if vd['timecode_sec'] is None else vd['timecode_sec']:>6}  {ev_text(events)}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "verdict", "type", "timecode_sec", "score", "threshold", "evidence", "n_events", "all_events"])
        for o in out:
            w.writerow([o["video"], o["verdict"], o["type"], "" if o["timecode_sec"] is None else o["timecode_sec"],
                        o["score"], o.get("threshold", ""), o.get("feature", ""), len(o["events"]),
                        " | ".join(f"{e['id']}@{e['timecode_sec']:.1f}" for e in o["events"])])
    Path(a.out).with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{sum(o['verdict'] == 'violation' for o in out)}/{len(out)} predicted as violation -> {a.out} (+ .json)")
    if truth:
        print("honest accuracy (leave-one-video-out):  python forte.py accuracy")


def cmd_accuracy(a):
    from pipeline import judge
    truth = load_truth(a.labels, a.catalog)
    if not truth and a.archive_labels:
        manifest = json.loads((ROOT / "catalog" / "archive_labels.json").read_text(encoding="utf-8"))
        truth = {v: {"label": x["label"], "times": [], "types": set(x["types"])}
                 for v, x in manifest["videos"].items()}
        print("Archive-derived binary labels. Original labels.csv is absent; localization is NOT evaluated.")
    if not truth:
        sys.exit(f"labels not found: {a.labels}; supply the original labels.csv or explicitly use --archive-labels")
    feats, facts, kind = load_features(Path(a.cache), truth, Path(a.facts))
    from pipeline.classify import VERSION as classifier_version
    if "vlm" in a.sources.split(","):
        stale = []
        for v in feats:
            p = Path(a.cache) / "vlm" / f"{v}.json"
            fp = Path(a.cache) / "features" / f"{v}.json"
            record = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
            if not p.exists() or json.loads(p.read_text(encoding="utf-8")).get("version") != classifier_version or \
                    {k:x for k,x in record.get("inference_signature", {}).items() if k not in ("focus", "state_version")} != {k:x for k,x in judge.inference_signature(include_focus=False).items() if k != "state_version"}:
                stale.append(v)
        if stale:
            sys.exit("Old or missing whole-flow scores: " + ", ".join(stale) +
                     ". Run python forte.py predict --extract-only on training videos before calibration.")
    if "focus" in a.sources.split(","):
        from pipeline import focus
        stale = []
        for v in feats:
            p = Path(a.cache) / "focus" / f"{v}.json"
            fc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            if fc.get("version") != focus.VERSION or fc.get("signature") != focus.signature():
                stale.append(v)
        if stale:
            sys.exit("Missing or outdated focus states: " + ", ".join(stale) +
                     ". Resume python forte.py predict --extract-only; compatible OWL/facts/flow caches are reused.")
    nfull = sum(k == "full" for k in kind.values())
    missing = [v for v in truth if v not in feats]
    print(f"{len(feats)} unique labelled videos ({sum(truth[v]['label'] == 'violation' for v in feats)} violation, "
          f"{sum(truth[v]['label'] == 'no_violation' for v in feats)} clean); full evidence for {nfull}, "
          f"state engine only for {len(feats) - nfull}" + (f"; not processed: {', '.join(missing)}" if missing else ""))
    print("Protocol: each video is tested with thresholds learned only from the OTHER videos "
          "(duplicates X-47=X-4, X-48=X-5 merged).\n")
    S = tuple(a.sources.split(","))
    print(f"sources: {', '.join(S)}   allow-fa {a.allow_fa}\n")
    rows = loo(feats, facts, truth, a.rule, S, allow_fa=a.allow_fa)
    m = score_rows(rows, a.tol)
    for r in sorted(rows, key=lambda r: (len(r["video"]), r["video"])):
        tr = r["truth"]
        lab = ("VIOL " + "/".join(f"{t:.0f}s" for t in tr["times"]) + " " + "/".join(sorted(tr["types"]))) \
            if tr["label"] == "violation" else "clean"
        print(f"  {tag(r, a.tol)} {r['video']:5} {lab:<48} -> {ev_text(r['events'])}")
    print(f"\nLEAVE-ONE-OUT ({a.rule}, allow-fa {a.allow_fa}): {m['text']}   <- report this one")
    if a.details and a.rule != "recall":
        print("\nSOURCES (leave-one-out):")
        for nm, srcs in (("state engine (Qwen facts + OWLv2)", ("engine",)),
                         ("condition chains (norm search + Qwen checks)", ("chain",)),
                         ("pixel change (safe zone, new object, hands)", ("change",)),
                         ("video classifier (Qwen: sequence + detections)", ("vlm",)),
                         ("engine + video classifier", ("engine", "vlm")),
                         ("engine + change", ("engine", "change")),
                         ("all (judge)", ("engine", "chain", "change"))):
            print(f"  {nm:<46} {score_rows(loo(feats, facts, truth, a.rule, srcs), a.tol)['text']}")
        print(f"\nTRADE false alarms for found violations (chains/change only, sources {','.join(S)}, leave-one-out):")
        for k in (0, 1, 2):
            print(f"  allow-fa {k}  {score_rows(loo(feats, facts, truth, a.rule, S, allow_fa=k), a.tol)['text']}")
        print("\nEVIDENCE: own videos value / threshold (LOO), clean videos that would fire (LOO), clean top values")
        names = sorted({k for fv in feats.values() for k in fv}, key=lambda f: (judge.source(f), f))
        clean = [v for v in feats if truth[v]["label"] == "no_violation"]
        full = judge.calibrate(feats, truth, set(), a.rule, S, allow_fa=a.allow_fa)
        for f in names:
            ftype = full[f]["type"]
            cells = []
            own = [v for v in feats if (full[f].get("generic") and truth[v]["label"] == "violation")
                   or judge.same_type(ftype, truth[v]["types"])]
            for v in own:
                th = judge.calibrate(feats, truth, {v}, a.rule, S, allow_fa=a.allow_fa)[f]["threshold"]
                s = feats[v].get(f, {}).get("score", 0.0)
                cells.append(f"{v} {s:.2f}/{th:.2f}{' HIT' if s >= th else ''}")
            fires = [c for c in clean if feats[c].get(f, {}).get("score", 0.0) >=
                     judge.calibrate(feats, truth, {c}, a.rule, S, allow_fa=a.allow_fa)[f]["threshold"]]
            top = sorted((feats[c].get(f, {}).get("score", 0.0) for c in clean), reverse=True)[:3]
            print(f"  {'ON ' if full[f]['enabled'] else 'off'} {f:<34} {', '.join(cells) or '-':<46} "
                  f"clean firing: {', '.join(fires) or 'none':<14} clean top {' '.join(f'{x:.2f}' for x in top)}")
    if a.rule == "recall":
        from pipeline.recall_policy import META
        full = judge.calibrate(feats, truth, set(), a.rule, S)
        summary = full[META]
        print("\nCALIBRATION ONLY (not an unseen-video score):")
        print(f"misses {len(summary['calibration_misses'])}/{summary['positive_count']}; "
              f"false alarms {len(summary['calibration_false_alarms'])}/{summary['negative_count']}")
        print("selected thresholds:", summary["selected"])
        print("No zero-miss guarantee on unseen video. Raw scores overlap; inspect this trade-off before deployment.")
        import pipeline.state_engine as se
        report = {"protocol": "leave one entire unique video out; all threshold choices refit inside the fold",
                  "state_version": se.VERSION, "metrics": m, "localization_evaluated": any(t["times"] for t in truth.values()),
                  "threshold_calibration": summary,
                  "rows": [{"video": r["video"], "truth": r["truth"]["label"], "events": r["events"]} for r in rows]}
        # One fixed score, several operating points: expose the trade-off rather than hide it behind F1.
        if "v_judge" in full:
            frontier = []
            for tau in sorted({0.0, 1.000001} | {float(fv["v_judge"]["score"]) for fv in feats.values() if "v_judge" in fv}):
                trial = {k: dict(v) for k, v in full.items()}
                trial["v_judge"].update(enabled=True, threshold=tau)
                rr = [{"video":v,"truth":truth[v],"events":judge.decide(feats[v],facts[v],trial)} for v in feats]
                counts = score_rows(rr,a.tol)
                frontier.append({"threshold":tau,"fn":counts["fn"],"fp":counts["fp"],"tp":counts["tp"],"tn":counts["tn"]})
            report["calibration_frontier_not_test_performance"] = frontier
        report_path = ROOT / "outputs" / "recall_validation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.write:
        mocked = []
        for v in feats:
            p = Path(a.cache) / "features" / f"{v}.json"
            if p.exists() and json.loads(p.read_text(encoding="utf-8")).get("mock"):
                mocked.append(v)
        if mocked:
            sys.exit("Refusing to deploy thresholds calibrated on MOCK observations: " + ", ".join(mocked))
    if a.write and missing:
        sys.exit("Refusing to save incomplete calibration: " + ", ".join(missing))
    if a.write:
        cal = judge.calibrate(feats, truth, set(), a.rule, S, allow_fa=a.allow_fa)
        from pipeline.recall_policy import META
        if cal.get(META, {}).get("calibration_misses"):
            sys.exit("Cannot write zero-miss policy: some calibration positives have no usable evidence.")
        if META in cal:
            cal[META]["required_fact_keys"] = sorted(set.intersection(*(set(fa["p"]) for fa in facts.values())))
        judge.save(cal, {"note": "recall-first joint calibration" if a.rule == "recall" else "legacy clean-max calibration",
                         "state_version": __import__("pipeline.state_engine", fromlist=["VERSION"]).VERSION,
                         "classifier_version": classifier_version,
                         "rule": a.rule, "allow_fa": a.allow_fa, "sources": list(S), "videos": sorted(feats)})
        on = [f for f, c in cal.items() if c["enabled"]]
        print(f"\nwrote {judge.JUDGE_PATH}: {len(on)} evidence ON: {', '.join(on)}")


def cmd_report(a):
    from pipeline import judge
    from pipeline.flow import frames_of, load_cached
    name = Path(a.video).stem
    rec, facts = load_cached(Path(a.cache), name)
    if rec is None:
        sys.exit(f"{name}: not in {a.cache} (run predict first)")
    cal = judge.load()
    print(f"{name}: {rec['duration']:.0f}s, detectors {facts.get('detectors')}, timing {rec['timing']}")
    for n_, roi in rec["scene"]["rois"].items():
        print(f"  region {n_:<7} " + (f"OWLv2 '{roi['prompt']}' score {roi['score']:.2f} stability {roi['stability']:.2f} "
                                       f"box {[round(x) for x in roi['box']]}" if roi else "not found"))
    print("\n  OBSERVATIONS")
    for e in rec["observations"]:
        print(f"    {e['en']}")
    print("\n  NORM SEARCH: " + ", ".join(f"{r['id']} ({r['score']:.1f})" for r in rec["retrieved"]))
    print(f"  rules checked: {', '.join(rec['checked']) or '-'}")
    ch = Path(a.cache) / "chain" / f"{name}.json"
    chains = json.loads(ch.read_text(encoding="utf-8"))["chains"] if ch.exists() else []
    for c in chains:
        print(f"\n  CHAIN {c['type']} @ {c['t']:.0f}s -> score {c['score']:.2f}")
        for s in c["steps"]:
            print(f"    {s['name']:<14} {s.get('p', 0):.2f}  {s['ru']}  ({s['kind']})")
    print(f"\n  {'evidence':<34} {'value':>6} {'thresh':>7}  state")
    for f, x in sorted(rec["features"].items(), key=lambda kv: (judge.source(kv[0]), kv[0])):
        c = cal.get(f)
        stt = "no threshold" if not c else ("off" if not c["enabled"] else ("FIRES" if x["score"] >= c["threshold"] else "below"))
        print(f"  {f:<34} {x['score']:6.2f} {c['threshold'] if c else float('nan'):7.2f}  {stt}  @{x['t']:.0f}s")
    if a.crops and chains:
        from pipeline.chain import _images
        vf = next((p for p in (Path(a.videos) / f"{name}{e}" for e in (".mov", ".MOV", ".mp4", ".avi")) if p.exists()), None)
        if vf is None:
            print(f"  (video {name} not found in {a.videos}: no crops saved)")
        else:
            frames = frames_of(str(vf))
            out = ROOT / "outputs" / "crops" / name
            out.mkdir(parents=True, exist_ok=True)
            for c in chains:
                for s in c["steps"]:
                    for k, fr in enumerate(_images(s.get("images", []), frames)):
                        fr.image.save(out / f"{c['t']:05.1f}s_{c['type']}_{s['name']}_{k}_p{s.get('p', 0):.2f}.jpg", quality=90)
            print(f"\n  images shown to Qwen saved to {out}")
    events = judge.decide(rec["features"], facts, cal)
    print(f"\n  verdict: {judge.verdict(events)}")
    for e in events:
        print(f"    {e['id']} @ {e['timecode_sec']:.1f}s: {e.get('why', '')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version="Forte inclusive v3 (focus-states-v1, classify-flow-v2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cache", default=str(ROOT / "eval" / "cache"))
        p.add_argument("--labels", default=str(ROOT / "data" / "labels.csv"))
        p.add_argument("--catalog", default=str(ROOT / "catalog" / "violations.yaml"))
        p.add_argument("--facts", default=str(ROOT / "eval" / "facts"), help="earlier Qwen facts to reuse")
        p.add_argument("--rule", default="strict", choices=("recall", "strict", "headroom", "support"))
        p.add_argument("--tol", type=float, default=3.0)
        p.add_argument("--allow-fa", type=int, default=0, help="chains/change: up to N training clean videos may fire")
        p.add_argument("--sources", default="engine,door,focus", help="evidence used by the judge: engine,door,focus,chain,change,vlm")

    p = sub.add_parser("predict")
    common(p)
    p.add_argument("--videos", default=str(ROOT / "data" / "videos"))
    p.add_argument("--video", default="")
    p.add_argument("--out", default=str(ROOT / "outputs" / "predictions.csv"))
    p.add_argument("--force", default="", help="recompute stages: det,facts,chain,change,vlm,focus,door")
    p.add_argument("--mock", action="store_true", help="fake models, to test the plumbing without a GPU")
    p.add_argument("--extract-only", action="store_true", help="extract TRAINING evidence without loading or changing thresholds")
    p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("accuracy")
    common(p)
    p.add_argument("--details", action="store_true")
    p.add_argument("--archive-labels", action="store_true", help="use explicitly reconstructed archive labels; no timing evaluation")
    p.add_argument("--write", action="store_true", help="save thresholds from all videos for the service")
    p.set_defaults(fn=cmd_accuracy)
    p = sub.add_parser("report")
    common(p)
    p.add_argument("--video", required=True)
    p.add_argument("--videos", default=str(ROOT / "data" / "videos"))
    p.add_argument("--crops", action="store_true", help="save the images Qwen saw to outputs/crops/<video>/")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
