"""
Decision thresholds for the verified engine (no labels are ever used at inference time).

    th = fit(records, truth, cfg)                 # from TRAINING videos only
    events = decide(record, th, cfg)              # one video, frozen thresholds
    rows = loo(records, truth, cfg)               # honest estimate: each video judged by thresholds fitted
                                                  # without it (the same protocol the final model is used in)

Objective (decision.objective / max_fpr): maximise F2 (recall counts twice: "a missed violation is
unacceptable") subject to the false-alarm rate on clean training videos <= max_fpr; ties -> the higher
threshold (fewer false alarms). One global threshold on the fused score; a violation type gets its own
threshold only when it has >= per_type_min_positives positive training videos (otherwise a single example
would set it). Rejected and internal candidates never count.

Conformal option (split-conformal / conformal p-values, Bates et al., Ann. Statist. 2023): whole-video
nonconformity = the highest fused score over all types and time windows (all scores live on the same 0..1
scale, so no extra normalisation is fitted); p = (1 + #clean calibration videos with a score >= s) / (n + 1);
flag when p <= alpha. The guarantee needs clean calibration videos that were NOT used to tune rules or
prompts and that are exchangeable with the test videos (same camera). Here the same clean videos were used
for development and the test camera differs, so the result is reported as descriptive only and a warning
is attached; with n clean videos the smallest reachable p is 1 / (n + 1).
"""
from __future__ import annotations

COLLECTOR_TYPES = ("collection_bag_not_inspected", "cash_collection_without_checks")


def same_type(pred: str, true_types) -> bool:
    """The two collector types are one observable event (a short handover visit) and count as the same."""
    true_types = set(true_types)
    return pred in true_types or (pred in COLLECTOR_TYPES and bool(true_types & set(COLLECTOR_TYPES)))


def video_scores(rec: dict) -> dict:
    """type -> best fused score of a reportable candidate."""
    out = {}
    for c in rec.get("candidates", []):
        if c.get("internal") or c.get("status") == "rejected":
            continue
        out[c["type"]] = max(out.get(c["type"], 0.0), float(c.get("score", 0.0)))
    return out


def _flag(scores: dict, th: dict) -> bool:
    return any(s >= th["per_type"].get(t, th["global"]) for t, s in scores.items() if s > 0)


def _fbeta(tp, fp, fn, beta=2.0):
    b2 = beta * beta
    den = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / den if den else 0.0


def _evaluate(S, truth, th):
    tp = fp = fn = tn = 0
    for v, sc in S.items():
        f = _flag(sc, th)
        if truth[v]["label"] == "violation":
            tp += f
            fn += not f
        else:
            fp += f
            tn += not f
    return tp, fp, fn, tn


def fit(records: dict, truth: dict, cfg: dict, exclude=()) -> dict:
    dc = cfg["decision"]
    S = {v: video_scores(r) for v, r in records.items() if v in truth and v not in exclude}
    n_clean = sum(truth[v]["label"] != "violation" for v in S)
    max_fpr = float(dc["max_fpr"])
    cands = sorted({s for sc in S.values() for s in sc.values() if s > 0} | {1.01})

    def best_for(make):
        best = None
        for tau in cands:
            th = make(tau)
            tp, fp, fn, tn = _evaluate(S, truth, th)
            fpr = fp / max(1, fp + tn)
            if fpr > max_fpr + 1e-9:
                continue
            key = (_fbeta(tp, fp, fn), tau)
            if best is None or key > best[0]:
                best = (key, tau)
        return best[1] if best else 1.01

    g = best_for(lambda tau: {"global": tau, "per_type": {}})
    th = {"global": round(g, 4), "per_type": {}}
    pos = {}
    for v, sc in S.items():
        if truth[v]["label"] == "violation":
            for t in truth[v]["types"]:
                pos[t] = pos.get(t, 0) + 1
    for t, k in sorted(pos.items()):
        if k < int(dc["per_type_min_positives"]):
            continue
        tau_t = best_for(lambda tau, t=t: {"global": th["global"], "per_type": {**th["per_type"], t: tau}})
        th["per_type"][t] = round(tau_t, 4)
    tp, fp, fn, tn = _evaluate(S, truth, th)
    th["train"] = {"videos": len(S), "clean": n_clean, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                   "objective": dc["objective"], "max_fpr": max_fpr}
    if n_clean < 1.0 / max(1e-6, max_fpr):
        th["warning"] = (f"only {n_clean} clean training videos: a false-alarm rate of {max_fpr:.0%} cannot be "
                         f"estimated precisely (one video = {1 / max(1, n_clean):.0%})")
    return th


def decide(rec: dict, th: dict, cfg: dict) -> list[dict]:
    review_ok = bool(cfg["decision"].get("report_review_as_violation", True))
    out = []
    for c in rec.get("candidates", []):
        if c.get("internal") or c.get("status") == "rejected":
            continue
        tau = th["per_type"].get(c["type"], th["global"])
        if float(c.get("score", 0.0)) < tau or float(c.get("score", 0.0)) <= 0:
            continue
        if c["status"] == "needs_review" and not review_ok:
            continue
        out.append({**c, "threshold": tau})
    best = {}
    for e in out:                                        # one event per type, the strongest
        if e["type"] not in best or e["score"] > best[e["type"]]["score"]:
            best[e["type"]] = e
    return sorted(best.values(), key=lambda e: -(e["score"] - e["threshold"]))


def loo(records: dict, truth: dict, cfg: dict) -> list[dict]:
    rows = []
    vids = [v for v in records if v in truth]
    for v in vids:
        th = fit(records, truth, cfg, exclude={v})
        rows.append({"video": v, "truth": truth[v], "events": decide(records[v], th, cfg), "threshold": th})
    return rows


# ------------------------------------------------------------------ conformal

def video_score(rec: dict) -> float:
    return max(video_scores(rec).values(), default=0.0)


def conformal_p(score: float, clean_scores: list[float]) -> float:
    return (1 + sum(s >= score for s in clean_scores)) / (len(clean_scores) + 1)


def conformal_loo(records: dict, truth: dict, alpha: float) -> dict:
    vids = [v for v in records if v in truth]
    clean = [v for v in vids if truth[v]["label"] != "violation"]
    rows = []
    for v in vids:
        cal = [video_score(records[c]) for c in clean if c != v]
        p = conformal_p(video_score(records[v]), cal)
        rows.append({"video": v, "p": round(p, 4), "flag": p <= alpha, "truth": truth[v]["label"]})
    n = len(clean) - 1
    warn = []
    if 1.0 / (n + 1) > alpha:
        warn.append(f"with {n} clean calibration videos the smallest p-value is {1 / (n + 1):.3f} > alpha {alpha}: "
                    f"nothing can be flagged at this level")
    warn.append("descriptive only: the clean videos were also used for development and the test camera differs, "
                "so the exchangeability assumption of the conformal guarantee does not hold")
    return {"rows": rows, "alpha": alpha, "warnings": warn}

