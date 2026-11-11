"""Inclusive frozen calibration: strong events OR flow risk OR focus risk.

Two generic cutoffs are jointly chosen to minimize (missed videos, false alarms,
active generic branches). Candidate admission .15 is never a final verdict.
Every threshold decision must be repeated inside the evaluation fold.
"""
from __future__ import annotations
import math
META='__recall_policy__'
VERSION='recall-inclusive-v3'


def choose_threshold(labels,scores):
    if len(labels)!=len(scores) or len(labels)==0:raise ValueError('Nonempty aligned labels and scores are required')
    if any(not math.isfinite(float(s)) for s in scores):raise ValueError('Nonfinite calibration score')
    positives=[float(s) for y,s in zip(labels,scores) if y]
    if not positives:raise ValueError('At least one positive calibration example is required')
    return min(positives)


def fit(feats,truth,source_fn,exclude=frozenset(),sources=('engine','chain','vlm','focus'),max_rules=2):
    names=sorted(v for v in feats if v in truth and v not in exclude)
    pos=[v for v in names if truth[v]['label']=='violation']
    clean=[v for v in names if truth[v]['label']=='no_violation']
    if not pos or not clean:raise ValueError('Calibration requires both violation and clean videos')
    channels=sorted({f for v in names for f in feats[v] if source_fn(f) in sources})
    out={}
    def value(v,f):
        x=float(feats[v].get(f,{}).get('score',0.))
        if not math.isfinite(x) or not 0<=x<=1:raise ValueError(f'Invalid score: {v}/{f}')
        return x
    def own(v,typ):
        types=set(truth[v].get('types',[]));coll={'collection_bag_not_inspected','cash_collection_without_checks'}
        return typ in types or (typ in coll and bool(types&coll))
    for f in channels:
        examples=[feats[v][f] for v in names if f in feats[v]]
        src,typ=source_fn(f),examples[0]['type'];cmax=max(value(v,f) for v in clean)
        tau=min(1.000001,max(cmax+.05,.10))
        hits=[v for v in pos if value(v,f)>=tau and own(v,typ) and
              (src!='focus' or feats[v].get(f,{}).get('type_supported',False))]
        out[f]={'type':typ,'source':src,'threshold':tau,
                'enabled':bool(hits) and src in ('engine','chain','focus') and f!='f_any',
                'generic':any(x.get('generic') for x in examples),'clean_max':cmax,
                'own_hits':hits,'clean_firing_loo':[],'selected_for_recall':False}
    generic=[f for f in ('v_judge','f_any') if f in out]
    for f in generic:
        if any(f not in feats[v] for v in names):raise ValueError(f'Missing {f} scores; resume predict --extract-only')
    fixed=[f for f,c in out.items() if c['enabled'] and f not in generic]
    def physical(v):return any(value(v,f)>0 and value(v,f)>=out[f]['threshold'] for f in fixed)
    uncovered=[v for v in pos if not physical(v)]
    # All useful focus cutoffs lie at positive-score boundaries. For each one,
    # the highest flow cutoff covering the remaining positives is optimal.
    focus_thresholds=[1.000001]
    if 'f_any' in generic:focus_thresholds+=sorted({value(v,'f_any') for v in uncovered if value(v,'f_any')>0})
    best=None
    for tf in focus_thresholds:
        rem=[v for v in uncovered if 'f_any' not in generic or value(v,'f_any')<=0 or value(v,'f_any')<tf]
        tv=choose_threshold([True]*len(rem),[value(v,'v_judge') for v in rem]) if rem and 'v_judge' in generic else 1.000001
        trial={f:t for f,t in (('v_judge',tv),('f_any',tf)) if f in generic and t<=1}
        def fires(v):return physical(v) or any(value(v,f)>0 and value(v,f)>=t for f,t in trial.items())
        miss=[v for v in pos if not fires(v)];fa=[v for v in clean if fires(v)]
        key=(len(miss),len(fa),len(trial),'f_any' in trial,-tv,-tf)
        if best is None or key<best[0]:best=(key,trial,miss,fa)
    _,selected,missed,false_alarms=best
    for f in generic:
        out[f].update(enabled=f in selected,threshold=selected.get(f,1.000001),generic=True,
                      selected_for_recall=f in selected)
    out[META]={'enabled':False,'source':'policy','version':VERSION,
               'objective':'minimize (missed videos, false alarms, active generic branches) for two frozen scores',
               'training_videos':names,'selected':[{'feature':f,'threshold':t} for f,t in selected.items()],
               'calibration_misses':missed,'calibration_false_alarms':false_alarms,
               'positive_count':len(pos),'negative_count':len(clean),'unseen_zero_miss_guarantee':False}
    return out
