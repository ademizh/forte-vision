"""Local multi-label observation -> targeted verification -> coherent evidence events.
Detector scores propose crops/checks; they never independently establish violations.
No training labels, video IDs, or catalog `videos` entries enter model prompts.
"""
import hashlib
import json
import math
import os
import time
from pathlib import Path

from .event_checks import CHECKS, OBJECT_TYPES, GLOBAL_CONTEXT
from .frames import Frame, sample_frames, sliding_windows, video_info
from .vlm import get_vlm, parse_json

VERSION = 'verified-native-tracks-v3'
SYSTEM = 'You inspect ordered surveillance frames. Report only visible evidence. ' + GLOBAL_CONTEXT


def settings():
    cfg = {
        'fps': float(os.environ.get('VERIFY_FPS', '2')),
        'window': float(os.environ.get('VERIFY_WINDOW', '8')),
        'step': float(os.environ.get('VERIFY_STEP', '6')),
        'context': float(os.environ.get('VERIFY_CONTEXT', '12')),
        'frames': int(os.environ.get('VERIFY_FRAMES', '8')),
        'width': int(os.environ.get('VERIFY_WIDTH', '1280')),
        'threshold': float(os.environ.get('VERIFY_THRESHOLD', '0.55')),
        'max_checks': int(os.environ.get('VERIFY_MAX_CHECKS', '80')),
        'seconds': float(os.environ.get('VERIFY_SECONDS', '270')),
        'detectors': os.environ.get('VERIFY_DETECTORS', 'auto').strip().lower(),
        'det_frames': int(os.environ.get('VERIFY_DET_FRAMES', '24')),
        'det_device': os.environ.get('VERIFY_DET_DEVICE', 'cuda'),
        'det_floor': float(os.environ.get('VERIFY_DET_FLOOR', '0.15')),
        'tile_frames': int(os.environ.get('VERIFY_TILE_FRAMES', '12')),
        'tile_size': int(os.environ.get('VERIFY_TILE_SIZE', '768')),
        'tile_overlap': float(os.environ.get('VERIFY_TILE_OVERLAP', '0.25')),
        'max_tiles': int(os.environ.get('VERIFY_MAX_TILES', '6')),
        'det_topk': int(os.environ.get('VERIFY_DET_TOPK', '4')),
        'track_high': float(os.environ.get('VERIFY_TRACK_HIGH', '0.25')),
        'track_iou': float(os.environ.get('VERIFY_TRACK_IOU', '0.15')),
        'track_gap': float(os.environ.get('VERIFY_TRACK_GAP', '3')),
        'roi_tracks': int(os.environ.get('VERIFY_ROI_TRACKS', '2')),

    }
    if any(cfg[k] <= 0 for k in ('fps','window','step','frames','width','max_checks','seconds','det_frames')):
        raise ValueError('VERIFY sizes and time budgets must be positive')
    if cfg['context'] < 0 or cfg['step'] > cfg['window']:
        raise ValueError('VERIFY_STEP must not exceed VERIFY_WINDOW; context must be nonnegative')
    if not 0 < cfg['threshold'] < 1 or not 0 <= cfg['det_floor'] <= 1:
        raise ValueError('Invalid verification/detector thresholds')
    if cfg['tile_size'] < 32 or cfg['tile_frames'] < 0 or not 0 <= cfg['tile_overlap'] < 1:
        raise ValueError('Invalid tile settings')
    if any(cfg[k] <= 0 for k in ('max_tiles','det_topk','track_gap','roi_tracks')):
        raise ValueError('Invalid tracking settings')
    if not cfg['det_floor'] <= cfg['track_high'] <= 1 or not 0 < cfg['track_iou'] <= 1:
        raise ValueError('Invalid tracking thresholds')
    return cfg


def fingerprint(catalog, cfg):
    public = [{k: v.get(k) for k in ('id','definition','source','legal_status')} for v in catalog['violations']]
    value = {'version': VERSION, 'settings': cfg, 'catalog': public, 'checks': CHECKS,
             'vlm_path': os.environ.get('VLM_PATH'), 'pixels': os.environ.get('VLM_MAX_PIXELS'),
             'quant': os.environ.get('VLM_QUANT')}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def evenly(frames, n):
    if len(frames) <= n: return list(frames)
    if n == 1: return [frames[-1]]
    return [frames[round(i*(len(frames)-1)/(n-1))] for i in range(n)]


def decode(text):
    # Accept a COMPLETE outer object inside prose/fences. Never complete truncated
    # JSON or salvage a nested object from a broken outer response.
    decoder = json.JSONDecoder()
    start = text.find('{')
    if start >= 0:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict) and obj and 'parse_error' not in obj:
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError('Model returned invalid or truncated JSON')


def checked_generate(vlm, frames, prompt, tokens=900, validator=None):
    attempts = []
    for attempt in range(2):
        suffix = '' if not attempt else '\nPrevious response did not match the schema. Output exactly ONE complete JSON object. Keep all text values short. No prose or markdown. Use integer image indices, not timestamps.'
        raw = vlm.generate([(SYSTEM, frames, prompt + suffix)],
                           max_new_tokens=tokens if not attempt else max(tokens,1200))[0]
        try:
            obj = decode(raw)
            if validator is not None and not validator(obj):
                raise ValueError('Malformed response schema')
            return obj
        except ValueError as exc:
            attempts.append({'attempt':attempt+1,'error':str(exc),'raw_response':raw})
    import uuid
    directory = Path(os.environ.get('VERIFY_DEBUG_DIR','outputs/verified_debug'))
    directory.mkdir(parents=True,exist_ok=True)
    target = directory / (uuid.uuid4().hex + '.json')
    target.write_text(json.dumps({'prompt':prompt,'image_times':[f.t for f in frames],
                                 'attempts':attempts},ensure_ascii=False,indent=2),encoding='utf-8')
    raise ValueError(f'Model response failed validation; raw responses saved to {target}')


def detector_candidates(rows, a, b, high):
    """Only persistent object hypotheses route extra checks. Procedure omissions
    require VLM context; a bag alone must not trigger every collection procedure."""
    from collections import defaultdict
    routes={'key':['safe_key_unattended'], 'cash':['cash_unattended'],
            'bag':['personal_items_cash_area']}
    groups=defaultdict(list)
    for row in rows:
        if a <= row['t'] <= b and row.get('track_id') is not None and row['object'] in routes:
            groups[(row['object'],row['track_id'])].append(row)
    result={}
    for (obj,_), group in groups.items():
        if len({r['t'] for r in group}) < 2 or max(r['score'] for r in group)<high:
            continue
        best=max(group,key=lambda r:r['score'])
        for kind in routes[obj]: result.setdefault(kind,best['t'])
    return result


def collect_detections(frames, cfg, guard, video_path=None):
    from . import detect
    mode = cfg['detectors']
    if mode == 'off': return [], []
    names = detect.available() if mode == 'auto' else [s.strip() for s in mode.split(',')]
    if any(n not in ('dino','owl') for n in names):
        raise ValueError('VERIFY_DETECTORS must be auto, off, dino, owl or dino,owl')
    if not names:
        raise RuntimeError('No local detectors available. Set DINO_PATH/OWL_PATH or explicitly VERIFY_DETECTORS=off')
    if video_path is None:
        raise ValueError('Native detection requires source video')
    from .roi_tracking import NativeReader, detect_native, associate
    rows = []
    sampled = evenly(frames, cfg['det_frames'])
    tile_times = {f.t for f in evenly(sampled, cfg['tile_frames'])} if cfg['tile_frames'] else set()
    for name in names:
        guard()
        detector = detect.Detector(detect.PATHS[name], device=cfg['det_device'], name=name)
        try:
            with NativeReader(video_path) as reader:
                for fr in sampled:
                    guard()
                    rows.extend(detect_native(detector, reader.get(fr.t), cfg, fr.t in tile_times, guard))
        finally:
            detector.model.to('cpu')
            del detector
            import torch
            torch.cuda.empty_cache()
    # Sample spacing grows on long videos; allow at most two sampled intervals.
    gap = max(cfg['track_gap'], 2*(sampled[-1].t-sampled[0].t)/max(1,len(sampled)-1))
    rows = associate(rows, cfg['track_high'], cfg['track_iou'], gap)
    print(f'[roi] native detection: {len(sampled)} frames, {len(tile_times)} tiled; '
          f'{len(rows)} boxes, {len({r["track_id"] for r in rows if "track_id" in r})} tracks')
    return rows, names


def detector_text(rows, a, b, objects=None):
    relevant=[r for r in rows if a <= r['t'] <= b and (objects is None or r['object'] in objects)]
    # Retain one best proposal per object and detector, with timestamp; never claim absence.
    best={}
    for r in relevant:
        key=(r['detector'],r['object'])
        if key not in best or r['score']>best[key]['score']: best[key]=r
    return '\n'.join(f"{r['detector']} proposes {r['object']} at {r['t']:.2f}s, raw score {r['score']:.3f}"
                     for r in best.values()) or 'No detector proposals. This does not establish object absence.'


def candidate_prompt(catalog, det_text, memory):
    defs='\n'.join(v['id']+': '+v['definition'] for v in catalog['violations'] if v['id'] in CHECKS)
    return (GLOBAL_CONTEXT + '\nPossible event categories to investigate independently:\n'+defs+
        '\nInspect safe/key, door, personal items, unattended cash and screen as well as customer actions. '
        'Several categories can occur together. A proposal is NOT a confirmed violation. '
        'For small unclear objects propose a closer inspection rather than inventing details. '
        '\nEarlier model observations (unverified context): '+memory+
        '\nDetector proposals (unverified, unequal score scales):\n'+det_text+
        '\nReturn JSON: {"summary":"brief visible actions and states", '
        '"candidates":[{"type":"catalog id","frame":0,"observation":"visible reason to inspect"}]}. '
        'frame is an index in the supplied images, not a second. Return candidates: [] if none. '
        'Keep the summary under 25 words, observations under 10 words, and at most 3 candidates. Output the complete JSON object.')


def inspection_frames(frames, a, b, focus_t, rows, kind, n, video_path=None, cfg=None):
    from .roi_tracking import NativeReader, crop_sequence
    selected = evenly([f for f in frames if a <= f.t <= b], n)
    target = min(frames, key=lambda f: abs(f.t-focus_t))
    if not any(f.t == target.t for f in selected): selected.append(target)
    selected.sort(key=lambda f: f.t)
    metadata = [{'t': f.t, 'view': 'full'} for f in selected]
    objects = {o for o, types in OBJECT_TYPES.items() if kind in types}
    if video_path is not None:
        with NativeReader(video_path) as reader:
            crops, meta = crop_sequence(reader, frames, rows, objects, a, b, focus_t, cfg or settings())
        for crop, m in zip(crops, meta):
            nearest = min(frames, key=lambda f: abs(f.t-crop.t))
            if abs(nearest.t-crop.t) < .06:
                m['native_timestamp'] = crop.t
                m['t'] = nearest.t
                crop.t = nearest.t
        selected.extend(crops); metadata.extend(meta)
    return selected, metadata


def verification_prompt(kind, entry, det_text, metadata):
    criteria=CHECKS[kind]
    checks='\n'.join(k+': '+v for k,v in criteria.items())
    views='; '.join(f"{i}={m['view']}@{m['t']:.2f}s track={m.get('track_id','none')} location={m.get('location_status','full scene')}" for i,m in enumerate(metadata))
    return (GLOBAL_CONTEXT+'\nCheck ONLY this event: '+entry['definition']+
        '\nEvery required condition must have visible support:\n'+checks+
        '\nImage map: '+views+'. Crops are details of existing timestamps, not later moments. Track IDs are unverified associations. search_region_only means inspect that location, NOT that an object is present. Verify actual visible identity and camera motion; never infer persistence from the track ID.'+
        '\nDetector suggestions only, verify in images:\n'+det_text+
        '\nReturn JSON: {"checks": {'+
        ', '.join('"'+k+'": {"state":"yes|no|unknown","frames":[0]}' for k in criteria)+
        '}, "evidence_frames":[0], "observation":"one short Russian sentence describing the supported event"}. '
        'Use valid supplied image indexes. For unknown use frames: []. '
        'Evidence must show the event AFTER relevant transitions, not just earlier presence. '
        'For omitted procedures, coverage must include the whole relevant operation. '
        'Do not give legal conclusions or invented confidence percentages.')


def validate_verification(obj, kind, metadata, a, b):
    checks=obj.get('checks')
    if not isinstance(checks,dict): return False, 'missing_checks', []
    for key in CHECKS[kind]:
        item=checks.get(key)
        if not isinstance(item,dict) or item.get('state')!='yes': return False,'unconfirmed_'+key,[]
        indexes=item.get('frames')
        if not isinstance(indexes,list) or not indexes: return False,'no_evidence_'+key,[]
        if any(type(i) is not int or not 0<=i<len(metadata) for i in indexes): return False,'bad_frame_index',[]
    if kind=='service_door_left_open':
        ts={metadata[i]['t'] for i in checks['persistent']['frames']}
        if len(ts)<2 or max(ts)-min(ts)<1: return False,'door_not_persistent',[]
    order_pairs = {
        'cash_unattended': [('cash','after'), ('away','after')],
        'safe_key_unattended': [('key','after'), ('away','after')],
        'counting_without_customer': [('receipt','departure'), ('departure','counting')],
    }
    for before, after in order_pairs.get(kind, []):
        t_before = min(metadata[i]['t'] for i in checks[before]['frames'])
        t_after = max(metadata[i]['t'] for i in checks[after]['frames'])
        # away and after may legitimately be observed in the same frame.
        if t_after < t_before or (before != 'away' and t_after == t_before):
            return False, 'transition_order_not_supported', []
    indexes=obj.get('evidence_frames')
    if not isinstance(indexes,list) or not indexes or any(type(i) is not int or not 0<=i<len(metadata) for i in indexes):
        return False,'bad_event_evidence',[]
    decisive={'cash_unattended':'after','safe_key_unattended':'after',
              'counting_without_customer':'counting','service_door_left_open':'persistent',
              'unlocked_bank_terminal':'unlocked'}.get(kind)
    allowed={metadata[i]['t'] for i in checks[decisive]['frames']} if decisive else None
    times=sorted({metadata[i]['t'] for i in indexes if a<=metadata[i]['t']<=b
                  and (allowed is None or metadata[i]['t'] in allowed)})
    if not times: return False,'evidence_outside_event',[]
    if not isinstance(obj.get('observation'),str) or not obj['observation'].strip():
        return False,'missing_observation',[]
    return True,'verified',times[:3]


def scan_verified(video_path, catalog, vlm=None, cfg=None):
    cfg=cfg or settings()
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>cfg['seconds']:
            raise TimeoutError('Verification budget exceeded; incomplete analysis is not no_violation')
    info=video_info(video_path)
    frames=sample_frames(video_path,fps=cfg['fps'],max_width=cfg['width'])
    if not frames: raise ValueError('Video contains no readable frames')
    frame_seconds=time.monotonic()-started
    # A warm service may already hold Qwen on GPU. Keep detector inference sequential.
    from . import vlm as vlm_module
    resident = vlm or getattr(vlm_module, '_instance', None)
    parked = False
    detector_cfg = dict(cfg)
    if resident is not None and cfg['detectors'] != 'off' and cfg['det_device'] == 'cuda':
        model = getattr(resident, 'model', None)
        if model is not None:
            if getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False):
                detector_cfg['det_device'] = 'cpu'
                print('[verified] Quantized resident VLM: detector pass uses CPU')
            else:
                model.to('cpu'); parked = True
                import torch
                torch.cuda.empty_cache()
    try:
        rows,names=collect_detections(frames,detector_cfg,guard,video_path)
    finally:
        if parked: resident.model.to('cuda')
    print(f'[verified] detectors used: {names or "none"}; verification threshold={cfg["threshold"]}')
    vlm=vlm or get_vlm()
    windows=sliding_windows(frames,cfg['window'],cfg['step'])
    candidates=[]; observations=[]; memory=''
    catalog_ids={v['id'] for v in catalog['violations']}
    for wi,window in enumerate(windows):
        guard()
        picks=evenly(window,cfg['frames'])
        a,b=window[0].t,window[-1].t
        def valid_candidates(obj):
            cs = obj.get('candidates')
            return (isinstance(obj.get('summary'), str) and isinstance(cs, list) and
                    all(isinstance(c, dict) and c.get('type') in catalog_ids and
                        c.get('type') in CHECKS and type(c.get('frame')) is int and
                        0 <= c['frame'] < len(picks) for c in cs))
        obj=checked_generate(vlm,picks,candidate_prompt(catalog,detector_text(rows,a,b),memory),
                             validator=valid_candidates)
        if not isinstance(obj.get('candidates'),list) or not isinstance(obj.get('summary'),str):
            raise ValueError('Malformed observation response; analysis incomplete')
        memory=obj['summary'][:500]
        observations.append({'start':a,'end':b,'summary':memory,'proposals':obj['candidates']})
        proposed={}
        for c in obj['candidates']:
            if not isinstance(c,dict): raise ValueError('Invalid candidate object')
            kind=c.get('type'); index=c.get('frame')
            if kind in catalog_ids and kind in CHECKS and type(index) is int and 0<=index<len(picks):
                proposed.setdefault(kind,picks[index].t)
            else:
                raise ValueError('Invalid candidate type/frame index; analysis incomplete')
        # Independent detector proposals route inspection even when the first VLM misses the object.
        for kind,t in detector_candidates(rows,a,b,cfg['track_high']).items():
            if kind in catalog_ids: proposed.setdefault(kind,t)
        for kind,t in proposed.items():
            candidates.append({'type':kind,'start':a,'end':b,'t':t,'window_id':wi})
    # Remove duplicate overlapping proposals of the SAME type only. Preserve separate types/events.
    selected=[]
    for c in sorted(candidates,key=lambda x:(x['type'],x['start'])):
        match=next((q for q in reversed(selected) if q['type']==c['type'] and
                    c['start']<=q['end'] and c['end']-q['start']<=2*cfg['window']),None)
        if match:
            match['end']=max(match['end'],c['end']); match['t']=c['t']
        else: selected.append(dict(c))
    if len(selected)>cfg['max_checks']:
        raise RuntimeError(f"{len(selected)} candidate checks exceed VERIFY_MAX_CHECKS; analysis incomplete")
    decisions=[]
    for c in sorted(selected,key=lambda x:x['start']):
        guard()
        a=max(frames[0].t,c['start']-cfg['context']); b=min(frames[-1].t,c['end']+2)
        picks,metadata=inspection_frames(frames,a,b,c['t'],rows,c['type'],cfg['frames'],video_path,cfg)
        entry=catalog['by_id'][c['type']]
        objects={o for o,types in OBJECT_TYPES.items() if c['type'] in types}
        prompt=verification_prompt(c['type'],entry,detector_text(rows,a,b,objects),metadata)
        prompt += (f'\nThe reported event must occur between {c["start"]:.2f} and {b:.2f} seconds. '
                   'Earlier images are context only. evidence_frames must identify images in this interval. '
                   'For each condition use at most 3 image indices; observation under 25 words.')
        def valid_verification(value):
            checks=value.get('checks')
            return (isinstance(checks,dict) and all(
                isinstance(checks.get(k),dict) and checks[k].get('state') in ('yes','no','unknown') and
                isinstance(checks[k].get('frames'),list) and
                all(type(i) is int and 0<=i<len(metadata) for i in checks[k]['frames'])
                for k in CHECKS[c['type']]) and isinstance(value.get('evidence_frames'),list) and
                all(type(i) is int and 0<=i<len(metadata) for i in value['evidence_frames']) and
                isinstance(value.get('observation'),str))
        obj=checked_generate(vlm,picks,prompt,900,validator=valid_verification)
        ok,reason,times=validate_verification(obj,c['type'],metadata,c['start'],b)
        score=0.0
        probs=None
        if ok:
            guard()
            question=(GLOBAL_CONTEXT+'\nEvent: '+entry['definition']+'\nRequired visible conditions:\n'+
                      '\n'.join(CHECKS[c['type']].values())+
                      '\nImages include chronological full frames followed by detail crops of their timestamps. '
                      'A. All conditions are visibly supported in this sequence. '
                      'B. Conditions are contradicted or behavior is normal. '
                      'C. Evidence is missing or unclear. Answer with one letter only.')
            probs=vlm.choice_probs([(SYSTEM,picks,question)],3)[0]
            if len(probs)!=3 or any(not math.isfinite(float(p)) for p in probs):
                raise ValueError('Invalid verifier scores')
            score=float(probs[0])
        accepted=ok and score>=cfg['threshold']
        decisions.append({**c,'t':times[0] if times else c['t'], 'confidence':score,
                          'p_types':{c['type']:1.0},
                          'observation':obj.get('observation',''), 'verified':ok,
                          'accepted':accepted,'reason':reason if accepted or not ok else 'below_threshold',
                          'checks':obj.get('checks',{}),'evidence_times':times,
                          'judge_probs':probs,'score_evaluated':probs is not None,
                          'verification_views':metadata,'engine':VERSION})
    from .analyze import merge_events
    events=merge_events(decisions,catalog,min_conf=cfg['threshold'],min_windows=1)
    guard()
    return {'engine':VERSION,'config_hash':fingerprint(catalog,cfg),'settings':cfg,
            'video':Path(video_path).name,'duration':info['duration'],'n_windows':len(windows),
            'sec_frames':round(frame_seconds,1),'sec_total':round(time.monotonic()-started,1),
            'windows':decisions,'events':events,'features':[],'frames':frames,'catalog':catalog,
            'observations':observations,'detector_proposals':rows,'detectors_used':names,'complete':True}
