"""Independent low-threshold proposals -> enlarged detail+context -> visual states.

Descriptions are STRUCTURED alternatives, not keyword matches in generated text.
Probabilities are read from answer-token logits, balanced across three letter
orders. Python combines observed conditions and performs 3-of-5 accumulation.
This branch can ADD evidence and can never veto the full-frame branches.
"""
from __future__ import annotations
import math
import os
import numpy as np
from PIL import Image, ImageDraw
from .frames import Frame
from .temporal import accumulate

VERSION='focus-states-v1'
FULL_MIN=float(os.environ.get('FOCUS_OWL_MIN','0.03'))
TILE_MIN=float(os.environ.get('FOCUS_TILE_MIN','0.05'))
CANDIDATE_P=float(os.environ.get('FOCUS_CANDIDATE_P','0.15'))
STATE_P=float(os.environ.get('FOCUS_STATE_P','0.50'))
MAX_WINDOWS=int(os.environ.get('FOCUS_MAX_WINDOWS','6'))
BATCH=int(os.environ.get('FOCUS_BATCH','4'))
BRIEF_WEIGHT=float(os.environ.get('FOCUS_BRIEF_WEIGHT','0.5'))
if not (0<FULL_MIN<=1 and 0<TILE_MIN<=1 and 0<CANDIDATE_P<STATE_P<=1
        and 1<=MAX_WINDOWS<=64 and 1<=BATCH<=16 and 0<BRIEF_WEIGHT<=1):
    raise ValueError('Invalid FOCUS settings')
# Keep the cached OWL extraction floors unchanged: full>=.03, tile>=.05.
# This bypasses Scene's .12+background tracking gate without rerunning OWL.
# Increasing these admission floors only filters already stored candidates.
if FULL_MIN<.03 or TILE_MIN<.05:
    raise ValueError('FOCUS admission must match stored OWL coverage: full >=.03, tile >=.05. '
                     'Lower extraction floors require re-extraction and a new calibrated pipeline.')

SYSTEM=('Describe only visible objects, positions and states in a silent bank cash-room image. '
        'The left panel shows context, the right panel is an enlarged crop of an uncertain proposal. '
        'A red rectangle is a proposed location, not proof that an object is there. '
        'You do not decide whether a legal or procedural violation occurred. '
        'Choose visually unresolved when the required detail cannot be determined. '
        'Do not identify faces or people, infer bank-system entries, money amounts or invisible checks.')

# Each tuple is: description when present, description when absent/normal, unresolved.
CONDITIONS={
 'cash_exposed':('Banknotes or a cash stack are lying exposed on the desk surface',
                 'No exposed banknotes are on the desk; the object is paper/equipment or cash is held',
                 'It is not possible to determine whether exposed cash is on the surface'),
 'away_desk':('The cash employee is away from the desk and not attending the exposed desk items',
              'The cash employee is at the desk or visibly handling/attending the desk items',
              'The employee position or attendance cannot be determined'),
 'key_in_lock':('A key is visibly inserted in the safe lock',
                'The safe lock has no inserted key, or the visible object is not a key in the lock',
                'The key and lock are too small/occluded to determine the state'),
 'safe_open':('The safe door is visibly open', 'The safe door is visibly closed',
               'The safe or its door state cannot be determined'),
 'away_safe':('The cash employee is away from the safe and not visibly attending it',
               'The cash employee is beside the safe and actively using or attending it',
               'Attendance at the safe cannot be determined'),
 'personal_storage':('A personal handbag, backpack or clothing is stored in the cash work area',
                     'Only normal work equipment, briefly held personal items or a cash-collection bag are visible',
                     'The item cannot be distinguished from work equipment or a collection bag'),
 'drink_over_cash':('A drink cup is held or placed directly over exposed banknotes',
                    'There is no drink directly over banknotes; a cup elsewhere or a lookalike is present',
                    'The relative position of a drink and banknotes cannot be determined'),
 'visitor_staff_side':('A visitor has crossed onto the employee side of the counter',
                       'People remain on their expected sides or are visibly handling a normal collector visit',
                       'Roles and positions relative to the counter cannot be determined'),
 'door_open':('The cash-room service door is visibly open', 'The service door is visibly closed',
               'The service door state cannot be determined'),
 'work_no_passage':('Work continues inside and nobody is currently passing through the service doorway',
                     'Someone is passing through the doorway or opening/closing the workplace',
                     'Work activity or doorway passage cannot be determined'),
 'bag_visible':('A sealed cash-collection bag is visibly left in the work area',
                 'No cash-collection bag is visible, or a bag is being actively handled',
                 'The bag cannot be distinguished or its position cannot be determined'),
 'counting_cash':('The cash employee is visibly counting banknotes',
                   'The employee is doing other work, carrying cash, or not handling banknotes',
                   'Counting cannot be distinguished from other hand motion'),
 'customer_absent':('No customer is visible at the customer side of the desk',
                    'A customer is visibly present at the customer side of the desk',
                    'Customer presence cannot be determined from this view'),
 'bank_session_unlocked':('Readable display content shows an active banking work session with no lock screen',
                          'A lock screen, switched-off display, closed laptop or no active banking session is visible',
                          'Display content is unreadable; illumination alone cannot establish an unlocked banking session'),
}
RULES={
 'cash_unattended':{'group':'cash','objects':('cash',),'facts':('cash_on_desk',),
                    'conditions':('cash_exposed','away_desk'), 'ru':'наличные на поверхности стола; кассир не у стола'},
 'safe_key_unattended':{'group':'safe','objects':('key','safe'),'facts':('key_in_safe_lock',),
                       'conditions':('key_in_lock','away_safe'),'ru':'ключ в замке сейфа; кассир не у сейфа'},
 'safe_left_open':{'group':'safe','objects':('safe',),'facts':('safe_open',),
                   'conditions':('safe_open','away_safe'),'ru':'дверца сейфа открыта; кассир не у сейфа'},
 'personal_items_cash_area':{'group':'personal','objects':('handbag','backpack','bag'),
                             'facts':('personal_item_on_desk',),'conditions':('personal_storage',),
                             'ru':'личная сумка или одежда находится в кассовой рабочей зоне'},
 'coffee_over_cash':{'group':'drink','objects':('cup',),'facts':('drink_over_cash',),
                     'conditions':('drink_over_cash',),'ru':'напиток расположен над банкнотами'},
 'unauthorized_cash_area_entry':{'group':'staff','objects':('person',),'facts':('outsider_behind_counter',),
                                 'conditions':('visitor_staff_side',),'ru':'посетитель находится на стороне кассира'},
 'service_door_left_open':{'group':'door','objects':('door',),'facts':('service_door_open',),
                          'conditions':('door_open','work_no_passage'),
                          'ru':'служебная дверь открыта во время работы, проход через дверь не наблюдается'},
 'collection_bag_unsecured':{'group':'bag','objects':('bag',),'facts':('collection_bag_visible',),
                            'conditions':('bag_visible','away_desk'),
                            'ru':'инкассаторская сумка находится в рабочей зоне; кассира рядом нет'},
 'counting_without_customer':{'group':'counting','objects':('cash',),'facts':('cash_in_hands',),
                              'conditions':('counting_cash','customer_absent'),
                              'history':'client_departed','ru':'после наблюдавшегося ухода клиента продолжается пересчёт'},
 'unlocked_bank_terminal':{'group':'terminal','objects':('laptop',),'facts':('computer_screen_on',),
                           'conditions':('bank_session_unlocked','away_desk'),
                           'ru':'видна незаблокированная банковская сессия; кассира у рабочего места нет'},
}

def signature():
    return {'version':VERSION,'full_min':FULL_MIN,'tile_min':TILE_MIN,'candidate_p':CANDIDATE_P,
            'state_p':STATE_P,'max_windows':MAX_WINDOWS,'brief_weight':BRIEF_WEIGHT,
            'window':5,'votes':3,'answer_orders':3}


def _box(row,size):
    if not isinstance(row,(list,tuple)) or len(row)<5:return None
    a=np.asarray(row[:5],float)
    if not np.isfinite(a).all() or not 0<=a[0]<=1:return None
    w,h=size;x0,y0,x1,y1=a[1:]
    b=[max(0.,min(float(w),x0)),max(0.,min(float(h),y0)),
       max(0.,min(float(w),x1)),max(0.,min(float(h),y1))]
    return b if b[2]-b[0]>=2 and b[3]-b[1]>=2 else None


def low_proposals(det,objects,index):
    """Bypass high/low Scene tracks; do not promote a weak box to a visual fact."""
    out=[];size=det.get('size',[0,0])
    for stage,floor in (('dets',FULL_MIN),('tiles',TILE_MIN)):
        for detector,per in det.get(stage,{}).items():
            if index>=len(per) or not per[index]:continue
            for obj in objects:
                for row in per[index].get(obj,[]):
                    b=_box(row,size)
                    if b is not None and float(row[0])>=floor:
                        out.append({'object':obj,'score':float(row[0]),'box':b,'source':detector+'/'+stage})
    return sorted(out,key=lambda r:-r['score'])


def collect(st,scene,det,flow_rec=None):
    """Rank candidate windows with per-group temporal suppression and diversity.

    At least one full-frame survey is independent of OWL. The existing whole-flow
    branch continues to inspect ALL video intervals, including deferred proposals.
    """
    groups={}
    for tid,r in RULES.items():groups.setdefault(r['group'],[]).append(tid)
    proposals=[]
    for group,tids in groups.items():
        objects=tuple(sorted({o for tid in tids for o in RULES[tid]['objects']}))
        qkeys=tuple(sorted({k for tid in tids for k in RULES[tid]['facts']}))
        seeds=[]
        for i in range(st.n):
            rows=low_proposals(det,objects,i)
            q=max((float(st.q[k][i]) for k in qkeys),default=0.)
            # An illuminated terminal by itself is an especially weak proposal.
            factor={'safe':1.2,'cash':1.1,'bag':1.1,'personal':.8,
                    'door':.8,'drink':.8,'staff':.5,'terminal':.4,'counting':.6}.get(group,1.)
            if rows or q>=CANDIDATE_P:
                object_rank=min(1.,(rows[0]['score'] if rows else 0.)/.40)
                if group=='staff':object_rank*=.15  # a person alone is normal, not an intrusion
                rank=factor*max(q,object_rank)
                seeds.append({'group':group,'rules':tids,'i':i,'rank':rank,
                              'box':rows[0]['box'] if rows else None,
                              'proposal_source':rows[0]['source'] if rows else 'full-frame facts'})
        selected=[]
        for seed in sorted(seeds,key=lambda x:(-x['rank'],x['i'])):
            if all(abs(st.t[seed['i']]-st.t[s['i']])>=5 for s in selected):selected.append(seed)
        proposals.extend(selected)
    # Independent focus survey at the strongest existing flow interval (or middle).
    t=float((flow_rec or {}).get('features',{}).get('v_judge',{}).get('t',st.t[st.n//2]))
    survey={'group':'survey','rules':list(RULES),'i':int(np.argmin(abs(st.t-t))),
            'rank':1.,'box':None,'proposal_source':'independent full-frame survey'}
    chosen=[survey];remaining=sorted(proposals,key=lambda x:(-x['rank'],x['group'],x['i']))
    seen=set()
    # Tiny key/safe/cash details particularly need enlarged views. Reserve their
    # best candidate before filling the remaining budget by rank and diversity.
    for group in ('safe','cash'):
        seed=next((s for s in remaining if s['group']==group),None)
        if seed is not None and len(chosen)<MAX_WINDOWS:
            chosen.append(seed);seen.add((seed['group'],seed['i']))
    # First cover different object groups; then inspect additional times.
    for diversity in (True,False):
        for seed in remaining:
            if len(chosen)>=MAX_WINDOWS:break
            key=(seed['group'],seed['i'])
            if key in seen or (diversity and any(x['group']==seed['group'] for x in chosen)):continue
            chosen.append(seed);seen.add(key)
    return chosen,{'proposed':len(proposals)+1,'checked':len(chosen),
                   'deferred':max(0,len(proposals)+1-len(chosen)),
                   'fallback':'full-frame whole-flow branch covers the entire timeline'}


def detail_image(image,box):
    """Native pixels enlarged for visibility, with full-frame actor context beside them."""
    if box is None:return image.copy()
    w,h=image.size
    a=np.array(box,float)
    if a.shape!=(4,) or not np.isfinite(a).all():return image.copy()
    a=np.array([max(0.,min(float(w),a[0])),max(0.,min(float(h),a[1])),
                max(0.,min(float(w),a[2])),max(0.,min(float(h),a[3]))])
    if a[2]-a[0]<2 or a[3]-a[1]<2:return image.copy()
    x0,y0,x1,y1=a;pad=max(16.,.4*max(x1-x0,y1-y0))
    crop_box=(max(0,int(x0-pad)),max(0,int(y0-pad)),min(w,int(x1+pad)),min(h,int(y1+pad)))
    if crop_box[2]-crop_box[0]<2 or crop_box[3]-crop_box[1]<2:return image.copy()
    context=image.copy();ImageDraw.Draw(context).rectangle([int(x) for x in a],outline=(255,40,40),width=2)
    scale=min(1.,464/max(1,w));context=context.resize((max(1,int(w*scale)),max(1,int(h*scale))))
    detail=image.crop(crop_box);ratio=448/max(detail.size)
    detail=detail.resize((max(1,int(detail.width*ratio)),max(1,int(detail.height*ratio))),Image.Resampling.LANCZOS)
    out=Image.new('RGB',(context.width+detail.width+12,max(context.height,detail.height)+22),(235,235,235))
    out.paste(context,(0,22));out.paste(detail,(context.width+12,22))
    draw=ImageDraw.Draw(out);draw.text((4,4),'CONTEXT',fill='black');draw.text((context.width+16,4),'DETAIL',fill='black')
    return out


def state_probabilities(vlm,frames,conditions):
    """Read relative token probabilities; every state visits every answer letter."""
    orders=((0,1,2),(1,2,0),(2,0,1));questions=[]
    for condition in conditions:
        options=CONDITIONS[condition]
        for order in orders:
            questions.append('Which description matches the visible state in THIS image?\n'+
                             '\n'.join(f'{chr(65+j)}. {options[o]}' for j,o in enumerate(order))+
                             '\nAnswer with the letter only.')
    raw=[]
    for a in range(0,len(frames),BATCH):
        raw.extend(vlm.frame_question_probs(SYSTEM,frames[a:a+BATCH],questions,3))
    p=np.asarray(raw,float)
    if p.shape!=(len(frames),len(questions),3) or not np.isfinite(p).all() or np.any(p<0) or \
            np.any(abs(p.sum(axis=-1)-1)>1e-3):
        raise ValueError('Incomplete or invalid visual-state probabilities')
    result={}
    for k,condition in enumerate(conditions):
        value=np.zeros((len(frames),3))
        for j,order in enumerate(orders):
            for letter,state in enumerate(order):value[:,state]+=p[:,3*k+j,letter]/3
        result[condition]=value
    return result


def _history(st,indices):
    # A person working alone is normal unless an earlier customer departure was
    # actually observed. This is a proposal condition, not proof of ownership.
    values=[]
    for i in indices:
        earlier=st.client[max(0,i-st.k(30)):i]
        recent=st.client[max(0,i-st.k(4)):i+1]
        values.append(1. if np.count_nonzero(earlier>=.5)>=3 and len(recent) and
                      np.min(recent)<.3 and st.client[i]<.3 else 0.)
    return np.array(values)


def run(st,scene,det,frames,vlm,flow_rec=None):
    if not st.n or len(frames)!=st.n:raise ValueError('No aligned frames for focus analysis')
    candidates,coverage=collect(st,scene,det,flow_rec)
    traces={};condition_records=[];n_questions=0
    for candidate in candidates:
        i=candidate['i'];indices=list(range(max(0,i-2),min(st.n,i+3)))
        conditions=sorted({c for tid in candidate['rules'] for c in RULES[tid]['conditions']})
        images=[];boxes={}
        for j in indices:
            box=candidate['box']
            if candidate['group'] in ('safe','door','terminal'):
                roi={'safe':'safe','door':'door','terminal':'laptop'}[candidate['group']]
                box=(scene.rois.get(roi) or {}).get('box',box)
            boxes[j]=box
            images.append(Frame(frames[j].t,detail_image(frames[j].image,box)))
        probs=state_probabilities(vlm,images,conditions);n_questions+=3*len(conditions)*len(images)
        condition_records.append({'group':candidate['group'],'indices':indices,'box':candidate['box'],
                                  'proposal_source':candidate['proposal_source'],
                                  'conditions':{k:v.tolist() for k,v in probs.items()}})
        for tid in candidate['rules']:
            rule=RULES[tid]
            vals=np.array([probs[c][:,0] for c in rule['conditions']])
            scores=np.exp(np.mean(np.log(np.maximum(vals,1e-9)),axis=0))
            strong=np.all(vals>=STATE_P,axis=0)
            if rule.get('history'):
                hist=_history(st,indices);scores*=hist;strong &= hist.astype(bool)
            tr=traces.setdefault(tid,{'scores':{},'strong':{},'boxes':{}})
            for j,s,ok in zip(indices,scores,strong):
                # Keep multiple views of the SAME timestamp, without inventing
                # extra temporal votes. A later blurry view must not erase a
                # useful enlarged view of that timestamp.
                tr['scores'].setdefault(j,[]).append(float(s))
                tr['strong'].setdefault(j,[]).append(float(ok))
                tr['boxes'].setdefault(j,boxes[j])
    features={};all_events=[];brief=[]
    for tid,tr in traces.items():
        indices=sorted(tr['scores']);times=st.t[indices]
        scores=np.array([max(tr['scores'][j]) for j in indices])
        strong=np.array([any(tr['strong'][j]) for j in indices],float)
        grouped=accumulate(times,scores,CANDIDATE_P,max_gap=1.75*st.dt)
        labels=accumulate(times,strong,STATE_P,max_gap=1.75*st.dt)['events']
        records=[]
        # Strong multi-label state votes can establish several distinct episodes.
        # Weak support that bridges them must not collapse two incidents into one.
        candidates_for_output=[{**e,'typed':True} for e in labels]
        for e in grouped['events']+grouped['brief']:
            if not any(x['start']<=e['end'] and x['end']>=e['start'] for x in labels):
                candidates_for_output.append({**e,'typed':False})
        for event in candidates_for_output:
            a,b=event['start'],event['end']
            local=[j for j in event['support']]
            peak_local=max(local,key=lambda j:float(scores[j]));peak=indices[peak_local]
            is_typed=event['typed']
            score=float(np.mean(sorted(float(scores[j]) for j in local)[-3:]))
            if not event['persistent']:score*=BRIEF_WEIGHT
            why=(f"На фрагменте {a:.1f}–{b:.1f} с есть визуальные признаки: {RULES[tid]['ru']}. "
                 + ('Состояния повторяются минимум в 3 из 5 наблюдений. ' if is_typed else
                    'Уверенного временного подтверждения всех условий нет. ')
                 + 'Это основание для проверки, а не подтверждение юридического нарушения.')
            rec={'type':tid,'score':score,'t':float(a),'start':float(a),'end':float(b),
                 'peak_t':float(st.t[peak]),'box':tr['boxes'].get(peak),'box_timecode':float(st.t[peak]),
                 'evidence_times':sorted({float(a),float(st.t[peak]),float(min(st.t[-1],b+st.dt))}),
                 'generic':not is_typed,'type_supported':is_typed,'persistent':event['persistent'],
                 'support_count':event['support_count'],'why':why}
            records.append(rec)
            (all_events if event['persistent'] else brief).append(rec)
        typed_records=[r for r in records if r['type_supported']]
        if typed_records:
            best=max(typed_records,key=lambda r:r['score'])
            features['f_'+tid]={**best,'episodes':typed_records}
    combined=all_events+brief
    if combined:
        best=max(combined,key=lambda r:r['score'])
        features['f_any']={**best,'type':'other','generic':True,'type_supported':False,
                           'episodes':[{**e,'candidate_type':e['type'],'type':'other','generic':True,'type_supported':False} for e in combined]}
    else:
        # An explicit zero indicates a completed branch with no candidate support.
        # It is different from a missing/failed branch and can never fire at tau=0.
        features['f_any']={'type':'other','score':0.,'t':float(st.t[0]),'generic':True,
                           'type_supported':False,'episodes':[]}
    return {'version':VERSION,'signature':signature(),'features':features,'coverage':coverage,
            'events':all_events,'brief':brief,'observations':condition_records,'n_questions':n_questions}
