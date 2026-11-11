"""Small timestamp-aware accumulator. A weak candidate is never an automatic verdict."""
from __future__ import annotations
import numpy as np


def accumulate(times, scores, threshold=.15, window=5, votes=3, max_gap=None):
    """Return persistent episodes and isolated/brief candidates separately.

    A qualifying window has at least `votes` actual positive observations among
    at most `window` consecutive observations. No zero-padding or interpolation
    invents observations. Large timestamp gaps split the accumulator. Onset is
    the FIRST positive observation belonging to a qualifying window, not the
    third vote, center, or strongest frame.
    """
    t, s = np.asarray(times, float), np.asarray(scores, float)
    if t.ndim != 1 or s.shape != t.shape or not np.isfinite(t).all() or not np.isfinite(s).all():
        raise ValueError('Invalid temporal observations')
    if len(t)>1 and np.any(np.diff(t)<=0):
        raise ValueError('Temporal timestamps must strictly increase')
    if not 1<=votes<=window or not 0<threshold<=1 or np.any((s<0)|(s>1)):
        raise ValueError('Invalid temporal rule')
    if not len(t):return {'events':[], 'brief':[]}
    gap = max_gap if max_gap is not None else (1.75*float(np.median(np.diff(t))) if len(t)>1 else 1.75)
    if gap<=0 or not np.isfinite(gap):raise ValueError('Invalid maximum observation gap')
    positive = s>=threshold
    linked=[];segment=0
    for i in range(len(t)):
        if i and t[i]-t[i-1]>gap:segment=i
        a=max(segment,i-window+1)
        indices=np.flatnonzero(positive[a:i+1])+a
        if len(indices)>=votes:
            start,end=int(indices[0]),int(indices[-1])
            # Overlapping qualifying windows form one episode. A long normal
            # gap cannot join two incidents of the same type.
            if linked and linked[-1]['last_window']>=a and t[start]-t[linked[-1]['end']]<=gap*window:
                linked[-1]['end']=end;linked[-1]['last_window']=i
                linked[-1]['support'].update(map(int,indices))
                linked[-1]['windows'].append((a,i))
            else:
                linked.append({'start':start,'end':end,'last_window':i,
                               'support':set(map(int,indices)), 'windows':[(a,i)]})
    used=set();events=[]
    for group in linked:
        support=sorted(group['support']);used.update(support)
        # Mean of the strongest `votes` observations in a qualifying window:
        # two weak/occluded frames do not erase three consistent observations.
        value=max(float(np.mean(np.sort(s[a:b+1])[-votes:])) for a,b in group['windows'])
        peak=max(support,key=lambda i:float(s[i]))
        events.append({'i0':support[0],'i1':support[-1],'peak_i':peak,
                       'start':float(t[support[0]]),'end':float(t[support[-1]]),
                       'score':value,'support':support,'support_count':len(support),
                       'votes_required':votes,'window_size':window,'persistent':True})
    brief=[{'i0':int(i),'i1':int(i),'peak_i':int(i),'start':float(t[i]),'end':float(t[i]),
            'score':float(s[i]),'support':[int(i)],'support_count':1,'persistent':False}
           for i in np.flatnonzero(positive) if int(i) not in used]
    return {'events':events,'brief':brief}
