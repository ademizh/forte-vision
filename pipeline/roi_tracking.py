"""Native-resolution tiles and conservative temporal ROI association.

Inspired by sliced inference and high/low-score association, not an implementation
of SAHI or ByteTrack. Tracks are crop proposals, NEVER proof of object presence.
No fixed camera coordinates, pretrained extra models, or label lookups.
"""
from collections import OrderedDict, defaultdict
import math

import cv2
from PIL import Image

from .frames import Frame


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1])-intersection
    return intersection / union if union > 0 else 0.


def nms(rows, threshold=.5, top_k=5):
    kept = []
    counts = defaultdict(int)
    for row in sorted(rows, key=lambda r: r['score'], reverse=True):
        obj = row['object']
        if counts[obj] >= top_k:
            continue
        if any(r['object'] == obj and iou(row['box'], r['box']) >= threshold for r in kept):
            continue
        kept.append(dict(row)); counts[obj] += 1
    return kept


def tile_boxes(width, height, size=768, overlap=.25):
    """Cover the whole image, including right/bottom edges; native pixel units."""
    if size < 32 or not 0 <= overlap < 1:
        raise ValueError('Invalid tile size/overlap')
    def starts(length):
        if length <= size:
            return [0]
        step = max(1, round(size*(1-overlap)))
        return sorted(set(list(range(0, length-size+1, step)) + [length-size]))
    return [(x, y, min(width, x+size), min(height, y+size))
            for y in starts(height) for x in starts(width)]


class NativeReader:
    """Bounded native frame cache; no whole-video full-resolution allocation."""
    def __init__(self, path, cache_size=3):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(f'Cannot open original video: {path}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.cap.release()
            raise ValueError('Invalid video FPS')
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def get(self, t):
        idx = max(0, min(self.count-1, round(t*self.fps)))
        if idx not in self.cache:
            seek_ok = self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = self.cap.read() if seek_ok else (False, None)
            actual = int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))-1
            if not ok or actual != idx:
                # MOV backends may seek to a keyframe or fail near EOF. Decode
                # from frame zero instead of accepting an unrelated frame.
                self.cap.release()
                self.cap = cv2.VideoCapture(self.path)
                if not self.cap.isOpened():
                    raise ValueError(f'Cannot reopen original video: {self.path}')
                for actual in range(idx+1):
                    ok, bgr = self.cap.read()
                    if not ok:
                        raise ValueError(f'Original video cannot be decoded sequentially at frame {actual} (requested {idx})')
                print(f'[roi] recovered frame {idx} by sequential decoding')
            self.cache[idx] = Frame(round(actual/self.fps, 4), Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(idx)
        return self.cache[idx]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cap.release(); self.cache.clear()


def detect_native(detector, frame, cfg, tiled, guard):
    """Global pass plus native tiles. Return normalized ORIGINAL-frame boxes."""
    image = frame.image
    w, h = image.size
    boxes = [(0, 0, w, h)]
    if tiled:
        # Larger tiles on large inputs keep inference work bounded without holes.
        size = cfg['tile_size']
        while len(tile_boxes(w, h, size, cfg['tile_overlap'])) > cfg['max_tiles']:
            size = math.ceil(size*1.25)
        boxes += [b for b in tile_boxes(w, h, size, cfg['tile_overlap']) if b != boxes[0]]
    rows = []
    for start in range(0, len(boxes), 2):
        guard()
        batch = boxes[start:start+2]
        results = detector.detections([image.crop(b) for b in batch], cfg['det_floor'], cfg['det_topk'])
        if len(results) != len(batch):
            raise ValueError('Detector returned incomplete batch')
        for tile, found in zip(batch, results):
            for r in found:
                x0, y0, x1, y1 = r['box']
                box = [max(0., (x0+tile[0])/w), max(0., (y0+tile[1])/h),
                       min(1., (x1+tile[0])/w), min(1., (y1+tile[1])/h)]
                if all(math.isfinite(v) for v in box) and box[2] > box[0] and box[3] > box[1]:
                    rows.append({**r, 'box': box, 't': frame.t, 'detector': detector.name,
                                 'coordinate_space': 'normalized', 'native_size': [w, h],
                                 'tile': list(tile)})
    return nms(rows, .5, cfg['det_topk'])


def associate(rows, high=.25, min_iou=.15, max_gap=3.):
    """Per-detector, per-class, one-to-one high-then-low IoU association.

    Weak detections may continue an existing track but never start one. Unmatched
    weak boxes remain available as inspection proposals. No synthetic detections.
    Fast motion/camera cuts naturally break tracks; no cross-camera identities.
    """
    output = [dict(r) for r in rows]
    groups = defaultdict(list)
    for r in output:
        groups[(r['detector'], r['object'])].append(r)
    next_id = 0
    for group in groups.values():
        active = []
        by_time = defaultdict(list)
        for r in group:
            by_time[r['t']].append(r)
        for t, current in sorted(by_time.items()):
            active = [tr for tr in active if t-tr['last']['t'] <= max_gap]
            used = set()
            for strong in (True, False):
                batch = [r for r in current if (r['score'] >= high) == strong]
                pairs = sorted([(iou(tr['last']['box'], r['box']), k, j)
                                for k, tr in enumerate(active) if k not in used
                                for j, r in enumerate(batch)], reverse=True)
                matched = set()
                for overlap, k, j in pairs:
                    if overlap < min_iou:
                        break
                    if k in used or j in matched:
                        continue
                    batch[j]['track_id'] = active[k]['id']
                    active[k]['last'] = batch[j]
                    used.add(k); matched.add(j)
                if strong:
                    for j, r in enumerate(batch):
                        if j not in matched:
                            next_id += 1
                            r['track_id'] = next_id
                            active.append({'id': next_id, 'last': r})
                            used.add(len(active)-1)
    return output


def crop_sequence(reader, frames, rows, objects, a, b, focus_t, cfg):
    """Same tracked ROI across ordered moments. Unobserved locations are labeled.

    At gaps, inspect the last/nearest location; this is NOT an assertion that the
    object persists. Full frames remain available for camera-motion/context checks.
    """
    relevant = [r for r in rows if a <= r['t'] <= b and r['object'] in objects]
    groups = defaultdict(list)
    for i, r in enumerate(relevant):
        groups[r.get('track_id', f'untracked_{i}')].append(r)
    ranked = sorted(groups.items(), key=lambda kv: (
        len({r['t'] for r in kv[1]}) >= 2, max(r['score'] for r in kv[1])), reverse=True)
    chosen = []
    for key, group in ranked:
        anchor = max(group, key=lambda r: r['score'])
        # Avoid spending both ROI slots on the same object detected by DINO/OWL.
        if any(anchor['object'] == old[2]['object'] and iou(anchor['box'], old[2]['box']) > .5 for old in chosen):
            continue
        chosen.append((key, group, anchor))
        if len(chosen) >= cfg['roi_tracks']:
            break
    available = [f for f in frames if a <= f.t <= b]
    if not available:
        return [], []
    targets = [a, max(a, min(b, focus_t)), b]
    times = sorted({min(available, key=lambda f: abs(f.t-t)).t for t in targets})
    # A sequence of the same automatically tiled region also works when no
    # detector proposal exists. Bounded fallback is broad, not high-confidence.
    if not chosen:
        for i, box in enumerate(((0,0,.6,.6),(.4,0,1,.6),(0,.4,.6,1),(.4,.4,1,1))):
            chosen.append((f'fallback_{i}', [], {'box': box, 'object': 'unknown'}))
        times = [times[0], times[-1]] if len(times) > 1 else times
    crops, meta = [], []
    for key, group, anchor in chosen:
        for t in times:
            native = reader.get(t)
            nearest = min(group, key=lambda r: abs(r['t']-native.t)) if group else anchor
            observed = bool(group) and abs(nearest['t']-native.t) < .06
            w, h = native.image.size
            x0, y0, x1, y1 = nearest['box']
            margin = 0 if not group else .6*max((x1-x0)*w, (y1-y0)*h)
            margin = max(margin, 24) if group else margin
            box = (max(0, int(x0*w-margin)), max(0, int(y0*h-margin)),
                   min(w, math.ceil(x1*w+margin)), min(h, math.ceil(y1*h+margin)))
            crops.append(Frame(native.t, native.image.crop(box)))
            meta.append({'t': native.t, 'view': 'native_crop', 'box': list(box),
                         'track_id': key, 'object_proposal': anchor['object'],
                         'location_status': 'detected' if observed else 'search_region_only',
                         'native_size': [w,h]})
    return crops, meta
