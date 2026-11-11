# Verified engine — architecture and data flow

```
video (.mov)
 │  pipeline/video.py        decode, metadata, real timestamps (frame index / container fps)
 ▼
coarse frames, 1 fps (native 464×848)
 │  pipeline/detections.py   Grounding DINO tiny, prompts in groups; top-k boxes per class (several
 │                           people / cash bundles / bags per frame); tiles 416 px for cash, key, cup, phone
 │                           every 2nd frame, boxes mapped back to the original frame, NMS per class
 ▼
frame detections [{i, t, dets:[{label, score, box, src}]}]          (cached: video id + config hash)
 │  pipeline/tracking.py     ByteTrack per class (Roboflow `trackers`, or built-in two-stage association)
 │  pipeline/zones.py        per video, no hand coordinates: safe / laptop / monitor / door / drawer regions,
 │                           cashier = longest-present person (ties: nearest to the work objects, else unknown),
 │                           cashier zone, desk zone, roles cashier / visitor / unknown
 ▼
temporal states  (pipeline/temporal.py, TRUE / FALSE / UNKNOWN per second)
   cashier_present, visitor_present, outsider_in_cashier_zone, <object>_visible (desk) / _anywhere
   missing box = FALSE only if the place is visible; covered by a person = UNKNOWN; gap tolerance 2 s,
   hysteresis 2 frames, a departure needs ≥ 3 s (brief occlusion is not a departure)
 │  pipeline/rules.py        sequences per violation → candidates {type, start, end, conditions, rule_score,
 │                           questions, real boxes}; a FALSE detector condition stops the rule
 ▼
candidates (+ fallback windows when the detector found no departure / visit at all)
 │  pipeline/sampler.py      adaptive sampling: highest rule scores first, 3 s before / after the transition at
 │                           2 fps, ≤ 6 frames per question; region close-ups for safe / door / screen;
 │                           budget of VLM passes; skipped questions are reported
 │  pipeline/smolvlm.py      SmolVLM2-2.2B: A/B/C answer-letter probabilities (Yes / No / Unclear), both
 │                           option orders averaged; loaded once; OOM → fewer frames
 │  pipeline/questions.py    VERA-style question bank (8 categories), attached only to matching candidates
 ▼
 │  pipeline/fusion.py       mandatory / support / contradiction questions → score and status
 │                           (confirmed_candidate / needs_review / rejected) + Russian explanation
 ▼
 │  pipeline/calibration.py  thresholds fitted on TRAINING videos: max F2 with FPR ≤ 20 %; per-type threshold
 │                           only with ≥ 2 positive videos; conformal p-values reported as descriptive
 ▼
verdict per video + all events (type, start, end, score, threshold, status, explanation)
 │  pipeline/evidence.py     1–3 real frames per event, only boxes the detector returned at that time
 ▼
JSONL (scripts/run_vlm.py dataset)   or   API (app/ → pipeline/analyze_verified.py, PIPELINE=verified)
```

Timing per video is recorded per stage (`decode`, `detection`, `tracking`, `temporal`, `rules`, `vlm`,
`fusion`, `total`) plus GPU peak memory. When 85 % of `decision.max_seconds` (280 s) is used, remaining VLM
questions are skipped, listed in `vlm_skipped`, and their conditions stay UNKNOWN (→ human review), never
silently "no violation".

Caches (`eval/cache_verified/det`, `/vlm`): keyed by video name + size + mtime, model path, and a hash of the
config sections the stage reads. The service runs without any cache.
