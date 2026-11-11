# Repository audit (before the verified engine)

| Question from the brief | Finding |
|---|---|
| Files named in the brief (`scripts/run_vlm.py`, `pipeline/fusion.py`, `pipeline/facts.py`, `pipeline/rules.py`) | Not present in this project version. The latest working version was `forte.py` + `pipeline/flow.py` (Qwen2.5-VL-7B + OWLv2). `scripts/run_vlm.py` with `dataset --engine verified` was created. |
| What works | Legacy engine end to end: OWLv2 (+SAHI tiles), Qwen per-frame facts, state engine, judge, service API. Measured by the team (leave-one-video-out, 40 unique labelled videos): **72 %, 11/21 violations found, 0/19 false alarms**. |
| Obsolete / not useful | Condition chains (0/21), pixel change (2/21 found, 2 false alarms), whole-video Qwen classifier (0/21): measured, switched off. |
| Decision logic | `pipeline/state_engine.py` (per-type evidence) + `pipeline/judge.py` (threshold = highest clean value + 0.05). |
| Rules depending on Qwen / OWLv2 | All of them (facts from Qwen, objects from OWLv2). The verified engine imports none of these modules (tested). |
| Scoring vs documentation | Legacy thresholds minimise false alarms; the bank's priority is the opposite ("a miss is unacceptable"). The verified engine maximises F2 under a false-alarm cap. |
| Boxes and timestamps preserved | Legacy: only the top box per class reached the state engine. Verified: every box above `min_score`, frame index and timestamp kept, in original-frame coordinates. |
| Data splits / leakage | No session or camera ids in `labels.csv`; X-47 = X-4 and X-48 = X-5 are identical files and are merged. Evaluation is leave-one-video-out; thresholds are fitted without the judged video (tested). Rules and prompts were designed by looking at all 40 videos, so the estimate is optimistic. |
| Output schema | `app/main.py`: `{"violations": [{type, timecode_sec, description, evidence_frames[{timestamp_sec, image_base64}], rule}]}`. Preserved; extra fields added. |
| `--engine verified` | Did not exist; implemented. `--engine legacy` points to `forte.py` (baseline A in the report). |
| Tests | There were none for the event logic. 42 CPU tests added (`tests/`). |


