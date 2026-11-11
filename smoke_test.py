"""
Does exactly what the jury does: upload a .mov, poll until done, validate the JSON schema.

    python smoke_test.py --url http://localhost:8000 --key YOUR_KEY --video some.mov
    python smoke_test.py --url https://your-public-url --key YOUR_KEY --video some.mov

Standard library only, so it runs anywhere (including Windows).
"""
import argparse
import base64
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

TIMEOUT_SEC = 300  # jury limit: 5 minutes


def request(method, url, key, body=None, ctype=None):
    req = urllib.request.Request(url, data=body, method=method)
    if key is not None:
        req.add_header("X-API-Key", key)
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def upload(url, key, path):
    boundary = uuid.uuid4().hex
    data = Path(path).read_bytes()
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{Path(path).name}"\r\n'
        f"Content-Type: video/quicktime\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return request("POST", f"{url}/api/v1/analyze", key, body, f"multipart/form-data; boundary={boundary}")


def check_schema(res):
    errors = []
    if res.get("verdict") not in ("violation", "no_violation"):
        errors.append(f"verdict must be violation|no_violation, got {res.get('verdict')!r}")
    v = res.get("violations")
    if not isinstance(v, list):
        return errors + ["violations must be a list"]
    if (res.get("verdict") == "violation") != bool(v):
        errors.append("verdict and violations list disagree")
    for i, x in enumerate(v):
        for f, t in (("type", str), ("timecode_sec", (int, float)), ("description", str), ("evidence_frames", list)):
            if not isinstance(x.get(f), t):
                errors.append(f"violations[{i}].{f} missing or wrong type")
        frames = x.get("evidence_frames") or []
        if not 1 <= len(frames) <= 3:
            errors.append(f"violations[{i}] needs 1-3 evidence_frames, has {len(frames)}")
        for j, fr in enumerate(frames):
            try:
                raw = base64.b64decode(fr["image_base64"], validate=True)
                if raw[:2] != b"\xff\xd8":
                    errors.append(f"violations[{i}].evidence_frames[{j}] is not a JPEG")
            except Exception:  # noqa: BLE001
                errors.append(f"violations[{i}].evidence_frames[{j}] bad base64")
            if not isinstance(fr.get("timestamp_sec"), (int, float)):
                errors.append(f"violations[{i}].evidence_frames[{j}].timestamp_sec missing")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--video", required=True)
    a = ap.parse_args()
    url = a.url.rstrip("/")

    code, _ = request("GET", f"{url}/api/v1/jobs/x", "wrong-key")
    print(f"[{'PASS' if code == 401 else 'FAIL'}] wrong key rejected (HTTP {code})")

    t0 = time.time()
    code, r = upload(url, a.key, a.video)
    if code != 200 or "job_id" not in r:
        print(f"[FAIL] upload: HTTP {code} {r}")
        return
    print(f"[PASS] upload accepted, job_id={r['job_id']}")

    while time.time() - t0 < TIMEOUT_SEC:
        code, res = request("GET", f"{url}/api/v1/jobs/{r['job_id']}", a.key)
        if res.get("status") != "processing":
            break
        time.sleep(3)
    elapsed = time.time() - t0

    if res.get("status") != "done":
        print(f"[FAIL] status={res.get('status')} after {elapsed:.0f}s: {res.get('error', '')}")
        return
    print(f"[{'PASS' if elapsed < TIMEOUT_SEC else 'FAIL'}] finished in {elapsed:.1f}s (limit {TIMEOUT_SEC}s)")

    errors = check_schema(res)
    print("[PASS] JSON schema OK" if not errors else "[FAIL] schema:\n  " + "\n  ".join(errors))
    short = {**res, "violations": [{**x, "evidence_frames": [
        {"timestamp_sec": f["timestamp_sec"], "image_base64": f["image_base64"][:20] + "..."}
        for f in x.get("evidence_frames", [])]} for x in res.get("violations", [])]}
    print(json.dumps(short, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
