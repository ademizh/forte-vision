"""
Cash-desk violation service: API + web UI + job queue.

Jury API (requires X-API-Key):
    POST /api/v1/analyze        multipart field "file" (.mov) -> {"job_id", "status"}
    GET  /api/v1/jobs/{job_id}  -> processing | done (+ verdict, violations) | failed

Web UI (no key, per FAQ #27 no auth needed for users):
    GET  /                      upload page
    POST /web/analyze, GET /web/jobs/{job_id}  same logic, used by the page

The ML team only replaces app/pipeline.py:analyze().
"""
import json
import logging
import os
import queue
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .pipeline import analyze

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("service")

API_KEY = os.environ.get("API_KEY", "")
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "jobs.db"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "300"))
ALLOWED_EXT = {".mov", ".mp4"}
STATIC_DIR = Path(__file__).parent / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------- storage (SQLite, survives restarts) ----------
_db_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.execute(
    "CREATE TABLE IF NOT EXISTS jobs ("
    "id TEXT PRIMARY KEY, status TEXT, result TEXT, error TEXT,"
    "created REAL, started REAL, finished REAL, filename TEXT, path TEXT)"
)
_conn.commit()


def db(sql, params=()):
    with _db_lock:
        cur = _conn.execute(sql, params)
        _conn.commit()
        return cur.fetchall()


# ---------- single GPU worker with a queue ----------
jobs_q: "queue.Queue[tuple[str, str]]" = queue.Queue()


def worker():
    while True:
        job_id, path = jobs_q.get()
        t0 = time.time()
        db("UPDATE jobs SET started=? WHERE id=?", (t0, job_id))
        try:
            result = analyze(path)
            violations = result.get("violations") or []
            result = {
                "verdict": "violation" if violations else "no_violation",
                "violations": violations,
            }
            db(
                "UPDATE jobs SET status='done', result=?, finished=? WHERE id=?",
                (json.dumps(result, ensure_ascii=False), time.time(), job_id),
            )
            log.info("job %s done in %.1fs, %d violation(s)", job_id, time.time() - t0, len(violations))
        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", job_id)
            db(
                "UPDATE jobs SET status='failed', error=?, finished=? WHERE id=?",
                (f"{type(e).__name__}: {e}", time.time(), job_id),
            )
        finally:
            jobs_q.task_done()


@asynccontextmanager
async def lifespan(_app):
    if not API_KEY:
        log.warning("API_KEY is empty: all /api/v1 requests will be rejected")
    # re-queue jobs interrupted by a restart
    for job_id, path in db("SELECT id, path FROM jobs WHERE status='processing' ORDER BY created"):
        if path and Path(path).exists():
            jobs_q.put((job_id, path))
            log.info("re-queued %s after restart", job_id)
        else:
            db("UPDATE jobs SET status='failed', error='file lost after restart' WHERE id=?", (job_id,))
    threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(title="Cash-desk violation detection", lifespan=lifespan)


# ---------- helpers ----------
def require_key(x_api_key: str | None = Header(default=None)):
    if not API_KEY or not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


async def create_job(file: UploadFile) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:          # never reject on the name alone: the decoder decides (a bad file -> failed job)
        ext = ".mov"
    job_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{job_id}{ext}"
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB")
            out.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty file")
    db(
        "INSERT INTO jobs (id, status, created, filename, path) VALUES (?, 'processing', ?, ?, ?)",
        (job_id, time.time(), file.filename, str(dest)),
    )
    jobs_q.put((job_id, str(dest)))
    log.info("job %s queued (%s, %.1f MB)", job_id, file.filename, size / 1e6)
    return {"job_id": job_id, "status": "processing"}


def job_payload(job_id: str) -> dict:
    rows = db("SELECT status, result, error FROM jobs WHERE id=?", (job_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="Job not found")
    status, result, error = rows[0]
    if status == "done":
        return {"job_id": job_id, "status": "done", **json.loads(result)}
    if status == "failed":
        return {"job_id": job_id, "status": "failed", "error": error}
    return {"job_id": job_id, "status": "processing"}


# ---------- jury API ----------
@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def api_analyze(file: UploadFile = File(...)):
    return await create_job(file)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def api_job(job_id: str):
    return job_payload(job_id)


# ---------- web UI ----------
@app.post("/web/analyze")
async def web_analyze(file: UploadFile = File(...)):
    return await create_job(file)


@app.get("/web/jobs/{job_id}")
def web_job(job_id: str):
    return job_payload(job_id)


@app.get("/health")
def health():
    return {"ok": True, "queue": jobs_q.qsize()}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
