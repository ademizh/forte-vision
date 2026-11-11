"""
Local norm search (step 2): finds the rules that match what was observed. No internet, no cloud.

Corpus
  * norms/norms.yaml   one entry per checkable rule: document, clause, text, conditions, exceptions, tags
  * norms/*.txt|*.md   optional official texts of Rules No. 231 / No. 14 (UTF-8). They are split into clauses
                       ("12. ...", "п. 12", "12)") and searched too, so the description can quote the real clause.

retrieve(tags, words, k) ranks the entries with BM25 over their tags + text; the query is the observation kinds
(cashier_leaves, region_safe, object_key, ...) plus words from the observations.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import yaml

NORMS_DIR = Path(__file__).resolve().parent.parent / "norms"
_TOKEN = re.compile(r"[a-zа-яё0-9_]+", re.I)


def tokens(text: str) -> list[str]:
    out = []
    for w in _TOKEN.findall(str(text).lower()):
        out.append(w)
        if len(w) > 5 and not w.isascii():
            out.append(w[:5])               # crude Russian stem: "ключа", "ключом" -> "ключ"
    return out


def _split_clauses(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"\n\s*(?=(?:п\.\s*)?\d{1,3}[\.\)]\s)", "\n" + text)
    out = []
    for p in parts:
        p = p.strip()
        m = re.match(r"(?:п\.\s*)?(\d{1,3})[\.\)]\s", p)
        if m and len(p) > 20:
            out.append((f"п. {m.group(1)}", p[:1200]))
    return out


@lru_cache(maxsize=1)
def corpus() -> dict:
    data = yaml.safe_load((NORMS_DIR / "norms.yaml").read_text(encoding="utf-8"))
    docs = data.get("documents", {})
    entries = []
    for n in data.get("norms", []):
        entries.append({**n, "document_name": docs.get(n["document"], n["document"]), "kind": "rule"})
    for f in sorted(NORMS_DIR.glob("*.txt")) + sorted(NORMS_DIR.glob("*.md")):
        if f.name.lower().startswith("readme"):
            continue
        for clause, txt in _split_clauses(f.read_text(encoding="utf-8", errors="replace")):
            entries.append({"id": f"{f.stem}:{clause}", "applies_to": None, "document": f.stem,
                            "document_name": f.stem, "clause": clause, "text": txt, "text_status": "official",
                            "conditions": [], "exceptions": [], "tags": [], "kind": "text"})
    for e in entries:
        e["_tok"] = tokens(" ".join([e.get("text", "")] + [str(x) for x in e.get("tags", [])] * 3 +
                                    e.get("conditions", [])))
    n = len(entries)
    df = {}
    for e in entries:
        for w in set(e["_tok"]):
            df[w] = df.get(w, 0) + 1
    avg = sum(len(e["_tok"]) for e in entries) / max(1, n)
    return {"entries": entries, "df": df, "n": n, "avg": avg}


def retrieve(tags: list[str], words: str = "", k: int = 5) -> list[dict]:
    """BM25 ranking of the norm entries for the observed tags + words."""
    c = corpus()
    q = tokens(" ".join(tags) + " " + words)
    scored = []
    for e in c["entries"]:
        tf = {}
        for w in e["_tok"]:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in set(q):
            if w not in tf:
                continue
            idf = math.log(1 + (c["n"] - c["df"][w] + 0.5) / (c["df"][w] + 0.5))
            f = tf[w]
            s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(e["_tok"]) / c["avg"]))
        if s > 0:
            scored.append((s, e))
    scored.sort(key=lambda x: -x[0])
    return [{**{kk: v for kk, v in e.items() if kk != "_tok"}, "score": round(s, 3)} for s, e in scored[:k]]


def for_type(type_id: str) -> dict | None:
    return next(({kk: v for kk, v in e.items() if kk != "_tok"} for e in corpus()["entries"]
                 if e.get("applies_to") == type_id), None)


def official_text(norm: dict) -> str | None:
    """Official clause text from norms/*.txt with the same clause number, if such a file was added."""
    if not norm:
        return None
    num = re.findall(r"\d+", norm.get("clause", ""))
    doc = re.findall(r"\d+", norm.get("document", ""))           # NBRK_231 -> 231: the file name must contain it
    if not num or not doc:
        return None
    for e in corpus()["entries"]:
        if e["kind"] == "text" and doc[0] in e["document"] and re.findall(r"\d+", e["clause"])[:1] == num[:1]:
            return e["text"]
    return None
