from __future__ import annotations
import os
import re
import logging
from time import time
from typing import Dict, List, Optional

import httpx
import numpy as np
from fastembed import TextEmbedding
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# -------------------
# Logging & config
# -------------------
logger = logging.getLogger("member-qa")
logging.basicConfig(level=logging.INFO)

APP_NAME = "member-qa"

# IMPORTANT: default to the exact host they gave (HTTPS, no extra path)
MESSAGES_API_BASE = os.getenv(
    "MESSAGES_API_BASE",
    "https://november7-730026606190.europe-west1.run.app",
)

TIMEOUT = float(os.getenv("MESSAGES_API_TIMEOUT", "25"))
PAGE_LIMIT = int(os.getenv("MESSAGES_API_LIMIT", "50"))
MAX_PAGES = int(os.getenv("MESSAGES_API_MAX_PAGES", "20"))

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "member-qa/1.0",
}

# RAG-lite config
EMBED_K = int(os.getenv("EMBED_TOPK", "8"))

_embedder: Optional[TextEmbedding] = None
_msg_texts: List[str] = []          # indexed message texts
_msg_meta: List[Dict] = []          # {id, user_name, timestamp}
_msg_vecs: Optional[np.ndarray] = None  # [N, D] normalized embeddings

# Simple cache of raw messages
_cache = {"t": 0.0, "data": []}
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "120"))

app = FastAPI(title=APP_NAME)


# -------------------
# Input / Output Models
# -------------------
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


# -------------------
# Upstream client
# -------------------
async def fetch_messages_page(skip: int = 0, limit: int = PAGE_LIMIT) -> Dict:
    """
    Fetch one page from the upstream /messages endpoint.

    We are defensive about MESSAGES_API_BASE:
    - If it already ends with /messages or /messages/, we don't append again.
    - Otherwise, we append /messages.
    """
    base = MESSAGES_API_BASE.rstrip("/")
    if base.endswith("/messages"):
        # e.g. base = https://.../messages
        url = base
    else:
        # e.g. base = https://... -> https://.../messages
        url = f"{base}/messages"

    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers=HEADERS,
    ) as client:
        r = await client.get(url, params={"skip": int(skip), "limit": int(limit)})
        if r.status_code >= 400:
            logger.error(
                "Upstream error %s for %s?skip=%s&limit=%s ; body=%s",
                r.status_code,
                url,
                skip,
                limit,
                r.text[:300],
            )
        r.raise_for_status()
        return r.json()


async def fetch_all_messages(max_pages: int = MAX_PAGES) -> List[Dict]:
    items: List[Dict] = []
    skip = 0
    for _ in range(max_pages):
        page = await fetch_messages_page(skip=skip, limit=PAGE_LIMIT)
        batch = page.get("items", []) or []
        if not batch:
            break
        items.extend(batch)
        if len(batch) < PAGE_LIMIT:
            break
        skip += PAGE_LIMIT
    logger.info(
        "Fetched %d messages (pages=%d, page_size=%d)",
        len(items),
        (skip // PAGE_LIMIT) + 1,
        PAGE_LIMIT,
    )
    return items


# -------------------
# Intent detection (regex)
# -------------------
TRIP_Q_RE = re.compile(
    r"(?i)when\s+is\s+(.+?)\s+planning\s+(?:her|his|their)?\s*trip\s+to\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*\??\s*$"
)

YESNO_TRIP_Q_RE = re.compile(
    r"(?i)is\s+(.+?)\s+(?:going|traveling|travelling|heading)\s+to\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*\??\s*$"
)

CARS_Q_RE = re.compile(r"(?i)how\s+many\s+cars\s+does\s+(.+?)\s+have\??")
FAV_Q_RE = re.compile(r"(?i)what\s+are\s+(.+?)['’]s\s+favorite\s+restaurants\??")

NAME_NORM = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
DATE_WORDS = r"(?:on|around|in|by|this|next|coming|on the)"

TRIP_PATTERNS = [
    re.compile(
        rf"(?i)\b(trip|travel|fly|flight|going)\b.*\bto\b\s*(?P<city>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)[^\n]*\b{DATE_WORDS}\b\s*(?P<when>[A-Za-z0-9 ,./-]+)"
    ),
    re.compile(
        r"(?i)\bto\s+(?P<city>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b[^\n]*\b(on|around|in|by)\b\s*(?P<when>[A-Za-z0-9 ,./-]+)"
    ),
    # extra leniency for phrasing like "headed to London next Friday"
    re.compile(
        rf"(?i)\b(?:to|headed to|off to)\s+(?P<city>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b.*?\b{DATE_WORDS}\b\s*(?P<when>[A-Za-z0-9 ,./-]+)"
    ),
]

CARS_PATTERNS = [
    re.compile(r"(?i)\b(?P<count>\d+)\s+cars?\b"),
    re.compile(r"(?i)\b(has|own(?:s)?)\b[^\n]*\b(?P<count>\d+)\s+cars?\b"),
]

FAV_PATTERNS = [
    re.compile(r"(?i)favorite\s+restaurants?\s*:?\s*(?P<list>.+)$"),
    re.compile(
        r"(?i)\b(love|loves|like|likes)\s+(?P<list>(?:[A-Z][\w'&]+(?:\s+[A-Z][\w'&]+)*)(?:\s*,\s*(?:and\s+)?[A-Z][\w'&]+(?:\s+[A-Z][\w'&]+)*)*)"
    ),
]


# -------------------
# Extraction helpers
# -------------------
def normalize_city(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def messages_for_user(all_msgs: List[Dict], name_query: str) -> List[str]:
    target = NAME_NORM(name_query)
    out: List[str] = []
    for m in all_msgs:
        if target in NAME_NORM(m.get("user_name")):
            msg = m.get("message") or ""
            if msg.strip():
                out.append(msg)
    return out


def extract_trip_when_to_city(texts: List[str], city: str) -> Optional[str]:
    city_norm = normalize_city(city)
    for t in texts:
        for pat in TRIP_PATTERNS:
            m = pat.search(t)
            if not m:
                continue
            det_city = (m.groupdict().get("city") or "").strip()
            when = (m.groupdict().get("when") or "").strip()
            if det_city and normalize_city(det_city) != city_norm:
                continue
            if when:
                return when
    return None


def extract_car_count(texts: List[str]) -> Optional[str]:
    best = None
    for t in texts:
        for pat in CARS_PATTERNS:
            m = pat.search(t)
            if m:
                try:
                    c = int(m.group("count"))
                    best = c if best is None or c > best else best
                except Exception:
                    pass
    return str(best) if best is not None else None


def extract_favorite_restaurants(texts: List[str]) -> Optional[str]:
    for t in texts:
        for pat in FAV_PATTERNS:
            m = pat.search(t)
            if m:
                raw = m.group("list")
                items = re.split(r"\s*,\s*|\s+and\s+", raw)
                items = [i.strip().strip(". ") for i in items if i.strip()]
                seen, ordered = set(), []
                for i in items:
                    k = i.lower()
                    if k not in seen:
                        seen.add(k)
                        ordered.append(i)
                return ", ".join(ordered)
    return None


# -------------------
# RAG-lite: embeddings index & retrieval
# -------------------
@app.on_event("startup")
async def build_index() -> None:
    """
    Build an embedding index over all messages once at startup.
    If it fails, we just skip RAG fallback.
    """
    global _embedder, _msg_texts, _msg_meta, _msg_vecs

    try:
        msgs = await fetch_all_messages()
    except Exception as e:
        logger.exception("Failed to build embedding index from upstream messages: %s", e)
        _embedder = None
        _msg_vecs = None
        _msg_texts = []
        _msg_meta = []
        return

    texts: List[str] = []
    meta: List[Dict] = []

    for m in msgs:
        t = (m.get("message") or "").strip()
        if not t:
            continue
        texts.append(t)
        meta.append(
            {
                "id": m.get("id"),
                "user_name": m.get("user_name"),
                "timestamp": m.get("timestamp"),
            }
        )

    if not texts:
        logger.warning("No non-empty messages found to index.")
        return

    logger.info("Building embedding index over %d messages", len(texts))
    _embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vecs = list(_embedder.embed(texts))
    _msg_vecs = np.array(vecs, dtype=np.float32)
    _msg_texts = texts
    _msg_meta = meta

    # L2-normalize for cosine similarity
    norms = np.linalg.norm(_msg_vecs, axis=1, keepdims=True) + 1e-12
    _msg_vecs /= norms


def retrieve_similar_messages(
    query: str,
    user_hint: Optional[str] = None,
    k: int = EMBED_K,
) -> List[Dict]:
    """
    RAG-lite retrieval: embed the query, find top-k nearest messages by cosine similarity.
    If user_hint is provided, lightly bias toward that user's messages.
    """
    if not query or _msg_vecs is None or _embedder is None:
        return []

    q_vec = list(_embedder.embed([query]))[0]
    qv = np.array(q_vec, dtype=np.float32)
    qv /= (np.linalg.norm(qv) + 1e-12)

    sims = _msg_vecs @ qv  # cosine similarity

    n = len(sims)
    if n == 0:
        return []

    k = min(k, n)
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    results: List[Dict] = []
    for i in idx:
        m = {
            **_msg_meta[i],
            "text": _msg_texts[i],
            "score": float(sims[i]),
        }
        results.append(m)

    if user_hint:
        hint = NAME_NORM(user_hint)
        results.sort(
            key=lambda r: 0 if hint in NAME_NORM(r.get("user_name") or "") else 1
        )

    return results


# -------------------
# API Endpoints
# -------------------
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    # Use cache to avoid hammering upstream
    global _cache
    now = time()
    if now - _cache["t"] > CACHE_TTL or not _cache["data"]:
        try:
            _cache["data"] = await fetch_all_messages()
            _cache["t"] = now
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text[:300]
            except Exception:
                pass
            code = e.response.status_code if e.response is not None else "unknown"
            return AskResponse(
                answer=f"Upstream API error ({code}): {body or 'no details'}"
            )
        except httpx.RequestError as e:
            return AskResponse(
                answer=f"Upstream API request failed (network/timeout): {e}"
            )
        except Exception as e:
            logger.exception("Unexpected error fetching messages: %s", e)
            return AskResponse(
                answer="Unexpected error fetching messages. Please try again."
            )

    all_msgs: List[Dict] = _cache["data"]

    # Intent 1: Trip timing to a city
    m = TRIP_Q_RE.search(q)
    if m:
        name = m.group(1).strip().rstrip("?.!,")
        city = m.group(2).strip().rstrip("?.!,")
        logger.info("Parsed trip intent → name=%r city=%r", name, city)
        texts = messages_for_user(all_msgs, name)
        if not texts:
            return AskResponse(answer=f"I couldn't find any messages for {name}.")
        when = extract_trip_when_to_city(texts, city)
        return AskResponse(answer=when or f"No trip to {city} found for {name}.")

    # Intent 1b: Yes/No trip ("Is Layla going to London?")
    m = YESNO_TRIP_Q_RE.search(q)
    if m:
        name = m.group(1).strip().rstrip("?.!,")
        city = m.group(2).strip().rstrip("?.!,")
        logger.info("Parsed yes/no trip intent → name=%r city=%r", name, city)
        texts = messages_for_user(all_msgs, name)
        if not texts:
            return AskResponse(answer=f"I couldn't find any messages for {name}.")

        when = extract_trip_when_to_city(texts, city)
        if when:
            return AskResponse(
                answer=f"Yes, {name} mentioned a trip to {city} around {when}."
            )

        city_lower = city.lower()
        found_city = any(
            (city_lower in t.lower())
            and ("trip" in t.lower() or "going" in t.lower())
            for t in texts
        )
        if found_city:
            return AskResponse(
                answer=f"{name} mentioned a trip to {city}, but I couldn't infer an exact date."
            )

        return AskResponse(
            answer=f"I couldn't find any trip to {city} mentioned by {name}."
        )

    # Intent 2: Car count
    m = CARS_Q_RE.search(q)
    if m:
        name = m.group(1).strip().rstrip("?.!,")
        texts = messages_for_user(all_msgs, name)
        if not texts:
            return AskResponse(answer=f"I couldn't find any messages for {name}.")
        count = extract_car_count(texts)
        return AskResponse(answer=count or f"I couldn't infer car ownership for {name}.")

    # Intent 3: Favorite restaurants
    m = FAV_Q_RE.search(q)
    if m:
        name = m.group(1).strip().rstrip("?.!,")
        texts = messages_for_user(all_msgs, name)
        if not texts:
            return AskResponse(answer=f"I couldn't find any messages for {name}.")
        favs = extract_favorite_restaurants(texts)
        return AskResponse(
            answer=favs or f"No favorite restaurants found for {name}."
        )

    # -------------------
    # RAG-lite fallback
    # -------------------
    user_hint = None
    name_match = re.search(r"(?i)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", q)
    if name_match:
        user_hint = name_match.group(1)

    candidates = retrieve_similar_messages(q, user_hint=user_hint, k=EMBED_K)
    texts = [c["text"] for c in candidates]

    if texts:
        # Trip: infer city from question if present
        city_match = re.search(
            r"(?i)\bto\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", q
        )
        city_guess = city_match.group(1) if city_match else None
        if city_guess:
            when = extract_trip_when_to_city(texts, city_guess)
            if when:
                return AskResponse(answer=when)

        count = extract_car_count(texts)
        if count:
            return AskResponse(answer=count)

        favs = extract_favorite_restaurants(texts)
        if favs:
            return AskResponse(answer=favs)

        # If nothing parsed but we have candidates, return a snippet
        snippet = texts[0]
        return AskResponse(answer=snippet[:220])

    # Final fallback
    return AskResponse(
        answer="I couldn't understand the question. Ask about trips to a city, car counts, or favorite restaurants."
    )


@app.get("/")
async def root():
    return {"service": APP_NAME, "endpoints": ["/ask"], "status": "ok"}
