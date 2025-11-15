from __future__ import annotations
import os
import re
import logging
from typing import Dict, List, Optional

import httpx
import numpy as np
from fastembed import TextEmbedding
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Logging & config

logger = logging.getLogger("member-qa")
logging.basicConfig(level=logging.INFO)

APP_NAME = "member-qa"


def _clean_base_url(raw: str) -> str:
    """
    Strip whitespace AND remove any control characters (like '\n', '\r', '\t')
    from the base URL. If it ends up empty, fall back to default.
    """
    default = "https://november7-730026606190.europe-west1.run.app"
    if raw is None:
        return default

    # Remove all ASCII control chars 0x00–0x1F and 0x7F
    cleaned = re.sub(r"[\x00-\x1F\x7F]", "", raw)
    cleaned = cleaned.strip()

    if not cleaned:
        logger.warning(
        "MESSAGES_API_BASE is empty after cleaning; falling back to default host."
        )
        return default

    return cleaned


_raw_base = os.getenv(
    "MESSAGES_API_BASE",
    "https://november7-730026606190.europe-west1.run.app",
)

MESSAGES_API_BASE = _clean_base_url(_raw_base)
logger.info("Using MESSAGES_API_BASE: %r", MESSAGES_API_BASE)

TIMEOUT = float(os.getenv("MESSAGES_API_TIMEOUT", "25"))
PAGE_LIMIT = int(os.getenv("MESSAGES_API_LIMIT", "50"))
MAX_PAGES = int(os.getenv("MESSAGES_API_MAX_PAGES", "20"))

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "member-qa/1.0",
}

# RAG-lite config
EMBED_K = int(os.getenv("EMBED_TOPK", "8"))

embedder: Optional[TextEmbedding] = None
msg_texts: List[str] = []          # indexed message texts
msg_meta: List[Dict] = []          # {id, user_name, timestamp}
msg_vecs: Optional[np.ndarray] = None  # [N, D] normalized embeddings


raw_msgs: List[Dict] = []

app = FastAPI(title=APP_NAME)



# Input / Output Models

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str



# Upstream client

async def fetch_messages_page(skip: int = 0, limit: int = PAGE_LIMIT) -> Dict:
    """
    Fetch one page from the upstream /messages endpoint.

    We are defensive about MESSAGES_API_BASE:
    - Strip whitespace / control chars.
    - If it already ends with /messages or /messages/, we don't append again.
    - Otherwise, we append /messages.
    """
    raw_base = MESSAGES_API_BASE or ""
    # remove any control chars that somehow survived
    cleaned_base = re.sub(r"[\x00-\x1F\x7F]", "", raw_base).strip()

    if not cleaned_base:
        raise RuntimeError("MESSAGES_API_BASE is empty or invalid after cleaning")

    base = cleaned_base.rstrip("/")
    if base.endswith("/messages"):
        url = base
    else:
        url = f"{base}/messages"

    # one more guard at the final URL level
    url = re.sub(r"[\x00-\x1F\x7F]", "", url)

    logger.info("Using upstream messages URL: %r", url)

    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers=HEADERS,
    ) as client:
        try:
            r = await client.get(url, params={"skip": int(skip), "limit": int(limit)})
        except httpx.InvalidURL as e:
            logger.error("Invalid URL used for upstream request: %r (%s)", url, e)
            raise

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
    """
    Fetch multiple pages, but stop gracefully on 400/401/404/405 instead of blowing up.
    This way, we still use whatever data we got from earlier pages.
    """
    items: List[Dict] = []
    skip = 0

    for _ in range(max_pages):
        try:
            page = await fetch_messages_page(skip=skip, limit=PAGE_LIMIT)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code if e.response is not None else None
            if code in (400, 401, 404, 405):
                logger.warning(
                    "Stopping pagination at skip=%s due to upstream %s",
                    skip,
                    code,
                )
                break
            raise

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
        (skip // PAGE_LIMIT) + 1 if items else 0,
        PAGE_LIMIT,
    )
    return items


# Intent detection (regex)

TRIP_Q_RE = re.compile(
    r"(?i)when\s+is\s+(.+?)\s+planning\s+(?:her|his|their)?\s*trip\s+to\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*\??\s*$"
)

YESNO_TRIP_Q_RE = re.compile(
    r"(?i)is\s+(.+?)\s+(?:going|traveling|travelling|heading)\s+to\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*\??\s*$"
)

# New: generic “tell me something about X’s trip”
TRIP_SUMMARY_Q_RE = re.compile(
    r"(?i)tell\s+me\s+something\s+about\s+(.+?)['’]s\s+trip"
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

# Generic "favorite things" / likes (not limited to restaurants)
FAV_THING_PATTERNS = [
    re.compile(
        r"(?i)\bfavorite\b\s+(?:thing|things|place|places|items?|stuff|)\s*(?:is|are|:)?\s*(?P<thing>[^.?!,\n]+)"
    ),
    re.compile(
        r"(?i)\b(love|loves|like|likes|adore|enjoy)\b\s+(?P<thing>[^.?!,\n]+)"
    ),
]

# For name extraction
QUESTION_CAP_STOPWORDS = {
    "What", "When", "Where", "Why", "How", "Who", "Which", "Tell",
    "Does", "Do", "Did", "Is", "Are", "Was", "Were", "Can", "Could",
    "Will", "Would", "Should", "May", "Might",
}



# Extraction helpers

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


def extract_generic_favorites(texts: List[str]) -> Optional[str]:
    """
    Heuristic extraction of 'favorite things' from a set of messages.
    """
    seen = set()
    items: List[str] = []

    for t in texts:
        for pat in FAV_THING_PATTERNS:
            for m in pat.finditer(t):
                thing = (m.group("thing") or "").strip()
                if not thing:
                    continue
                thing = re.sub(
                    r"\b(but|though|however|except)\b.*$", "", thing
                ).strip()
                thing = re.sub(
                    r"^(really|so|just|kind of|kinda)\s+",
                    "",
                    thing,
                    flags=re.I,
                ).strip()
                if not thing:
                    continue
                key = thing.lower()
                if key not in seen:
                    seen.add(key)
                    items.append(thing)

    if not items:
        return None
    return "; ".join(items[:5])


def extract_name_from_question(q: str) -> Optional[str]:
    """
    Try to extract a plausible person name from the question.
    """
    m = re.search(r"(?i)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)['’]s\b", q)
    if m:
        return m.group(1).strip()

    m = re.search(
        r"(?i)\b(?:are|is|does|do|did|was|were|can|could|will|would|should|has|have|had)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        q,
    )
    if m:
        return m.group(1).strip()

    caps = re.findall(r"\b([A-Z][a-z]+)\b", q)
    caps = [c for c in caps if c not in QUESTION_CAP_STOPWORDS]
    if caps:
        return caps[-1].strip()

    return None



# RAG-lite: embeddings index & retrieval

@app.on_event("startup")
async def build_index() -> None:
    """
    Build an embedding index over all messages once at startup.
    If it fails, we just skip RAG fallback.
    """
    global _embedder, _msg_texts, _msg_meta, _msg_vecs, _raw_msgs

    try:
        msgs = await fetch_all_messages()
    except Exception as e:
        logger.exception("Failed to build embedding index from upstream messages: %s", e)
        _embedder = None
        _msg_vecs = None
        _msg_texts = []
        _msg_meta = []
        _raw_msgs = []
        return

    _raw_msgs = msgs  # store raw messages for rule-based logic

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

    sims = _msg_vecs @ qv

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



# API Endpoints

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    global _raw_msgs
    all_msgs: List[Dict] = _raw_msgs

    if not all_msgs:
        try:
            all_msgs = await fetch_all_messages()
            _raw_msgs = all_msgs
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
                answer=f"Unexpected error fetching messages: {type(e).__name__}: {e!r}"
            )

    q_lower = q.lower()

    # Trip summary intent 
    m = TRIP_SUMMARY_Q_RE.search(q)
    if m:
        name = m.group(1).strip().rstrip("?.!,")
        texts = messages_for_user(all_msgs, name)
        if not texts:
            return AskResponse(answer=f"I couldn't find any messages for {name}.")

        trip_snips: List[str] = []
        for t in texts:
            tl = t.lower()
            if (
                "trip" in tl
                or "flight" in tl
                or "fly to" in tl
                or "going to" in tl
                or "travel to" in tl
            ):
                trip_snips.append(t.strip())
            if len(trip_snips) >= 3:
                break

        if not trip_snips:
            return AskResponse(
                answer=f"I couldn't find any detailed trip messages for {name}."
            )

        joined = " | ".join(trip_snips)
        return AskResponse(
            answer=f"Here are some details mentioned about {name}'s trip: {joined}"
        )

    # Generic favorite things
    if "favorite" in q_lower and "restaurant" not in q_lower:
        name = extract_name_from_question(q)
        if name:
            texts_user = messages_for_user(all_msgs, name)
            cands = retrieve_similar_messages(q, user_hint=name, k=EMBED_K)
            cand_texts = [
                c["text"]
                for c in cands
                if NAME_NORM(name) in NAME_NORM(c.get("user_name") or "")
            ]
            texts = texts_user + cand_texts
            if texts:
                favs = extract_generic_favorites(texts)
                if favs:
                    return AskResponse(
                        answer=f"{name}'s favorite things mentioned in the messages include: {favs}."
                    )
                else:
                    return AskResponse(
                        answer=f"I couldn't infer specific favorite things for {name} from the messages."
                    )

    # Looser restaurant intent 
    if "restaurant" in q_lower or "restaurants" in q_lower:
        name = extract_name_from_question(q)
        if name:
            texts = messages_for_user(all_msgs, name)
            if not texts:
                return AskResponse(answer=f"I couldn't find any messages for {name}.")

            favs = extract_favorite_restaurants(texts)
            if favs:
                return AskResponse(
                    answer=f"{name} talks about these restaurants: {favs}."
                )

            restaurant_snips: List[str] = []
            for t in texts:
                tl = t.lower()
                if (
                    "restaurant" in tl
                    or "dinner" in tl
                    or "table at" in tl
                    or "reservation" in tl
                ):
                    restaurant_snips.append(t.strip())
                if len(restaurant_snips) >= 3:
                    break

            if restaurant_snips:
                joined = " | ".join(restaurant_snips)
                return AskResponse(
                    answer=(
                        f"I couldn't extract a clean restaurant list, but here are some "
                        f"restaurant-related messages for {name}: {joined}"
                    )
                )

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

    
    # RAG-lite fallback (for these 3 domains only)
    
    user_hint = extract_name_from_question(q)
    candidates = retrieve_similar_messages(q, user_hint=user_hint, k=EMBED_K)
    texts = [c["text"] for c in candidates]

    if texts:
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

        snippet = texts[0]
        return AskResponse(answer=snippet[:220])

    return AskResponse(
        answer="I couldn't understand the question. Ask about trips to a city, car counts, or favorite restaurants."
    )


@app.post("/ask_generic", response_model=AskResponse)
async def ask_generic(req: AskRequest) -> AskResponse:
    """
    Generic RAG-based Q&A endpoint.
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    if _embedder is None or _msg_vecs is None or not _msg_texts:
        try:
            await build_index()
        except Exception as e:
            logger.exception("Failed to (re)build embedding index: %s", e)
            return AskResponse(
                answer="Search index is not available right now. Please try again later."
            )
        if _embedder is None or _msg_vecs is None:
            return AskResponse(
                answer="Search index is not available right now. Please try again later."
            )

    user_hint = extract_name_from_question(q)
    candidates = retrieve_similar_messages(q, user_hint=user_hint, k=EMBED_K)
    if not candidates:
        return AskResponse(
            answer="I couldn't find anything in the messages that looked relevant to your question."
        )

    top = candidates[:3]
    lines = []
    for c in top:
        uname = c.get("user_name") or "Unknown user"
        text = (c.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"- [{uname}] {text}")

    if not lines:
        return AskResponse(
            answer="I couldn't find anything in the messages that looked relevant to your question."
        )

    answer = (
        "Here are some messages that may answer your question or provide context:\n"
        + "\n".join(lines)
    )
    return AskResponse(answer=answer)


@app.get("/")
async def root():
    return {
        "service": APP_NAME,
        "endpoints": ["/ask", "/ask_generic"],
        "status": "ok",
    }
