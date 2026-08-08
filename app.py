"""
Voice RAG — single-file backend.

Run:
    pip install -r requirements.txt
    uvicorn app:app --port 8000
    open http://localhost:8000

Drop .pdf / .txt / .md files into ./docs and hit "Rebuild index" in the UI.
"""

import json
import os
import pathlib
import re
from typing import Iterator, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
HERE = pathlib.Path(__file__).parent

# Reads .env sitting next to this file. Optional — plain shell exports still
# work, and anything already exported wins over the file.
try:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")
except ImportError:
    pass

DOCS_DIR = HERE / "docs"
DOCS_DIR.mkdir(exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | anthropic | ollama

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "ollama": "llama3.2:3b",
}
LLM_MODEL = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(LLM_PROVIDER, "gpt-4o-mini")

# Sonnet 5 and Opus 5 default to high effort, which is the wrong trade for a
# voice turn — low keeps thinking shallow and the first token early.
ANTHROPIC_EFFORT = os.getenv("ANTHROPIC_EFFORT", "low")

# effort is not accepted by every model; Haiku 4.5 in particular rejects it
EFFORT_MODELS = (
    "claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5",
    "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
    "claude-sonnet-4-6",
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.18"))

SUPPORTED = {".pdf", ".txt", ".md"}

print(f"[voice-rag] loading embedder: {EMBED_MODEL}")
encoder = SentenceTransformer(EMBED_MODEL)

# in-memory index — fine up to ~50k chunks, and it cannot break at demo time
CHUNKS: List[str] = []
SOURCES: List[str] = []
MATRIX: Optional[np.ndarray] = None


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------
def read_file(path: pathlib.Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = 700, overlap: int = 120) -> List[str]:
    """Paragraph-aware chunking. Keeps semantic units together where it can."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}" if buf else para
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        if len(para) <= size:
            buf = para
        else:
            step = max(size - overlap, 200)
            for i in range(0, len(para), step):
                piece = para[i : i + size].strip()
                if piece:
                    chunks.append(piece)
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) > 40]


def rebuild_index() -> int:
    global CHUNKS, SOURCES, MATRIX
    CHUNKS, SOURCES = [], []

    for path in sorted(DOCS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        try:
            body = read_file(path)
        except Exception as exc:  # a bad PDF should not take down the index
            print(f"[voice-rag] skipped {path.name}: {exc}")
            continue
        for chunk in chunk_text(body):
            CHUNKS.append(chunk)
            SOURCES.append(path.name)

    if not CHUNKS:
        MATRIX = None
        print("[voice-rag] index empty — add files to ./docs")
        return 0

    MATRIX = encoder.encode(
        CHUNKS, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    ).astype("float32")
    print(f"[voice-rag] indexed {len(CHUNKS)} chunks from {len(set(SOURCES))} files")
    return len(CHUNKS)


def retrieve(query: str, k: int = TOP_K):
    if MATRIX is None:
        return []
    q = encoder.encode([query], normalize_embeddings=True).astype("float32")[0]
    scores = MATRIX @ q  # cosine, vectors are normalized
    order = np.argsort(-scores)[:k]
    return [
        (CHUNKS[i], SOURCES[i], float(scores[i]))
        for i in order
        if scores[i] >= MIN_SCORE
    ]


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
SYSTEM = """You answer questions from the provided context. Your reply is read aloud by a speech synthesizer.

Rules:
- Two or three short sentences. Never more.
- Use only the context. If the answer is not there, say you don't have it in the documents.
- Plain spoken prose. No markdown, no bullets, no headings, no code, no URLs, no citations.
- Write numbers and dates the way a person would say them out loud.
- Lead with the answer. No preamble."""

REWRITE_SYSTEM = """Rewrite the user's latest question as a standalone search query.

- Resolve every pronoun and reference using the conversation above it.
- Keep the user's own vocabulary; do not add topics they did not mention.
- Output only the query. No quotes, no explanation, no preamble.
- If the question already stands alone, output it unchanged."""

# Retrieval embeds the question literally, so "how does it differ from RNNs"
# retrieves nothing useful — "it" carries no meaning. Rewriting is only worth
# the extra round trip when the question actually depends on what came before.
_DEICTIC = re.compile(
    r"\b(it|its|it's|that|this|they|them|their|those|these|he|him|she|her|"
    r"the same|the above|the former|the latter|instead)\b",
    re.IGNORECASE,
)


def is_followup(question: str) -> bool:
    return len(question.split()) <= 6 or bool(_DEICTIC.search(question))


def format_history(history, limit: int = 3) -> str:
    return "\n".join(f"Q: {t.q}\nA: {t.a}" for t in history[-limit:])


def rewrite_query(question: str, history) -> str:
    prompt = f"{format_history(history)}\n\nLatest question: {question}"
    out = "".join(STREAMERS[LLM_PROVIDER](REWRITE_SYSTEM, prompt)).strip()
    # A rewrite that comes back empty or absurdly long is a failed rewrite
    if not out or len(out) > 300:
        return question
    return out.strip('"')


def stream_ollama(system: str, user: str) -> Iterator[str]:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "options": {"temperature": 0.2, "num_predict": 220},
    }
    with httpx.stream(
        "POST", f"{OLLAMA_HOST}/api/chat", json=payload, timeout=180
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            token = json.loads(line).get("message", {}).get("content", "")
            if token:
                yield token


_openai_client = None


def stream_openai(system: str, user: str) -> Iterator[str]:
    global _openai_client
    from openai import OpenAI

    if _openai_client is None:
        _openai_client = OpenAI()  # reads OPENAI_API_KEY

    stream = _openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
        temperature=0.2,
        max_tokens=250,
    )
    for event in stream:
        if not event.choices:
            continue
        token = event.choices[0].delta.content
        if token:
            yield token


_anthropic_client = None


def stream_anthropic(system: str, user: str) -> Iterator[str]:
    """Anthropic Messages API. text_stream yields only visible text, so any
    thinking blocks are skipped rather than spoken aloud."""
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    kwargs = {
        "model": LLM_MODEL,
        "max_tokens": 1024,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    # No temperature: models with adaptive thinking reject anything but the
    # default. Brevity comes from the system prompt instead.
    if LLM_MODEL.startswith(EFFORT_MODELS):
        kwargs["output_config"] = {"effort": ANTHROPIC_EFFORT}

    with _anthropic_client.messages.stream(**kwargs) as stream:
        for token in stream.text_stream:
            yield token


STREAMERS = {
    "ollama": stream_ollama,
    "openai": stream_openai,
    "anthropic": stream_anthropic,
}


# --------------------------------------------------------------------------
# api
# --------------------------------------------------------------------------
app = FastAPI(title="Voice RAG")


class Turn(BaseModel):
    q: str
    a: str


class Ask(BaseModel):
    question: str
    history: List[Turn] = []


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


KEY_VAR = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


@app.get("/health")
def health():
    var = KEY_VAR.get(LLM_PROVIDER)
    return {
        "chunks": len(CHUNKS),
        "files": sorted(set(SOURCES)),
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "key_ok": bool(os.getenv(var)) if var else True,
    }


@app.post("/reindex")
def reindex():
    count = rebuild_index()
    return {"chunks": count, "files": sorted(set(SOURCES))}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        suffix = pathlib.Path(f.filename or "").suffix.lower()
        if suffix not in SUPPORTED:
            continue
        target = DOCS_DIR / pathlib.Path(f.filename).name
        target.write_bytes(await f.read())
        saved.append(target.name)
    count = rebuild_index()
    return {"saved": saved, "chunks": count}


@app.get("/retrieve")
def retrieve_debug(q: str, k: int = TOP_K):
    """Retrieval without generation. Used by eval.py and handy for debugging
    whether a bad answer is a retrieval problem or a generation problem."""
    hits = retrieve(q, k)
    return {
        "query": q,
        "hits": [
            {"source": s, "score": round(sc, 3), "text": t}
            for t, s, sc in hits
        ],
    }


@app.post("/ask")
def ask(body: Ask):
    question = body.question.strip()
    history = body.history

    def gen():
        if not question:
            yield sse({"type": "done"})
            return

        # Resolve pronouns against the conversation before embedding
        search_q = question
        if history and is_followup(question):
            try:
                search_q = rewrite_query(question, history)
                if search_q != question:
                    yield sse({"type": "query", "text": search_q})
            except Exception as exc:
                print(f"[voice-rag] rewrite failed, using original: {exc}")

        hits = retrieve(search_q)
        if not hits:
            msg = (
                "I don't have anything about that in the documents."
                if CHUNKS
                else "There are no documents indexed yet. Add some and rebuild the index."
            )
            for word in msg.split(" "):
                yield sse({"type": "token", "text": word + " "})
            yield sse({"type": "done"})
            return

        yield sse({"type": "sources", "sources": sorted({s for _, s, _ in hits})})

        context = "\n\n---\n\n".join(f"[{src}]\n{txt}" for txt, src, _ in hits)
        convo = ""
        if history:
            convo = f"Earlier in this conversation:\n{format_history(history)}\n\n"
        user = f"{convo}Context:\n{context}\n\nQuestion: {question}"

        try:
            for token in STREAMERS[LLM_PROVIDER](SYSTEM, user):
                yield sse({"type": "token", "text": token})
        except Exception as exc:
            yield sse({"type": "error", "text": f"{LLM_PROVIDER}: {exc}"})
        yield sse({"type": "done"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.on_event("startup")
def startup():
    var = KEY_VAR.get(LLM_PROVIDER)
    if var and not os.getenv(var):
        print(f"\n  !! {var} is not set — every question will fail.")
        print(f"     export {var}=sk-...\n")
    print(f"[voice-rag] {LLM_PROVIDER} · {LLM_MODEL}")
    rebuild_index()