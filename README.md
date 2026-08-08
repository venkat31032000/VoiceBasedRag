# Voice RAG

Talk to your documents, hear the answer back. Two files, no vector database, no
audio libraries.

- **Speech in / out** — browser Web Speech API. Nothing to install, nothing to
  pay for, no Whisper model competing for your 4 GB of VRAM.
- **Retrieval** — `all-MiniLM-L6-v2` embeddings, cosine similarity over a numpy
  matrix held in memory. Instant up to roughly 50k chunks.
- **Generation** — Ollama by default; flip one env var for OpenAI or Anthropic.
- **Streaming** — tokens stream to the page and are spoken sentence by sentence,
  so audio starts about a second after you stop talking instead of after the
  full answer is written.

Chrome or Edge only. Firefox and Safari have no `SpeechRecognition`.

---

## Run it

```bash
pip install -r requirements.txt

# option A — local, free
ollama pull llama3.2:3b
uvicorn app:app --port 8000

# option B — hosted, faster and noticeably smarter
export LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-...
uvicorn app:app --port 8000
```

Open <http://localhost:8000>, click **Add files**, then hold a conversation.
Space bar toggles the mic. Pressing it while the assistant is talking cuts it
off mid-sentence.

### Environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama` \| `openai` \| `anthropic` |
| `LLM_MODEL` | `llama3.2:3b` | e.g. `gpt-4o-mini`, `claude-sonnet-4-6` |
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `TOP_K` | `4` | chunks retrieved per question |
| `MIN_SCORE` | `0.18` | below this, it says it doesn't know |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | |

Recognition language is `LANG` at the top of the `<script>` block in
`index.html` — it ships as `en-IN`.

---

## Three hours

Working backwards from a demo you can record.

| | | |
| --- | --- | --- |
| **0:00–0:25** | Install, `ollama pull`, first successful `uvicorn` boot | The MiniLM download is ~90 MB and only happens once |
| **0:25–0:40** | Load your documents, `curl` the backend | Get text-only RAG correct before any audio is involved |
| **0:40–0:55** | Open the UI in Chrome, grant mic, ask one question end to end | If this works, everything after is polish |
| **0:55–1:45** | Tune retrieval | The three knobs below, in that order |
| **1:45–2:20** | Write and rehearse a demo script | Five questions you know retrieve well, plus one deliberately out of scope |
| **2:20–2:50** | Record the screen capture, write the README | Audio matters — record with the laptop mic off and system audio on |
| **2:50–3:00** | Slack | You will need it |

**Do not** spend the first hour on local Whisper. That single decision is the
difference between finishing and not.

### Tuning, in priority order

1. **Chunk size** (`chunk_text`, default 700). Dense reference material wants
   400–500. Narrative or transcript-like text wants 900–1200.
2. **`TOP_K`**. More context is not better here — the answer is capped at three
   sentences, and a 3B model given eight chunks will hedge and ramble.
3. **`MIN_SCORE`**. Raise it if it invents answers to off-topic questions, lower
   it if it refuses things that are plainly in the documents.

Leave the system prompt alone unless answers are running long. Brevity is what
makes a voice interface bearable; every sentence costs about four seconds of
somebody's patience.

---

## When something breaks

| Symptom | Cause | Fix |
| --- | --- | --- |
| Mic button dead, state reads "no speech recognition" | Firefox or Safari | Chrome or Edge |
| Mic permission never prompts | Page not on `localhost` or HTTPS | Use `localhost`, not the LAN IP |
| Transcript appears, no answer | Ollama not running | `ollama serve`, check `/health` |
| Answers are long and unreadable aloud | Model ignoring the prompt | Lower `num_predict`, or move to `gpt-4o-mini` |
| Robotic or wrong-accent voice | Whatever the OS offers | Change `LANG`; on Windows add voices in Settings → Time & Language → Speech |
| Meter frozen while listening | Analyser could not get the mic | Harmless, it falls back to a synthetic animation |
| "I don't have anything about that" for everything | Index empty | Check `/health` for a non-zero chunk count |

`GET /health` is the first thing to check for any backend problem — it reports
chunk count, indexed filenames, and the active provider.

---

## What to say about it afterwards

The interesting engineering here is not the retrieval, it's the latency budget.
A voice turn has roughly a two-second tolerance before it feels broken, and a
naive implementation spends all of it waiting for the LLM to finish. Three
choices buy that back: sentence-buffered TTS so speech begins on the first
period rather than the last, a hard three-sentence cap in the system prompt so
there is less to say, and a similarity floor so out-of-scope questions return
immediately without a generation call at all.

### Worth adding, if there's time later

- Conversation memory — currently every question is independent, so follow-ups
  like "what about the second one" fail
- Swap numpy for FAISS once the corpus is large enough to justify it
- `faster-whisper` as an STT fallback for browsers without Web Speech
- Streaming TTS from a real model (Piper, ElevenLabs) instead of OS voices
