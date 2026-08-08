"""
Measure the two things that decide whether this is usable:
retrieval accuracy, and how long you wait before hearing anything.

    uvicorn app:app --port 8000     # in another terminal
    python eval.py

Writes results.md, which you can paste straight into the README.
"""

import json
import pathlib
import statistics
import sys
import time

import httpx

BASE = "http://localhost:8000"
HERE = pathlib.Path(__file__).parent
QUESTIONS = HERE / "questions.json"


def retrieval_hit(question: str, expect: list[str]) -> tuple[bool, float, str]:
    """A hit means at least one retrieved chunk contains one of the expected
    keywords. Crude, but it is objective and it catches real failures."""
    r = httpx.get(f"{BASE}/retrieve", params={"q": question}, timeout=30)
    r.raise_for_status()
    hits = r.json()["hits"]
    if not hits:
        return False, 0.0, "nothing retrieved"

    blob = " ".join(h["text"] for h in hits).lower()
    matched = [kw for kw in expect if kw.lower() in blob]
    top = hits[0]["score"]
    return bool(matched), top, ", ".join(matched) or "no keyword found"


def timings(question: str) -> tuple[float, float, str]:
    """Returns (seconds to first token, seconds to last token, answer).

    Time to first token is the number that matters — it is when speech starts,
    not when the answer finishes.
    """
    start = time.perf_counter()
    first = None
    parts: list[str] = []

    with httpx.stream(
        "POST", f"{BASE}/ask", json={"question": question, "history": []}, timeout=120
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            msg = json.loads(line[6:])
            if msg.get("type") == "token":
                if first is None:
                    first = time.perf_counter() - start
                parts.append(msg["text"])

    total = time.perf_counter() - start
    return (first if first is not None else total), total, "".join(parts).strip()


def main() -> int:
    if not QUESTIONS.exists():
        print(f"missing {QUESTIONS.name} — see the template in the repo")
        return 1

    try:
        health = httpx.get(f"{BASE}/health", timeout=10).json()
    except Exception:
        print(f"no server at {BASE} — start uvicorn first")
        return 1

    if not health["chunks"]:
        print("index is empty, nothing to evaluate")
        return 1

    cases = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    print(f"{health['chunks']} chunks · {health['provider']} · {health['model']}")
    print(f"{len(cases)} questions\n")

    rows = []
    for i, case in enumerate(cases, 1):
        q = case["question"]
        expect = case.get("expect", [])

        try:
            hit, score, detail = retrieval_hit(q, expect)
            ttft, total, answer = timings(q)
        except Exception as exc:
            print(f"{i:2}. ERROR  {q[:50]} — {exc}")
            continue

        rows.append(
            {"q": q, "hit": hit, "score": score, "ttft": ttft,
             "total": total, "answer": answer, "detail": detail}
        )
        mark = "hit " if hit else "MISS"
        print(f"{i:2}. {mark} top={score:.2f}  first={ttft:.2f}s  {q[:46]}")
        if not hit:
            print(f"      → {detail}")

    if not rows:
        print("\nno results")
        return 1

    hits = sum(r["hit"] for r in rows)
    rate = hits / len(rows) * 100
    ttfts = sorted(r["ttft"] for r in rows)
    totals = sorted(r["total"] for r in rows)
    p90 = ttfts[max(0, int(len(ttfts) * 0.9) - 1)]

    print(f"\nretrieval        {hits}/{len(rows)}  ({rate:.0f}%)")
    print(f"first token      median {statistics.median(ttfts):.2f}s   p90 {p90:.2f}s")
    print(f"full answer      median {statistics.median(totals):.2f}s")

    out = [
        "## Evaluation\n",
        f"{len(rows)} questions against `{', '.join(health['files'])}` "
        f"({health['chunks']} chunks, {health['model']}).\n",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Retrieval hit rate | **{rate:.0f}%** ({hits}/{len(rows)}) |",
        f"| Time to first token, median | **{statistics.median(ttfts):.2f}s** |",
        f"| Time to first token, p90 | {p90:.2f}s |",
        f"| Full answer, median | {statistics.median(totals):.2f}s |",
        "",
        "A retrieval hit means at least one of the four chunks returned contains "
        "an expected keyword. Time to first token is when speech begins, which "
        "is what a listener actually experiences — the full answer arrives later "
        "and is spoken sentence by sentence as it streams.\n",
        "| # | Question | Retrieved | Top score | First token |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(rows, 1):
        out.append(
            f"| {i} | {r['q']} | {'yes' if r['hit'] else 'no'} | "
            f"{r['score']:.2f} | {r['ttft']:.2f}s |"
        )

    (HERE / "results.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\nwrote results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())