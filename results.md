## Evaluation

20 questions against `_OceanofPDF.com_Hands-On_Large_Language_Models_-_Jay_Alammar.pdf` (1317 chunks, gpt-4o-mini).

| Metric | Value |
| --- | --- |
| Retrieval hit rate | **95%** (19/20) |
| Time to first token, median | **3.27s** |
| Time to first token, p90 | 4.10s |
| Full answer, median | 3.67s |

A retrieval hit means at least one of the four chunks returned contains an expected keyword. Time to first token is when speech begins, which is what a listener actually experiences — the full answer arrives later and is spoken sentence by sentence as it streams.

| # | Question | Retrieved | Top score | First token |
| --- | --- | --- | --- | --- |
| 1 | What is tokenization? | yes | 0.66 | 5.32s |
| 2 | What are token embeddings? | yes | 0.72 | 3.31s |
| 3 | How does self-attention work? | yes | 0.69 | 3.43s |
| 4 | What is the difference between an encoder and a decoder model? | yes | 0.59 | 3.33s |
| 5 | What is a context window? | yes | 0.45 | 4.18s |
| 6 | How does semantic search differ from keyword search? | yes | 0.63 | 3.33s |
| 7 | What is dense retrieval? | yes | 0.56 | 3.35s |
| 8 | What is a reranker used for? | yes | 0.56 | 3.05s |
| 9 | What is retrieval augmented generation? | yes | 0.71 | 3.22s |
| 10 | How does text classification with embeddings work? | yes | 0.70 | 4.10s |
| 11 | What is topic modeling? | yes | 0.72 | 3.30s |
| 12 | What does temperature control during generation? | yes | 0.54 | 3.20s |
| 13 | What is few shot prompting? | yes | 0.58 | 3.12s |
| 14 | What is chain of thought prompting? | yes | 0.64 | 3.09s |
| 15 | What is LoRA? | yes | 0.50 | 3.54s |
| 16 | What is quantization and why is it used? | yes | 0.61 | 3.24s |
| 17 | How is a reward model used in preference tuning? | yes | 0.80 | 3.09s |
| 18 | What is the difference between fine tuning and prompting? | yes | 0.59 | 3.05s |
| 19 | What are sentence transformers used for? | yes | 0.60 | 3.03s |
| 20 | What is the capital of Mongolia? | no | 0.24 | 3.04s |
