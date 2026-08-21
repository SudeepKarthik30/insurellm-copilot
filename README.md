---
title: Insurellm RAG Assistant
sdk: gradio
sdk_version: 5.47.2
app_file: app.py
pinned: false
---

# Insurellm Assistant

A retrieval-augmented generation (RAG) system that answers questions over a
company knowledge base, with a companion evaluation dashboard for measuring
retrieval and answer quality.

## What this is

Two things live in this repo:

- **`app.py`** — a chat interface. Ask a question, get an answer grounded in
  retrieved knowledge-base documents, and see exactly which chunks were used
  and how relevant each one was scored.
- **`evaluator.py`** — a batch evaluation dashboard that runs a fixed set of
  test questions (`evaluation/tests.jsonl`) through the pipeline and reports
  retrieval metrics (MRR, nDCG, keyword coverage) and answer-quality metrics
  (accuracy, completeness, relevance, scored by an LLM judge), broken down by
  question category.

## Architecture

```
User question
     |
     v
Query rewrite (llama-3.1-8b-instant)  --  turns the question into a short,
     |                                     specific search query
     v
Dual retrieval (Chroma / all-MiniLM-L6-v2)
  - search on the original question
  - search on the rewritten query
  - merge, de-duplicate
     |
     v
Local reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
  - scores every candidate chunk against the original question
  - no API call, no rate limit, runs on CPU
     |
     v
Top-K chunks -> final answer (llama-3.3-70b-versatile)
```

Two models are used deliberately for different reasons: a fast/cheap model
for the mechanical query-rewrite step, and the strongest available model for
the answer itself, since that's the part actually being judged.

Reranking used to be done by asking an LLM to output a ranked list of chunk
IDs. That approach burned API tokens fast, was the main source of rate-limit
failures during evaluation, and occasionally hallucinated chunk IDs outside
the valid range. Replacing it with a local cross-encoder removed all three
problems at once — it's a small model purpose-trained to score
(query, passage) relevance pairs, so it produces a plain number instead of
free text that has to be parsed and trusted.

## Evaluation results

150 test questions across 7 categories (direct fact, temporal, comparative,
numerical, relationship, spanning, holistic):

| Metric | Score |
|---|---|
| Mean Reciprocal Rank (MRR) | 0.87 |
| Normalized DCG (nDCG) | 0.85 |
| Keyword coverage | 91.5% |
| Answer accuracy | 4.31 / 5 |
| Answer completeness | 4.33 / 5 |
| Answer relevance | 4.05 / 5 |

Per-category retrieval performance is not uniform — direct-fact and
numerical questions score highest, while comparative and holistic questions
(which require synthesizing information across multiple chunks) score
lower. This is expected: multi-hop synthesis is a harder retrieval problem
than single-fact lookup, and the gap narrowed substantially after increasing
chunk size and switching to cross-encoder reranking, but did not fully
close. See "Known limitations" below.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your API key

Copy `.env.example` to `.env` and add a Groq API key:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Add your knowledge base

Place markdown documents under `knowledge-base/<category>/*.md`. Each
top-level folder under `knowledge-base/` becomes a `doc_type` tag in the
vector store's metadata.

### 4. Run

```bash
python app.py
```

The vector store is built automatically from `knowledge-base/` the first
time you run the app (or `ingest.py`) — there's no separate manual ingest
step required. To force a full rebuild after changing the knowledge base:

```bash
python implementation/ingest.py
```

### 5. Run the evaluation dashboard (optional)

```bash
python evaluator.py
```

Requires `evaluation/tests.jsonl` — a JSONL file of test questions, each
with `question`, `keywords`, `reference_answer`, and `category` fields.

## Deploying to Hugging Face Spaces

1. Create a new Space, SDK: Gradio.
2. Push this repo (including `knowledge-base/`, excluding `vector_db/` —
   see `.gitignore`).
3. Add `GROQ_API_KEY` as a Space secret (Settings → Repository secrets).
4. The Space will build the vector store automatically on first launch.

## Project structure

```
.
├── app.py                     # Chat UI (main entry point)
├── evaluator.py                # Evaluation dashboard
├── implementation/
│   ├── answer.py               # Retrieval + rerank + answer pipeline
│   └── ingest.py                # Knowledge base -> vector store
├── evaluation/
│   ├── eval.py                  # Retrieval & answer-quality metrics
│   ├── test.py                   # Test question schema/loader
│   └── tests.jsonl                # Test question set (not included — add your own)
├── knowledge-base/              # Source markdown documents (not included — add your own)
├── vector_db/                   # Auto-generated, gitignored
├── requirements.txt
└── .env.example
```

## Known limitations

- **Comparative and holistic-category questions score lower than
  direct-fact questions.** These require the model to reason across
  multiple chunks rather than retrieve a single fact. A further improvement
  not yet implemented here is query decomposition — splitting a multi-part
  question into sub-queries before retrieval.
- **The LLM judge used for answer-quality scoring is not perfectly
  deterministic.** Its JSON output is parsed defensively (see
  `evaluation/eval.py`) to handle formatting inconsistencies, but scores
  should be read as directionally reliable rather than exact.
- **Evaluation sample size per category is moderate** (roughly 20 questions
  per category out of 150 total). Category-level scores are useful for
  spotting patterns, not for fine-grained statistical claims.
- **This is a single-user demo**, not built for concurrent production
  traffic — no request queuing, no persistent conversation storage beyond
  the current session.

## Tech stack

Gradio · LangChain · ChromaDB · Groq (Llama 3.1 / 3.3) · HuggingFace
sentence-transformers (embeddings + cross-encoder reranking) · Pydantic
