import os
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "vector_db")

wait = wait_exponential(multiplier=1, min=5, max=60)
stop = stop_after_attempt(3)

FAST_MODEL = "llama-3.1-8b-instant"
MAIN_MODEL = "llama-3.3-70b-versatile"

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

fast_llm = ChatOpenAI(
    model=FAST_MODEL,
    temperature=0,
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)

llm = ChatOpenAI(
    model=MAIN_MODEL,
    temperature=0,
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _ensure_vector_db():
    """
    Build the vector store from the knowledge base if it doesn't exist yet,
    or is empty. This makes the app self-sufficient on a fresh checkout or a
    fresh Hugging Face Spaces container, where vector_db won't exist until
    the first run.
    """
    needs_build = not os.path.exists(DB_NAME)

    if not needs_build:
        try:
            existing = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
            needs_build = existing._collection.count() == 0
        except Exception:
            needs_build = True

    if needs_build:
        print("[answer] vector store missing or empty — building from knowledge-base...", flush=True)
        from implementation.ingest import fetch_documents, create_chunks, create_embeddings
        documents = fetch_documents()
        chunks = create_chunks(documents)
        create_embeddings(chunks)
        print("[answer] vector store build complete", flush=True)


_ensure_vector_db()
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

# Local reranker: no API calls, no rate limits, purpose-built for scoring
# (query, passage) relevance. Downloads once (~80MB) then runs on CPU.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

RETRIEVAL_K = 10
FINAL_K = 8

SYSTEM_PROMPT = """
You are a knowledgeable assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Answer only what is asked. Be accurate and complete, but do not add
information beyond what the question requires.
If you don't know the answer, say so.

Context from the knowledge base:
{context}

Using only this context, answer the user's question.
"""


def rerank(question, chunks):
    """
    Score every chunk against the question using a local cross-encoder and
    return chunks sorted best-to-worst. Each chunk's relevance score is
    attached to its metadata so callers (e.g. the UI) can display it.
    """
    print(f"[rerank] starting, {len(chunks)} chunks to rank...", flush=True)
    if not chunks:
        print("[rerank] no chunks to rank", flush=True)
        return []

    pairs = [[question, chunk.page_content] for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)

    for chunk, score in ranked:
        chunk.metadata["relevance_score"] = float(score)

    print(
        f"[rerank] success, top score: {ranked[0][1]:.3f}, bottom score: {ranked[-1][1]:.3f}",
        flush=True,
    )
    return [chunk for chunk, score in ranked]


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata.get('source', 'unknown')}:\n{chunk.page_content}"
        for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait, stop=stop)
def rewrite_query(question, history=[]):
    """Rewrite the user's question into a short, specific search query."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

Conversation history so far:
{history}

Current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    print(f"[rewrite_query] starting for: {question}", flush=True)
    try:
        response = fast_llm.invoke([{"role": "system", "content": message}])
        print(f"[rewrite_query] success: {response.content}", flush=True)
    except Exception as e:
        print(f"[rewrite_query] ERROR: {type(e).__name__}: {e}", flush=True)
        raise
    return response.content


def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question):
    print(f"[fetch_context_unranked] searching for: {question}", flush=True)
    results = vectorstore.similarity_search(question, k=RETRIEVAL_K)
    print(f"[fetch_context_unranked] found {len(results)} chunks", flush=True)
    return results


def fetch_context(original_question):
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    print(f"[fetch_context] merged total: {len(chunks)} chunks, sending to rerank...", flush=True)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]


@retry(wait=wait, stop=stop)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """Answer a question using RAG and return the answer and the retrieved context."""
    print(f"[answer_question] question: {question}", flush=True)
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    print("[answer_question] calling final LLM for answer...", flush=True)
    try:
        response = llm.invoke(messages)
        print("[answer_question] success", flush=True)
    except Exception as e:
        print(f"[answer_question] ERROR: {type(e).__name__}: {e}", flush=True)
        raise
    return response.content, chunks


if __name__ == "__main__":
    q = "Who founded Insurellm?"
    ans, ctx = answer_question(q)
    print(f"\nQuestion: {q}")
    print(f"\nAnswer: {ans}")
    print(f"\nChunks used: {len(ctx)}")
