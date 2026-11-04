"""
Minimal RAG (Retrieval-Augmented Generation) app using Azure OpenAI
(Azure AI Foundry's /openai/v1 endpoint), plus two extra pipeline modes
used to test noise sensitivity and summarization score:

    answer_with_noise()   answers a question with an irrelevant paragraph
                           mixed into the context, to test whether the
                           answer gets distracted by it
    summarize_source()    summarizes the whole knowledge file, instead of
                           answering a specific question

Needs these environment variables:
    AZURE_OPENAI_ENDPOINT              the full base_url, e.g. https://<your-resource>.services.ai.azure.com/openai/v1
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT  name of your embeddings deployment
    AZURE_OPENAI_CHAT_DEPLOYMENT       name of your chat deployment

Run:
    python rag.py "What is RAG?"
"""

import os
import sys
import math

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

KNOWLEDGE_FILE = os.path.join(os.path.dirname(__file__), "knowledge.txt")


def get_client() -> OpenAI:
    """Build a client pointed at the Azure AI Foundry /openai/v1 endpoint.

    AZURE_OPENAI_ENDPOINT must already include the /openai/v1 path
    (i.e. exactly what Azure's "Get code" sample shows as `base_url`).
    """
    return OpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )


def load_chunks(path: str = KNOWLEDGE_FILE) -> list[str]:
    """Load the knowledge file and split it into paragraph-sized chunks."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks


def load_source_text(path: str = KNOWLEDGE_FILE) -> str:
    """Returns the whole knowledge file as one string (used for summarization)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def embed_text(client: OpenAI, text: str, deployment: str) -> list[float]:
    """Return the embedding vector for a single piece of text."""
    response = client.embeddings.create(input=text, model=deployment)
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_top_chunk(
    query: str,
    chunks: list[str],
    chunk_embeddings: list[list[float]],
    client: OpenAI,
    embed_deployment: str,
) -> str:
    """Return the single chunk whose embedding is closest to the query's embedding."""
    query_embedding = embed_text(client, query, embed_deployment)
    scores = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
    best_index = scores.index(max(scores))
    return chunks[best_index]


def answer_question(query: str) -> tuple[str, str]:
    """Full RAG pipeline. Returns (retrieved_context, answer)."""
    client = get_client()
    embed_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    chunks = load_chunks()
    chunk_embeddings = [embed_text(client, c, embed_deployment) for c in chunks]
    context = retrieve_top_chunk(query, chunks, chunk_embeddings, client, embed_deployment)

    response = client.responses.create(
        model=chat_deployment,
        instructions="Answer the question using only the provided context. "
        "If the context doesn't contain the answer, say you don't know.",
        input=f"Context:\n{context}\n\nQuestion: {query}",
    )
    return context, response.output_text


def answer_with_noise(query: str, noise_paragraph: str) -> tuple[str, str]:
    """Like answer_question(), but mixes an irrelevant noise paragraph into
    the context alongside the real retrieved chunk, to test whether the
    answer gets distracted by it. Returns (relevant_context, answer)."""
    client = get_client()
    embed_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    chunks = load_chunks()
    chunk_embeddings = [embed_text(client, c, embed_deployment) for c in chunks]
    relevant_context = retrieve_top_chunk(query, chunks, chunk_embeddings, client, embed_deployment)

    noisy_context = relevant_context + "\n\n" + noise_paragraph

    response = client.responses.create(
        model=chat_deployment,
        instructions="Answer the question using only the provided context. "
        "If the context doesn't contain the answer, say you don't know.",
        input=f"Context:\n{noisy_context}\n\nQuestion: {query}",
    )
    return relevant_context, response.output_text


def summarize_source() -> tuple[str, str]:
    """Summarizes the whole knowledge file. Returns (source_text, summary)."""
    client = get_client()
    chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    source_text = load_source_text()

    response = client.responses.create(
        model=chat_deployment,
        instructions="Summarize the following text in 2-3 sentences, "
        "covering its key points.",
        input=source_text,
    )
    return source_text, response.output_text


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is RAG?"
    context, answer = answer_question(question)
    print(f"Context used:\n{context}\n")
    print(f"Answer:\n{answer}")
