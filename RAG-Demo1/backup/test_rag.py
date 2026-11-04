"""
Pytest test for rag.py — calls the real Azure OpenAI embeddings deployment.

Needs a working .env (or environment variables) with:
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_API_VERSION
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT

Run:
    pytest test_rag.py -v
"""

import os

from rag import embed_text, get_client, load_chunks, retrieve_top_chunk


def test_retrieve_top_chunk_returns_most_similar_chunk():
    chunks = load_chunks()
    client = get_client()
    embed_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

    chunk_embeddings = [embed_text(client, c, embed_deployment) for c in chunks]

    result = retrieve_top_chunk(
        query="What is cosine similarity used for?",
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        client=client,
        embed_deployment=embed_deployment,
    )

    assert "Cosine similarity" in result