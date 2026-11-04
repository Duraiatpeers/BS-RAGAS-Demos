"""
Simplified, teaching versions of RAGAS's four core metrics.

The real `ragas` library computes these with more sophisticated
techniques (breaking answers into individual claims, generating
multiple hypothetical questions, using an LLM as a judge). To keep
this simple, every metric here is reduced to one idea: embed two
pieces of text with the same embedding model used by the RAG
pipeline, then measure how close they are with cosine similarity.
A score near 1.0 means "very similar in meaning", near 0.0 means
"unrelated".

    faithfulness_score        answer <-> context
    answer_relevancy_score    answer <-> question
    context_precision_score   context <-> question
    context_recall_score      context <-> reference answer
"""

from openai import OpenAI

from rag import embed_text, cosine_similarity


def faithfulness_score(answer: str, context: str, client: OpenAI, embed_deployment: str) -> float:
    """Is the answer grounded in the retrieved context?
    Proxy for RAGAS faithfulness."""
    answer_embedding = embed_text(client, answer, embed_deployment)
    context_embedding = embed_text(client, context, embed_deployment)
    return cosine_similarity(answer_embedding, context_embedding)


def answer_relevancy_score(answer: str, question: str, client: OpenAI, embed_deployment: str) -> float:
    """Does the answer actually address the question?
    Proxy for RAGAS answer relevancy."""
    answer_embedding = embed_text(client, answer, embed_deployment)
    question_embedding = embed_text(client, question, embed_deployment)
    return cosine_similarity(answer_embedding, question_embedding)


def context_precision_score(context: str, question: str, client: OpenAI, embed_deployment: str) -> float:
    """Is the retrieved context relevant to the question?
    Proxy for RAGAS context precision."""
    context_embedding = embed_text(client, context, embed_deployment)
    question_embedding = embed_text(client, question, embed_deployment)
    return cosine_similarity(context_embedding, question_embedding)


def context_recall_score(context: str, reference_answer: str, client: OpenAI, embed_deployment: str) -> float:
    """Does the retrieved context contain what was needed to produce
    the known-good reference answer? Proxy for RAGAS context recall."""
    context_embedding = embed_text(client, context, embed_deployment)
    reference_embedding = embed_text(client, reference_answer, embed_deployment)
    return cosine_similarity(context_embedding, reference_embedding)
