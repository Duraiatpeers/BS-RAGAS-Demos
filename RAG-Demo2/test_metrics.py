"""
Run:
    pytest test_metrics.py -v -s
"""

import os

import pytest

from rag import get_client, answer_question
from metrics import (
    faithfulness_score,
    answer_relevancy_score,
    context_precision_score,
    context_recall_score,
)

QUESTION = "What is RAG?"
REFERENCE_ANSWER = (
    "RAG combines retrieving relevant text from a knowledge source with a "
    "language model, so the model can generate an answer grounded in that text."
)
SCORE_THRESHOLD = 0.5  
# loose bar - these are similarity proxies, not calibrated scores


@pytest.fixture(scope="module")
def client():
    return get_client()


@pytest.fixture(scope="module")
def embed_deployment():
    return os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]


@pytest.fixture(scope="module")
def rag_result():
    """Runs the real RAG pipeline once; shared by all four metric tests."""
    context, answer = answer_question(QUESTION)
    return context, answer


def test_faithfulness(rag_result, client, embed_deployment):
    context, answer = rag_result
    score = faithfulness_score(answer, context, client, embed_deployment)
    print(f"\nfaithfulness score: {score:.2f}")
    assert score > SCORE_THRESHOLD


def test_answer_relevancy(rag_result, client, embed_deployment):
    context, answer = rag_result
    score = answer_relevancy_score(answer, QUESTION, client, embed_deployment)
    print(f"\nanswer relevancy score: {score:.2f}")
    assert score > SCORE_THRESHOLD


def test_context_precision(rag_result, client, embed_deployment):
    context, answer = rag_result
    score = context_precision_score(context, QUESTION, client, embed_deployment)
    print(f"\ncontext precision score: {score:.2f}")
    assert score > SCORE_THRESHOLD


def test_context_recall(rag_result, client, embed_deployment):
    context, answer = rag_result
    score = context_recall_score(context, REFERENCE_ANSWER, client, embed_deployment)
    print(f"\ncontext recall score: {score:.2f}")
    assert score > SCORE_THRESHOLD
