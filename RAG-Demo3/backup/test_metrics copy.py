"""
Pytest tests for four RAGAS-style metrics (see metrics.py), run against
the real RAG pipeline in rag.py and a real Azure OpenAI deployment.
No mocking. Scoring is done by an LLM judge, not cosine similarity.

Uses one pytest fixture (rag_result) so the RAG pipeline runs once and
all four tests reuse the same context/answer, instead of each test
re-running the whole pipeline.

Needs a working .env with:
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    AZURE_OPENAI_CHAT_DEPLOYMENT
    AZURE_OPENAI_JUDGE_DEPLOYMENT      optional - defaults to the chat deployment

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


@pytest.fixture(scope="module")
def client():
    return get_client()


@pytest.fixture(scope="module")
def judge_deployment():
    """The model that grades each metric. Point AZURE_OPENAI_JUDGE_DEPLOYMENT
    at a different deployment to use a different LLM as judge than the one
    that generated the answer."""
    return os.environ.get("AZURE_OPENAI_JUDGE_DEPLOYMENT") or os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]


@pytest.fixture(scope="module")
def rag_result():
    """Runs the real RAG pipeline once; shared by all four metric tests."""
    context, answer = answer_question(QUESTION)
    return context, answer


def test_faithfulness(rag_result, client, judge_deployment):
    context, answer = rag_result
    score = faithfulness_score(answer, context, client, judge_deployment)
    print(f"\nfaithfulness score: {score:.2f}")
    assert score > SCORE_THRESHOLD


def test_answer_relevancy(rag_result, client, judge_deployment):
    context, answer = rag_result
    score = answer_relevancy_score(answer, QUESTION, client, judge_deployment)
    print(f"\nanswer relevancy score: {score:.2f}")
    assert score > SCORE_THRESHOLD


def test_context_precision(rag_result, client, judge_deployment):
    context, answer = rag_result
    score = context_precision_score(context, QUESTION, client, judge_deployment)
    print(f"\ncontext precision score: {score:.2f}")
    assert score > SCORE_THRESHOLD


def test_context_recall(rag_result, client, judge_deployment):
    context, answer = rag_result
    score = context_recall_score(context, REFERENCE_ANSWER, client, judge_deployment)
    print(f"\ncontext recall score: {score:.2f}")
    assert score > SCORE_THRESHOLD
