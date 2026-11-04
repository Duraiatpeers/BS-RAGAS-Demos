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

from rag import get_client, answer_question, load_chunks
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

# A question knowledge.txt doesn't actually cover in detail - it never
# explains claim decomposition, so retrieval should pull back context
# that doesn't contain what's needed. Used to check that context_recall
# can legitimately score LOW, not just high.
HARDER_QUESTION = (
    "How does the real ragas library break an answer into individual "
    "claims when computing faithfulness?"
)
HARDER_REFERENCE_ANSWER = (
    "The real ragas library decomposes an answer into individual factual "
    "claims and checks each claim separately against the retrieved context."
)

# A deliberately fabricated answer with a claim the context doesn't
# support. Used to check that faithfulness_score can legitimately
# score LOW, not just high.
FLAWED_ANSWER = (
    "RAG was invented by Google in 2015 and requires at least 10,000 "
    "GPUs to run in production."
)


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


def test_context_recall_on_harder_question(client, judge_deployment):
    """knowledge.txt never explains claim decomposition, so retrieval for
    this question should come back without what's needed - context recall
    should come out LOW. If this fails, the judge isn't discriminating."""
    context, answer = answer_question(HARDER_QUESTION)
    score = context_recall_score(context, HARDER_REFERENCE_ANSWER, client, judge_deployment)
    print(f"\ncontext recall score (harder question): {score:.2f}")
    assert score < SCORE_THRESHOLD


def test_faithfulness_catches_planted_flaw(client, judge_deployment):
    """Hands faithfulness_score() a fabricated answer that the context
    doesn't support. Score should come out LOW. If this fails, the judge
    isn't discriminating - it's just agreeing with everything."""
    context = load_chunks()[0]  # the paragraph defining RAG
    score = faithfulness_score(FLAWED_ANSWER, context, client, judge_deployment)
    print(f"\nfaithfulness score (planted flaw): {score:.2f}")
    assert score < SCORE_THRESHOLD