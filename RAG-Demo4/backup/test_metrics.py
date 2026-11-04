"""
Pytest tests for two more RAGAS-style metrics (see metrics.py): noise
sensitivity and summarization score. Run against the real rag.py
pipeline and a real Azure OpenAI deployment. No mocking.

For each metric there are two tests: one where the pipeline should
genuinely score well, and one with a deliberately bad input where the
score should come out low - the second kind checks the judge is
actually discriminating, not just agreeing with everything.

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

from rag import get_client, answer_with_noise, summarize_source, load_source_text, load_chunks
from metrics import noise_sensitivity_score, summarization_score

QUESTION = "What is RAG?"

# An unrelated paragraph deliberately mixed into the context to test
# whether the answer gets distracted by it.
NOISE_PARAGRAPH = (
    "The capital of Australia is Canberra, not Sydney. Canberra was chosen "
    "as a compromise between Sydney and Melbourne in the early 1900s."
)

# A hand-crafted answer that pulls in the noise paragraph's content -
# used to check noise_sensitivity_score can legitimately score LOW.
CONTAMINATED_ANSWER = (
    "RAG retrieves relevant text to ground answers. By the way, the capital "
    "of Australia is Canberra, which was chosen as a compromise between "
    "Sydney and Melbourne."
)

# A hand-crafted bad summary that misses the source's key points - used
# to check summarization_score can legitimately score LOW.
BAD_SUMMARY = "RAG is a programming language used to build websites."

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


def test_noise_sensitivity(client, judge_deployment):
    """The question is clearly about RAG; an unrelated noise paragraph is
    mixed into the context. The real pipeline's answer should ignore it,
    so noise sensitivity should come out HIGH (robust)."""
    relevant_context, answer = answer_with_noise(QUESTION, NOISE_PARAGRAPH)
    score = noise_sensitivity_score(answer, relevant_context, NOISE_PARAGRAPH, client, judge_deployment)
    print(f"\nnoise sensitivity score: {score:.2f}")
    assert score > SCORE_THRESHOLD


def test_noise_sensitivity_catches_contaminated_answer(client, judge_deployment):
    """Hands the metric a hand-crafted answer that pulls in the noise
    paragraph's content. Score should come out LOW. If this fails, the
    judge isn't discriminating."""
    relevant_context = load_chunks()[0]
    score = noise_sensitivity_score(CONTAMINATED_ANSWER, relevant_context, NOISE_PARAGRAPH, client, judge_deployment)
    print(f"\nnoise sensitivity score (contaminated answer): {score:.2f}")
    assert score < SCORE_THRESHOLD


def test_summarization_score(client, judge_deployment):
    """Summarizes the real knowledge file with the real pipeline. The
    summary should cover its key points, so the score should come out HIGH."""
    source_text, summary = summarize_source()
    score = summarization_score(summary, source_text, client, judge_deployment)
    print(f"\nsummarization score: {score:.2f}")
    assert score > SCORE_THRESHOLD


def test_summarization_score_catches_bad_summary(client, judge_deployment):
    """Hands the metric a hand-crafted summary that's wrong and misses
    everything in the source. Score should come out LOW. If this fails,
    the judge isn't discriminating."""
    source_text = load_source_text()
    score = summarization_score(BAD_SUMMARY, source_text, client, judge_deployment)
    print(f"\nsummarization score (bad summary): {score:.2f}")
    assert score < SCORE_THRESHOLD
