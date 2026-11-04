"""
RAGAS-style core metrics computed with an LLM as judge.

Each metric asks the judge model to briefly reason first, then give a
single SCORE line between 0 and 1. Asking for reasoning before the
number (rather than a bare number) makes agreeable/lenient judges
actually discriminate instead of defaulting to a flat 1 every time.

The judge can be a different deployment than the one used to generate
the answer - set AZURE_OPENAI_JUDGE_DEPLOYMENT in .env to point at it;
it falls back to AZURE_OPENAI_CHAT_DEPLOYMENT if that's not set. This
is still a simplified, teaching version of RAGAS's real metrics (which
break answers into individual claims and do more careful prompting),
but closer in spirit than a plain cosine-similarity proxy.

    faithfulness_score        is the answer supported by the context?
    answer_relevancy_score    does the answer address the question?
    context_precision_score   is the context relevant to the question?
    context_recall_score      does the context cover the reference answer?
"""

import re

from openai import OpenAI


def _llm_score(client: OpenAI, judge_model: str, instructions: str, input_text: str) -> float:
    """Ask the judge model to reason briefly, then give a 0-1 score, and parse it out."""
    response = client.responses.create(
        model=judge_model,
        instructions=instructions
        + " First give one short sentence of reasoning. Then, on its own line, "
          "respond with exactly: SCORE: <a number between 0 and 1>.",
        input=input_text,
    )
    text = response.output_text.strip()
    print(f"[{judge_model}] {text}")

    match = re.search(r"score\s*:?\s*(\d*\.?\d+)", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Judge model did not return a SCORE line: {text!r}")

    score = float(match.group(1))
    if score > 1.0:
        # The judge likely answered on a different scale (e.g. "9/10", "95%")
        # instead of 0-1. Surface the raw reply instead of silently clamping it.
        print(f"[warning] judge model '{judge_model}' gave {score} - "
              f"expected a 0-1 decimal. Clamping to 1.0.")
    return max(0.0, min(1.0, score))


def faithfulness_score(answer: str, context: str, client: OpenAI, judge_model: str) -> float:
    """Is every claim in the answer actually supported by the context?"""
    instructions = (
        "You are grading a RAG answer for faithfulness: whether its claims "
        "are supported by the given context, with no made-up facts. "
        "Score 1 if fully supported, 0 if entirely unsupported."
    )
    input_text = f"Context:\n{context}\n\nAnswer:\n{answer}"
    return _llm_score(client, judge_model, instructions, input_text)


def answer_relevancy_score(answer: str, question: str, client: OpenAI, judge_model: str) -> float:
    """Does the answer actually address the question that was asked?"""
    instructions = (
        "You are grading a RAG answer for relevancy: whether it actually "
        "addresses the question asked, without going off-topic. "
        "Score 1 if fully relevant, 0 if it doesn't address the question."
    )
    input_text = f"Question:\n{question}\n\nAnswer:\n{answer}"
    return _llm_score(client, judge_model, instructions, input_text)


def context_precision_score(context: str, question: str, client: OpenAI, judge_model: str) -> float:
    """Is the retrieved context relevant to answering the question?"""
    instructions = (
        "You are grading retrieved context for precision: whether it is "
        "relevant to answering the question, without irrelevant content. "
        "Score 1 if highly relevant, 0 if irrelevant."
    )
    input_text = f"Question:\n{question}\n\nContext:\n{context}"
    return _llm_score(client, judge_model, instructions, input_text)


def context_recall_score(context: str, reference_answer: str, client: OpenAI, judge_model: str) -> float:
    """Does the context contain what was needed to produce the reference answer?"""
    instructions = (
        "You are grading retrieved context for recall: whether it contains "
        "the information needed to produce the reference answer below. "
        "Score 1 if all needed information is present, 0 if none of it is."
    )
    input_text = f"Reference answer:\n{reference_answer}\n\nContext:\n{context}"
    return _llm_score(client, judge_model, instructions, input_text)