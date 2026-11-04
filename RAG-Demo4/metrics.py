"""
Two more RAGAS-style metrics computed with an LLM as judge:
noise sensitivity and summarization score. Same pattern as the core
4-metric project - the judge gives one short sentence of reasoning,
then a SCORE line, which is more reliable than asking for a bare number.

The judge can be a different deployment than the one used to generate
the answer - set AZURE_OPENAI_JUDGE_DEPLOYMENT in .env to point at it;
it falls back to AZURE_OPENAI_CHAT_DEPLOYMENT if that's not set.

    noise_sensitivity_score   did irrelevant content distract the answer?
    summarization_score       does the summary cover the source's key points?
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
        print(f"[warning] judge model '{judge_model}' gave {score} - "
              f"expected a 0-1 decimal. Clamping to 1.0.")
    return max(0.0, min(1.0, score))


def noise_sensitivity_score(
    answer: str,
    relevant_context: str,
    noise_paragraph: str,
    client: OpenAI,
    judge_model: str,
) -> float:
    """Did the answer stay grounded in the relevant context and ignore an
    irrelevant noise paragraph mixed in alongside it? Score 1 = robust
    (ignored the noise), 0 = distracted (pulled in claims from the noise).
    Proxy for RAGAS noise sensitivity, reported as a robustness score so
    higher is still better, consistent with the other metrics."""
    instructions = (
        "You are checking whether a RAG answer stayed grounded in its relevant "
        "context and ignored an irrelevant noise paragraph that was mixed into "
        "the context it was given. Score 1 if the answer only uses information "
        "from the relevant context and ignores the noise paragraph. Score 0 if "
        "the answer includes claims that come from the noise paragraph instead."
    )
    input_text = (
        f"Relevant context:\n{relevant_context}\n\n"
        f"Noise paragraph (irrelevant, mixed into the context given to the model):\n"
        f"{noise_paragraph}\n\n"
        f"Answer:\n{answer}"
    )
    return _llm_score(client, judge_model, instructions, input_text)


def summarization_score(
    summary: str,
    source_text: str,
    client: OpenAI,
    judge_model: str,
) -> float:
    """Does the summary cover the source's key points, without missing
    important information or adding anything not in the source?
    Proxy for RAGAS summarization score."""
    instructions = (
        "You are grading a summary for quality: does it cover the key points "
        "of the source text, without missing important information and without "
        "adding anything not in the source? Score 1 if it covers the key points "
        "accurately, 0 if it misses most of them or adds unsupported content."
    )
    input_text = f"Source text:\n{source_text}\n\nSummary:\n{summary}"
    return _llm_score(client, judge_model, instructions, input_text)
