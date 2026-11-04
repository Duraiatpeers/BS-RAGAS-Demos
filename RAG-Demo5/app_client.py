"""Thin HTTP client for the deployed PX Service Assistant app (see
px-rag-app/DEPLOY.md). Same contract smoke_test.py uses - this just
wraps it so the tests read cleanly.

Needs in .env:
    PX_URL      e.g. https://boschapp-l2.azurewebsites.net
    PX_KEY      the training API key (X-API-Key header)
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ["PX_URL"].rstrip("/")
API_KEY = os.environ["PX_KEY"]


def ask(question: str, prompt_version: str = "v1", **kwargs) -> dict:
    """Calls the live /ask endpoint. Returns the full JSON response:
    answer, citations, chunks, tool_calls, session, settings, latency_ms."""
    response = requests.post(
        f"{BASE_URL}/ask",
        headers={"X-API-Key": API_KEY},
        json={"question": question, "prompt_version": prompt_version, **kwargs},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def health() -> dict:
    response = requests.get(f"{BASE_URL}/health", timeout=30)
    response.raise_for_status()
    return response.json()
