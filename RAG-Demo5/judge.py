"""Our own Azure OpenAI client, used only to GRADE the live app's answers.
Separate from the app's own model config in px-rag-app/app/llm.py - this
talks to the Azure AI Foundry /openai/v1 endpoint the same way the other
ragas-* demo projects in this series do.

Needs in .env:
    AZURE_OPENAI_ENDPOINT              must already include /openai/v1
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_CHAT_DEPLOYMENT
    AZURE_OPENAI_JUDGE_DEPLOYMENT       optional - defaults to the chat deployment
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )


def get_judge_deployment() -> str:
    return os.environ.get("AZURE_OPENAI_JUDGE_DEPLOYMENT") or os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
