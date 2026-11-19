"""Settings for the agent, read from environment variables / the .env file."""

import os

from dotenv import load_dotenv

# Read the .env file (if there is one) into environment variables.
load_dotenv()

# Which Gemini model to use. Change it in .env with GEMINI_MODEL=...
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# The hard limit: the agent may make at most this many tool calls.
DEFAULT_MAX_STEPS = 8

# How long to wait for a web page before giving up (seconds).
HTTP_TIMEOUT = 15

# How many characters of each page we keep. Pages can be huge, and we only
# need enough text to answer the question (this also saves tokens).
MAX_PAGE_CHARS = 6000

# How long to wait before retrying a failed tool (seconds).
RETRY_DELAY = 1.0


def make_llm(model_name: str | None = None):
    """Create the Gemini chat model. Fails with a clear message if no key is set."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # Imported here so tests (which use a fake LLM) never need this package set up.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model_name or MODEL_NAME, temperature=0)
