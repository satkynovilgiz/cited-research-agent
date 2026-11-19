"""Settings for the agent, read from environment variables / the .env file."""

import os
import warnings

from dotenv import load_dotenv

# Read the .env file (if there is one) into environment variables.
load_dotenv()

# Which Gemini model to use. Change it in .env with GEMINI_MODEL=...
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# The hard limit: the agent may make at most this many tool calls.
DEFAULT_MAX_STEPS = 8

# How long to wait for a web page before giving up (seconds).
HTTP_TIMEOUT = 15

# How many characters of each page we keep. Pages can be huge, and we only
# need enough text to answer the question (this also saves tokens).
MAX_PAGE_CHARS = 6000

# How long to wait for a search engine (seconds).
SEARCH_TIMEOUT = 10

# How long to wait before the first retry (seconds). Each further retry waits
# twice as long as the one before (1s, 2s, ...) - this is called "backoff".
RETRY_DELAY = 1.0

# How many times to retry each tool after a failure. Search gets an extra
# retry because DuckDuckGo sometimes blocks a request for a few seconds.
RETRIES = {"web_search": 2}
DEFAULT_RETRIES = 1

# The agent should read at least this many pages before answering. If it tries
# to stop with fewer (and has steps left), it is reminded once.
MIN_SOURCES = 2

# The LLM: give up on one call after this many seconds, and retry it at most
# this many times. Without this, a busy model ("503 high demand") is retried
# quietly for minutes.
LLM_TIMEOUT = 60
LLM_RETRIES = 2


def make_llm(model_name: str | None = None):
    """Create the Gemini chat model. Fails with a clear message if no key is set."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # Imported here so tests (which use a fake LLM) never need this package set up.
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Some Gemini models ignore temperature and print a warning about it - hide it.
    warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

    return ChatGoogleGenerativeAI(
        model=model_name or MODEL_NAME,
        temperature=0,        # same question -> (nearly) same answer; good for evaluation
        timeout=LLM_TIMEOUT,  # never wait forever for a busy model
        max_retries=LLM_RETRIES,
    )
