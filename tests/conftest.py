"""Fakes used by the tests, so no internet or API key is needed."""

import itertools

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agent.tools import ToolError

_ids = itertools.count(1)


def call(name: str, **args) -> dict:
    """Build one tool call, like the LLM would produce."""
    return {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}


def decide(*calls) -> AIMessage:
    """An LLM reply that asks for these tool calls."""
    return AIMessage(content="", tool_calls=list(calls))


USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


class FakeLLM:
    """Pretends to be Gemini.

    * Agent decisions come from `script` (a list of AIMessages), or from
      `forever` (a function that returns the next decision, for loop tests).
    * The final-answer call (a plain string prompt) returns `answer`.
    """

    def __init__(self, script=None, answer="Fasting helps [1].", forever=None, fail=False):
        self.script = list(script or [])
        self.answer = answer
        self.forever = forever
        self.fail = fail
        self.agent_calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        if self.fail:
            raise ConnectionError("LLM is down")
        if isinstance(prompt, str):  # final answer (or summary) request
            return AIMessage(self.answer, usage_metadata=USAGE)
        self.agent_calls += 1
        if self.forever:
            return self.forever()
        if self.script:
            return self.script.pop(0)
        return AIMessage("DONE", usage_metadata=USAGE)  # no tool calls -> finish


def page(url: str, title: str = "A page") -> dict:
    return {"url": url, "title": title, "text": f"Some useful facts from {url}."}


def fake_tools(search=None, fetch=None, summarise=None) -> dict:
    """Build the three tools from plain Python functions."""

    def default_search(query):
        return {"results": [{"title": "Result", "url": "https://search-only.example/x",
                             "snippet": "a snippet"}]}

    def default_summarise(url, focus):
        return page(url)

    def _search(query: str) -> dict:
        return (search or default_search)(query)

    def _fetch(url: str) -> dict:
        return (fetch or page)(url)

    def _summarise(url: str, focus: str) -> dict:
        return (summarise or default_summarise)(url, focus)

    return {t.name: t for t in [
        StructuredTool.from_function(_search, name="web_search", description="search"),
        StructuredTool.from_function(_fetch, name="fetch_page", description="fetch"),
        StructuredTool.from_function(_summarise, name="summarise_page", description="sum"),
    ]}


def always_fail(message):
    def _fail(*args, **kwargs):
        raise ToolError(message)
    return _fail


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    """Don't really sleep between retries in tests."""
    monkeypatch.setattr("agent.graph.time.sleep", lambda s: None)
