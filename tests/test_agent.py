"""Tests for the research agent. Run with:  python -m pytest -v"""

import httpx
import pytest

from agent import run_agent
from agent.citations import check_citations
from agent.tools import ToolError, fetch_page
from tests.conftest import FakeLLM, always_fail, call, decide, fake_tools

Q = "What are the health effects of intermittent fasting?"


# ---------------------------------------------------------------------------
# Requirement 2: hard step limit
# ---------------------------------------------------------------------------
def test_step_limit_stops_an_endless_loop():
    # This LLM would search forever if we let it.
    llm = FakeLLM(forever=lambda: decide(call("web_search", query="fasting")))
    result = run_agent(Q, max_steps=3, llm=llm, tools=fake_tools())

    assert result["steps"] == 3
    assert len(result["tool_calls"]) == 3
    assert result["limit_reached"] is True
    assert "may be incomplete" in result["answer"]


def test_step_limit_applies_to_parallel_tool_calls():
    # One reply asking for 5 tools at once must still only run 2.
    llm = FakeLLM(script=[decide(*[call("fetch_page", url=f"https://s{i}.org")
                                   for i in range(5)])])
    result = run_agent(Q, max_steps=2, llm=llm, tools=fake_tools())

    assert result["steps"] == 2
    assert [s["url"] for s in result["sources"]] == ["https://s0.org", "https://s1.org"]
    assert result["limit_reached"] is True


def test_no_limit_note_when_agent_finishes_early():
    llm = FakeLLM(script=[decide(call("fetch_page", url="https://a.org"))])
    result = run_agent(Q, max_steps=8, llm=llm, tools=fake_tools())

    assert result["steps"] == 1
    assert result["limit_reached"] is False
    assert "may be incomplete" not in result["answer"]


# ---------------------------------------------------------------------------
# Requirement 3: citations only use fetched URLs
# ---------------------------------------------------------------------------
def test_citations_only_point_to_fetched_pages():
    def fetch(url):
        if "broken" in url:
            raise ToolError("HTTP 404")
        return {"url": url, "title": "Good page", "text": "facts"}

    llm = FakeLLM(
        script=[
            decide(call("web_search", query="fasting")),
            decide(call("fetch_page", url="https://good.org/a"),
                   call("fetch_page", url="https://broken.org/b")),
        ],
        # The LLM "hallucinates" citations [2] and [7].
        answer="Fasting can aid weight loss [1]. It cures everything [2]. Also [7].",
    )
    result = run_agent(Q, llm=llm, tools=fake_tools(fetch=fetch))

    fetched_urls = {s["url"] for s in result["sources"]}
    assert fetched_urls == {"https://good.org/a"}       # the 404 page is not a source
    assert "https://search-only.example/x" not in fetched_urls  # search hits aren't sources
    assert "[1]" in result["answer"]
    assert "[2]" not in result["answer"] and "[7]" not in result["answer"]
    assert result["citation_check"]["removed_ids"] == [2, 7]
    # Every cited source is a fetched source.
    cited = [s for s in result["sources"] if s["cited"]]
    assert cited and all(s["url"] in fetched_urls for s in cited)


def test_same_url_is_not_fetched_twice():
    calls = []

    def fetch(url):
        calls.append(url)
        return {"url": url, "title": "t", "text": "x"}

    llm = FakeLLM(script=[decide(call("fetch_page", url="https://a.org")),
                          decide(call("fetch_page", url="https://a.org"))])
    result = run_agent(Q, llm=llm, tools=fake_tools(fetch=fetch))

    assert calls == ["https://a.org"]
    assert len(result["sources"]) == 1


def test_check_citations_unit():
    sources = [{"id": 1}, {"id": 2}]
    out = check_citations("A is true [1][3]. B is true [2]. C has no citation at all here.",
                          sources)
    assert out["answer"] == "A is true [1]. B is true [2]. C has no citation at all here."
    assert out["cited_ids"] == [1, 2]
    assert out["removed_ids"] == [3]
    assert out["uncited_sentences"] == 1


# ---------------------------------------------------------------------------
# Requirement 4: tool failures are handled gracefully
# ---------------------------------------------------------------------------
def test_failed_tool_is_retried_once_then_succeeds():
    attempts = []

    def flaky_fetch(url):
        attempts.append(url)
        if len(attempts) == 1:
            raise ToolError("timed out")
        return {"url": url, "title": "t", "text": "x"}

    llm = FakeLLM(script=[decide(call("fetch_page", url="https://a.org"))])
    result = run_agent(Q, llm=llm, tools=fake_tools(fetch=flaky_fetch))

    assert len(attempts) == 2
    assert result["tool_calls"][0]["success"] is True
    assert result["tool_calls"][0]["attempts"] == 2
    assert len(result["sources"]) == 1


def test_tool_that_keeps_failing_is_reported_and_agent_moves_on():
    def fetch(url):
        if "dead" in url:
            raise ToolError("HTTP 404")
        return {"url": url, "title": "t", "text": "x"}

    llm = FakeLLM(script=[decide(call("fetch_page", url="https://dead.org")),
                          decide(call("fetch_page", url="https://alive.org"))])
    result = run_agent(Q, llm=llm, tools=fake_tools(fetch=fetch))

    first, second = result["tool_calls"]
    assert first["success"] is False and first["attempts"] == 2
    assert first["error"] == "HTTP 404"
    assert second["success"] is True
    assert len(result["errors"]) == 2  # two failed attempts on the dead page
    assert [s["url"] for s in result["sources"]] == ["https://alive.org"]


def test_empty_search_and_no_sources_gives_honest_answer():
    llm = FakeLLM(script=[decide(call("web_search", query="zzzz"))], answer="Made up [1].")
    result = run_agent(Q, llm=llm,
                       tools=fake_tools(search=always_fail("search returned no results"),
                                        wiki=always_fail("wikipedia: no results")))

    assert result["sources"] == []
    assert "couldn't find" in result["answer"]
    assert "[1]" not in result["answer"]
    assert "search returned no results" in result["tool_calls"][0]["error"]


def test_unexpected_exception_in_tool_does_not_crash():
    def buggy(url):
        raise ValueError("something odd")

    llm = FakeLLM(script=[decide(call("fetch_page", url="https://a.org"))])
    result = run_agent(Q, llm=llm, tools=fake_tools(fetch=buggy))
    assert result["tool_calls"][0]["success"] is False
    assert "ValueError" in result["tool_calls"][0]["error"]


def test_unknown_tool_and_bad_arguments_do_not_crash():
    llm = FakeLLM(script=[decide(call("delete_files", path="/")),
                          decide(call("fetch_page"))])  # missing url argument
    result = run_agent(Q, llm=llm, tools=fake_tools())
    assert [c["success"] for c in result["tool_calls"]] == [False, False]


def test_llm_failure_does_not_crash():
    result = run_agent(Q, llm=FakeLLM(fail=True), tools=fake_tools())
    assert result["answer"]
    assert result["errors"][0]["where"] == "agent_llm"


def test_result_has_all_fields_for_evaluation():
    llm = FakeLLM(script=[decide(call("fetch_page", url="https://a.org"))])
    result = run_agent(Q, llm=llm, tools=fake_tools())
    for key in ("answer", "sources", "steps", "tool_calls", "errors", "token_usage"):
        assert key in result
    assert result["token_usage"]["total_tokens"] > 0
    step = result["tool_calls"][0]
    assert {"step", "tool", "input", "success", "attempts", "error"} <= step.keys()


# ---------------------------------------------------------------------------
# The real fetch_page tool, with the network mocked
# ---------------------------------------------------------------------------
def fake_response(status=200, html="", content_type="text/html"):
    return httpx.Response(status, text=html, headers={"content-type": content_type},
                          request=httpx.Request("GET", "https://x.org"))


def test_fetch_page_extracts_clean_text(monkeypatch):
    html = ("<html><head><title>Fasting</title><script>evil()</script></head><body>"
            "<nav>Home About Contact Login Menu</nav><article>"
            + "<p>Intermittent fasting may improve insulin sensitivity in adults.</p>" * 10
            + "</article></body></html>")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: fake_response(html=html))
    out = fetch_page("https://x.org")
    assert out["title"] == "Fasting"
    assert "insulin sensitivity" in out["text"]
    assert "evil" not in out["text"] and "Login" not in out["text"]


@pytest.mark.parametrize("response, message", [
    (fake_response(404), "HTTP 404"),
    (fake_response(403), "blocked"),
    (fake_response(html="<p>hi</p>"), "empty or blocked"),
    (fake_response(content_type="application/pdf"), "unsupported content type"),
])
def test_fetch_page_errors(monkeypatch, response, message):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: response)
    with pytest.raises(ToolError, match=message):
        fetch_page("https://x.org")


def test_fetch_page_timeout(monkeypatch):
    def slow(*a, **k):
        raise httpx.ReadTimeout("slow")
    monkeypatch.setattr(httpx, "get", slow)
    with pytest.raises(ToolError, match="timed out"):
        fetch_page("https://x.org")


# ---------------------------------------------------------------------------
# Bonus: the API endpoint
# ---------------------------------------------------------------------------
def test_api_ask(monkeypatch):
    from fastapi.testclient import TestClient

    from agent import api

    fake = {"answer": "A [1].", "sources": [{"id": 1, "url": "https://a.org"}],
            "tool_calls": [{"step": 1}]}
    monkeypatch.setattr(api, "run_agent", lambda q, max_steps: fake)
    resp = TestClient(api.app).post("/ask", json={"question": "Is tea healthy?"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": "A [1].", "sources": fake["sources"],
                           "steps": fake["tool_calls"]}


def test_web_search_gets_two_retries_with_backoff(monkeypatch):
    waits, attempts = [], []
    monkeypatch.setattr("agent.graph.time.sleep", waits.append)

    def flaky_search(query):
        attempts.append(query)
        if len(attempts) < 3:
            raise ToolError("search failed: No results found.")
        return {"results": [{"title": "t", "url": "https://a.org", "snippet": "s"}]}

    llm = FakeLLM(script=[decide(call("web_search", query="fasting"))])
    result = run_agent(Q, llm=llm, tools=fake_tools(search=flaky_search), retry_delay=1)

    assert len(attempts) == 3                 # first try + 2 retries
    assert waits == [1, 2]                    # backoff: wait doubles
    assert result["tool_calls"][0]["success"] is True


# ---------------------------------------------------------------------------
# The real wikipedia_search tool, with the network mocked
# ---------------------------------------------------------------------------
def wiki_response(hits):
    return httpx.Response(200, json={"query": {"search": hits}},
                          request=httpx.Request("GET", "https://en.wikipedia.org"))


def test_wikipedia_search_parses_results(monkeypatch):
    from agent.tools import wikipedia_search

    hits = [{"title": "Intermittent fasting",
             "snippet": '<span class="searchmatch">Intermittent</span> fasting is...'}]
    monkeypatch.setattr(httpx, "get", lambda *a, **k: wiki_response(hits))
    out = wikipedia_search("intermittent fasting")["results"]
    assert out == [{"title": "Intermittent fasting",
                    "url": "https://en.wikipedia.org/wiki/Intermittent_fasting",
                    "snippet": "Intermittent fasting is..."}]


def test_wikipedia_search_no_results(monkeypatch):
    from agent.tools import wikipedia_search

    monkeypatch.setattr(httpx, "get", lambda *a, **k: wiki_response([]))
    with pytest.raises(ToolError, match="no results"):
        wikipedia_search("zzzzqqq")


# ---------------------------------------------------------------------------
# Search fallback: web_search -> wikipedia_search
# ---------------------------------------------------------------------------
def test_web_search_fails_so_agent_uses_wikipedia():
    wiki_queries = []

    def wiki(query):
        wiki_queries.append(query)
        return {"results": [{"title": "Intermittent fasting",
                             "url": "https://en.wikipedia.org/wiki/Intermittent_fasting",
                             "snippet": "..."}]}

    llm = FakeLLM(script=[
        decide(call("web_search", query="intermittent fasting")),
        decide(call("fetch_page", url="https://en.wikipedia.org/wiki/Intermittent_fasting")),
    ])
    result = run_agent(Q, llm=llm, tools=fake_tools(
        search=always_fail("search failed: No results found."), wiki=wiki))

    step = result["tool_calls"][0]
    assert step["success"] is True
    assert step["attempts"] == 3                       # 1 try + 2 retries
    assert step["search_tool_used"] == "wikipedia_search"
    assert "fell back to wikipedia_search" in step["fallback_reason"]
    assert wiki_queries == ["intermittent fasting"]    # same query was reused
    assert result["steps"] == 2                        # fallback did not cost a step
    assert result["sources"][0]["url"].startswith("https://en.wikipedia.org/")
    assert "[1]" in result["answer"]


def test_successful_web_search_is_logged_without_fallback():
    llm = FakeLLM(script=[decide(call("web_search", query="fasting"))])
    result = run_agent(Q, llm=llm, tools=fake_tools(wiki=always_fail("should not run")))
    step = result["tool_calls"][0]
    assert step["search_tool_used"] == "web_search"
    assert step["fallback_reason"] is None


def test_both_searches_fail_gives_honest_answer_without_citations():
    llm = FakeLLM(
        script=[decide(call("web_search", query="fasting")),
                decide(call("web_search", query="fasting health"))],
        answer="Fasting is great [1][2].",  # would be a made-up citation
    )
    result = run_agent(Q, llm=llm, tools=fake_tools(
        search=always_fail("search failed: No results found."),
        wiki=always_fail("wikipedia search timed out")))

    assert result["sources"] == []
    assert "couldn't find" in result["answer"]
    assert "[" not in result["answer"]                 # no citations at all
    for step in result["tool_calls"]:
        assert step["success"] is False
        assert step["search_tool_used"] is None
        assert "which also failed (wikipedia search timed out)" in step["fallback_reason"]
    tools_in_errors = {e["tool"] for e in result["errors"]}
    assert tools_in_errors == {"web_search", "wikipedia_search"}


# ---------------------------------------------------------------------------
# Nudge: don't let the agent give up before fetching anything
# ---------------------------------------------------------------------------
def test_agent_is_reminded_once_to_fetch_before_stopping():
    from langchain_core.messages import AIMessage

    llm = FakeLLM(script=[
        decide(call("web_search", query="fasting")),
        AIMessage("DONE"),                                   # gives up too early
        decide(call("fetch_page", url="https://a.org")),     # after the reminder
    ])
    result = run_agent(Q, llm=llm, tools=fake_tools())
    assert [s["url"] for s in result["sources"]] == ["https://a.org"]
    assert "[1]" in result["answer"]


def test_reminder_is_only_sent_once():
    from langchain_core.messages import AIMessage

    llm = FakeLLM(forever=lambda: AIMessage("DONE"))  # refuses to ever fetch
    result = run_agent(Q, llm=llm, tools=fake_tools())
    assert llm.agent_calls == 2                        # first try + one reminder
    assert result["sources"] == [] and "couldn't find" in result["answer"]


def test_agent_is_reminded_when_it_has_only_one_source():
    from langchain_core.messages import AIMessage

    llm = FakeLLM(script=[
        decide(call("fetch_page", url="https://a.org")),
        AIMessage("DONE"),                                   # only 1 source so far
        decide(call("fetch_page", url="https://b.org")),     # after the reminder
    ])
    result = run_agent(Q, llm=llm, tools=fake_tools())
    assert len(result["sources"]) == 2
