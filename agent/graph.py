"""The research agent as a LangGraph state machine.

    START -> agent --(wants a tool)--> tools --(steps left)--> agent ...
               |                         |
               | (done)                  | (step limit hit)
               v                         v
             answer  <-------------------+
               |
              END

* agent  : the LLM looks at what it has so far and picks the next tool call
           (or decides it has enough).
* tools  : runs the requested tool(s), retries a failure once, records the
           step, and saves fetched pages as numbered sources.
* answer : writes the final answer from the fetched sources only, then
           citations.py deletes any citation that isn't a fetched source.
"""

import json
import logging
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent import config
from agent.citations import check_citations
from agent.tools import ToolError, make_tools, message_text

log = logging.getLogger("agent")

# Tools whose successful result is a downloaded page we can cite.
PAGE_TOOLS = {"fetch_page", "summarise_page"}
# Tools that return a list of search results.
SEARCH_TOOLS = {"web_search", "wikipedia_search"}
# If web_search fails completely, this tool is tried automatically instead.
FALLBACK_SEARCH = "wikipedia_search"


# ---------------------------------------------------------------------------
# 1. The state: everything the agent knows, passed from node to node.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    messages: Annotated[list, add_messages]  # chat history (add_messages appends)
    sources: list[dict]      # fetched pages: {id, url, title, text, tool}
    step_count: int          # how many tool calls have been used
    max_steps: int           # the hard limit
    tool_log: list[dict]     # one entry per tool call, for evaluation
    errors: list[dict]       # every failed attempt (including retries)
    usage: dict              # token counts added up over all LLM calls
    limit_reached: bool
    answer: str
    citation_check: dict


# ---------------------------------------------------------------------------
# 2. Prompts
# ---------------------------------------------------------------------------
AGENT_PROMPT = """You are a careful research agent. Answer the user's question using web sources.

Tools:
- web_search(query): search the whole web. Snippets are NOT sources and cannot be cited.
- wikipedia_search(query): search Wikipedia only. Use for background facts, or if
  web_search fails. Snippets cannot be cited either.
- fetch_page(url): download a page's text. It becomes a numbered source.
- summarise_page(url, focus): like fetch_page but returns a focused summary (good for long pages).

How to work:
1. Search first. 2. Fetch 2-4 of the most reliable results (official health/government
bodies, universities, peer-reviewed journals, established reference sites).
3. As soon as the fetched pages cover the question, STOP calling tools and reply "DONE".
If a tool fails, try a different URL or rephrase the search. Never fetch the same URL twice.

Budget: you have used {used} of {max_steps} tool calls ({left} left).
Sources fetched so far:
{sources}"""

# Sent (once) if the LLM tries to finish before fetching any page.
NUDGE = ("You have not fetched any page yet, so nothing can be cited. Call fetch_page on "
         "one or more of the most relevant URLs from the search results above (pick a "
         "different site if one was blocked). If you have no URLs, search again.")

ANSWER_PROMPT = """Write the final answer to the question using ONLY the numbered sources below.

Rules:
- After every sentence that states a fact, cite its source number(s), like [1] or [2][3].
- Only use these numbers: {ids}. Never invent a source or a URL.
- If the sources don't cover part of the question, say so plainly.
- Do not add a reference list at the end (it is added automatically).
- Be clear and concise: about 150-300 words, short paragraphs or bullet points.

QUESTION: {question}

SOURCES:
{sources}"""


# ---------------------------------------------------------------------------
# 3. Small helpers
# ---------------------------------------------------------------------------
def add_usage(total: dict, usage: dict | None) -> dict:
    """Add one LLM call's token counts to the running total."""
    total = dict(total)
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + int((usage or {}).get(key, 0) or 0)
    total["llm_calls"] = total.get("llm_calls", 0) + 1
    return total


def call_with_retry(tool, args: dict, delay: float,
                    retries: int = config.DEFAULT_RETRIES) -> tuple[dict | None, int, list[str]]:
    """Run a tool; if it fails, wait and try again, up to `retries` more times.

    The wait doubles each time (backoff): delay, 2*delay, 4*delay...
    Returns (result or None, number of attempts, list of error messages).
    Any exception is caught here, so a broken tool can never crash the agent.
    """
    errors = []
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return tool.invoke(args), attempt, errors
        except ToolError as exc:
            errors.append(str(exc))
        except Exception as exc:  # bad arguments, unexpected bugs, etc.
            errors.append(f"{exc.__class__.__name__}: {exc}")
        if attempt < attempts:
            time.sleep(delay * 2 ** (attempt - 1))
    return None, attempts, errors


def format_search_results(result: dict) -> str:
    lines = ["Search results (snippets cannot be cited - fetch a page to use it):"]
    for i, r in enumerate(result["results"], 1):
        lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}")
    return "\n".join(lines)


def format_sources(sources: list[dict], with_text: bool) -> str:
    if not sources:
        return "(none yet)"
    parts = []
    for s in sources:
        header = f"[{s['id']}] {s['title'] or 'Untitled'} - {s['url']}"
        parts.append(f"{header}\n{s['text']}\n" if with_text else header)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 4. Build the graph
# ---------------------------------------------------------------------------
def build_graph(llm, tools: dict, retry_delay: float = config.RETRY_DELAY):
    """Wire the three nodes together and return a runnable graph."""
    llm_with_tools = llm.bind_tools(list(tools.values()))

    # ---- Node: agent (LLM decides what to do next) -----------------------
    def agent_node(state: AgentState) -> dict:
        used, max_steps = state["step_count"], state["max_steps"]
        system = AGENT_PROMPT.format(
            used=used, max_steps=max_steps, left=max_steps - used,
            sources=format_sources(state["sources"], with_text=False),
        )
        messages = [SystemMessage(system), *state["messages"]]
        new_messages, usage = [], state["usage"]
        try:
            reply = llm_with_tools.invoke(messages)
            usage = add_usage(usage, getattr(reply, "usage_metadata", None))
            new_messages.append(reply)

            # The LLM wants to stop, but it has fetched nothing, so nothing could be
            # cited. If steps are left, remind it ONCE to fetch a page.
            already_nudged = any(getattr(m, "content", None) == NUDGE
                                 for m in state["messages"])
            if (not reply.tool_calls and not state["sources"]
                    and used < max_steps and not already_nudged):
                log.info(json.dumps({"event": "nudge", "reason": "tried to stop with no sources"}))
                new_messages.append(HumanMessage(NUDGE))
                reply = llm_with_tools.invoke(messages + new_messages)
                usage = add_usage(usage, getattr(reply, "usage_metadata", None))
                new_messages.append(reply)
        except Exception as exc:
            # The LLM itself failed (rate limit, network...). Don't crash:
            # go straight to answering with whatever we already have.
            error = {"where": "agent_llm", "error": f"{exc.__class__.__name__}: {exc}"}
            log.warning(json.dumps(error))
            return {"messages": new_messages + [AIMessage("(LLM error - stopping research)")],
                    "errors": state["errors"] + [error], "usage": usage}
        return {"messages": new_messages, "usage": usage}

    # ---- Node: tools (run the chosen tools, with retry + logging) --------
    def tools_node(state: AgentState) -> dict:
        step_count = state["step_count"]
        sources = list(state["sources"])
        tool_log = list(state["tool_log"])
        errors = list(state["errors"])
        usage = state["usage"]
        replies = []

        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], call["args"]

            # The hard step limit: never run more than max_steps tool calls.
            if step_count >= state["max_steps"]:
                replies.append(ToolMessage("Skipped: step limit reached.",
                                           tool_call_id=call["id"]))
                continue
            step_count += 1
            started = time.time()

            # Don't download the same page twice - point at the existing source.
            known = next((s for s in sources if s["url"] == args.get("url")
                          or s.get("requested_url") == args.get("url")), None)

            if name not in tools:
                result, attempts, errs = None, 1, [f"unknown tool '{name}'"]
            elif name in PAGE_TOOLS and known:
                result, attempts, errs = {"already": known}, 1, []
            else:
                retries = config.RETRIES.get(name, config.DEFAULT_RETRIES)
                result, attempts, errs = call_with_retry(tools[name], args, retry_delay,
                                                         retries)
            all_errors = [(name, e) for e in errs]

            # Automatic search fallback: web_search gave nothing -> try Wikipedia.
            search_tool_used = name if name in SEARCH_TOOLS and result else None
            fallback_reason = None
            if name == "web_search" and result is None and FALLBACK_SEARCH in tools:
                fallback_reason = (f"web_search failed after {attempts - 1} retries "
                                   f"({errs[-1]}) -> fell back to {FALLBACK_SEARCH}")
                log.warning(json.dumps({"step": step_count, "event": "fallback",
                                        "reason": fallback_reason}))
                result, _, fb_errs = call_with_retry(
                    tools[FALLBACK_SEARCH], args, retry_delay,
                    config.RETRIES.get(FALLBACK_SEARCH, config.DEFAULT_RETRIES))
                all_errors += [(FALLBACK_SEARCH, e) for e in fb_errs]
                if result is not None:
                    search_tool_used = FALLBACK_SEARCH
                else:
                    fallback_reason += f", which also failed ({fb_errs[-1]})"

            # Turn the result into a message the LLM can read.
            if result is None and fallback_reason:
                content = (f"ERROR: both searches failed: {fallback_reason}. "
                           "Try a different, simpler search query.")
            elif result is None:
                content = (f"ERROR: {name} failed after {attempts} attempt(s): {errs[-1]}. "
                           "Try a different URL or a different search query.")
            elif "already" in result:
                content = f"Already fetched as source [{known['id']}]."
            elif name in SEARCH_TOOLS:
                content = format_search_results(result)
                if fallback_reason:
                    content = (f"(web_search failed, so these results come from "
                               f"{FALLBACK_SEARCH}.)\n" + content)
            else:  # a page was fetched: save it as a new numbered source
                if result.get("usage"):
                    usage = add_usage(usage, result["usage"])
                source = {"id": len(sources) + 1, "url": result["url"],
                          "requested_url": args.get("url"), "title": result["title"],
                          "text": result["text"], "tool": name}
                sources.append(source)
                content = (f"SOURCE [{source['id']}] {source['title']}\n"
                           f"URL: {source['url']}\n\n{source['text']}")

            # Structured log entry for this step.
            entry = {
                "step": step_count, "tool": name, "input": args,
                "success": result is not None, "attempts": attempts,
                "error": (fallback_reason or errs[-1]) if result is None else None,
                "search_tool_used": search_tool_used,
                "fallback_reason": fallback_reason,
                "source_id": sources[-1]["id"] if result and name in PAGE_TOOLS
                             and "already" not in result else None,
                "duration_s": round(time.time() - started, 2),
            }
            tool_log.append(entry)
            errors += [{"step": step_count, "tool": tool_name, "input": args, "error": e}
                       for tool_name, e in all_errors]
            log.info(json.dumps(entry))
            replies.append(ToolMessage(content, tool_call_id=call["id"]))

        return {"messages": replies, "sources": sources, "step_count": step_count,
                "tool_log": tool_log, "errors": errors, "usage": usage,
                "limit_reached": step_count >= state["max_steps"]}

    # ---- Node: answer (write the cited final answer) ----------------------
    def answer_node(state: AgentState) -> dict:
        sources, usage = state["sources"], state["usage"]
        errors = list(state["errors"])

        if not sources:
            # Nothing was fetched, so nothing can be cited. Be honest.
            answer = ("I couldn't find and download any sources for this question, "
                      "so I can't give a cited answer. Please try rephrasing the question "
                      "or try again later.")
        else:
            prompt = ANSWER_PROMPT.format(
                ids=", ".join(f"[{s['id']}]" for s in sources),
                question=state["question"],
                sources=format_sources(sources, with_text=True),
            )
            try:
                reply = llm.invoke(prompt)
                usage = add_usage(usage, getattr(reply, "usage_metadata", None))
                answer = message_text(reply).strip()
            except Exception as exc:
                answer = ""
                errors.append({"where": "answer_llm",
                               "error": f"{exc.__class__.__name__}: {exc}"})
            if not answer:
                answer = ("I fetched the sources listed below but could not write a "
                          "summary because the language model failed. "
                          + " ".join(f"[{s['id']}]" for s in sources))

        check = check_citations(answer, sources)
        answer = check["answer"]
        if state["limit_reached"]:
            answer += (f"\n\nNote: I stopped after the {state['max_steps']}-tool-call limit, "
                       "so this answer may be incomplete.")
        log.info(json.dumps({"event": "answer", "cited": check["cited_ids"],
                             "removed_invalid": check["removed_ids"]}))
        return {"answer": answer, "citation_check": check, "usage": usage,
                "errors": errors}

    # ---- Routing: which node runs next? ----------------------------------
    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "answer"

    def after_tools(state: AgentState) -> str:
        return "answer" if state["step_count"] >= state["max_steps"] else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "answer": "answer"})
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "answer": "answer"})
    graph.add_edge("answer", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 5. The one function other code should call
# ---------------------------------------------------------------------------
def run_agent(question: str, max_steps: int = config.DEFAULT_MAX_STEPS,
              llm=None, tools: dict | None = None,
              retry_delay: float = config.RETRY_DELAY) -> dict:
    """Answer `question` with cited sources. Never raises; always returns a dict.

    `llm` and `tools` can be swapped for fakes in tests.
    """
    started = time.time()
    state: dict = {"question": question, "messages": [HumanMessage(question)],
                   "sources": [], "step_count": 0, "max_steps": max_steps,
                   "tool_log": [], "errors": [], "usage": {}, "limit_reached": False,
                   "answer": "", "citation_check": {}}
    try:
        llm = llm or config.make_llm()
        tools = tools or make_tools(llm)
        graph = build_graph(llm, tools, retry_delay)
        # recursion_limit is LangGraph's own safety net on top of our step limit.
        state = graph.invoke(state, {"recursion_limit": max_steps * 2 + 10})
    except Exception as exc:  # last line of defence: report, don't crash
        state["errors"] = state["errors"] + [{"where": "run_agent",
                                              "error": f"{exc.__class__.__name__}: {exc}"}]
        state["answer"] = state["answer"] or f"The agent failed to run: {exc}"

    cited = set(state["citation_check"].get("cited_ids", []))
    return {
        "question": question,
        "answer": state["answer"],
        "sources": [{"id": s["id"], "url": s["url"], "title": s["title"],
                     "tool": s["tool"], "cited": s["id"] in cited}
                    for s in state["sources"]],
        "steps": state["step_count"],
        "max_steps": max_steps,
        "limit_reached": state["limit_reached"],
        "tool_calls": state["tool_log"],
        "errors": state["errors"],
        "token_usage": state["usage"],
        "citation_check": {k: v for k, v in state["citation_check"].items() if k != "answer"},
        "model": getattr(llm, "model", None) or getattr(llm, "model_name", None),
        "duration_s": round(time.time() - started, 2),
    }
