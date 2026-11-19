"""The tools the agent can choose between.

Each tool does ONE job and either returns a result or raises ToolError with a
short, human-readable reason. The graph (graph.py) catches these errors,
retries once, and then tells the LLM to try something else. Tools never decide
what to do about failures themselves - that keeps them simple.
"""

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import StructuredTool

from agent import config


class ToolError(Exception):
    """A tool could not do its job (timeout, 404, empty result, blocked page...)."""


# ---------------------------------------------------------------------------
# Tool 1: web_search
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> dict:
    """Search DuckDuckGo and return a list of {title, url, snippet}."""
    from ddgs import DDGS  # imported here so tests never touch the network

    try:
        # backend="auto": if DuckDuckGo blocks us (e.g. an HTTP 202 bot challenge),
        # ddgs quietly tries other engines. timeout stops a hung request.
        raw = DDGS(timeout=config.SEARCH_TIMEOUT).text(
            query, max_results=max_results, backend="auto"
        )
    except Exception as exc:  # ddgs raises its own errors, e.g. "No results found"
        raise ToolError(f"search failed: {exc}") from exc

    results = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw or []
        if r.get("href")
    ]
    if not results:
        raise ToolError("search returned no results")
    return {"results": results}


# ---------------------------------------------------------------------------
# Tool 1b: wikipedia_search (a second, independent search engine)
# ---------------------------------------------------------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks API users to send a descriptive User-Agent.
WIKI_HEADERS = {"User-Agent": "cited-research-agent/0.1 (student learning project)"}


def wikipedia_search(query: str, max_results: int = 5) -> dict:
    """Search English Wikipedia (free MediaWiki API, no key) for {title, url, snippet}."""
    params = {"action": "query", "list": "search", "srsearch": query,
              "srlimit": max_results, "format": "json"}
    try:
        response = httpx.get(WIKI_API, params=params, headers=WIKI_HEADERS,
                             timeout=config.SEARCH_TIMEOUT)
        response.raise_for_status()
        hits = response.json().get("query", {}).get("search", [])
    except httpx.TimeoutException as exc:
        raise ToolError("wikipedia search timed out") from exc
    except Exception as exc:  # network error, HTTP error, bad JSON
        raise ToolError(f"wikipedia search failed: {exc.__class__.__name__}") from exc

    if not hits:
        raise ToolError("wikipedia search returned no results")
    return {"results": [
        {"title": h["title"],
         "url": "https://en.wikipedia.org/wiki/" + h["title"].replace(" ", "_"),
         # Snippets contain <span> highlight tags - strip them.
         "snippet": BeautifulSoup(h.get("snippet", ""), "html.parser").get_text()}
        for h in hits
    ]}


# ---------------------------------------------------------------------------
# Tool 2: fetch_page
# ---------------------------------------------------------------------------
# Some sites block requests that don't look like a browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_page(url: str) -> dict:
    """Download a web page and return {url, title, text} with clean plain text."""
    if not url.startswith(("http://", "https://")):
        raise ToolError(f"not a web URL: {url}")

    try:
        response = httpx.get(
            url, headers=HEADERS, timeout=config.HTTP_TIMEOUT, follow_redirects=True
        )
    except httpx.TimeoutException as exc:
        raise ToolError("timed out") from exc
    except httpx.HTTPError as exc:
        raise ToolError(f"network error: {exc.__class__.__name__}") from exc

    if response.status_code in (401, 403, 429):
        raise ToolError(f"blocked by the site (HTTP {response.status_code})")
    if response.status_code >= 400:
        raise ToolError(f"HTTP {response.status_code}")

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text/plain" not in content_type:
        raise ToolError(f"unsupported content type: {content_type or 'unknown'}")

    title, text = html_to_text(response.text)
    if len(text) < 200:
        # Almost no text usually means a cookie wall, captcha or JavaScript-only page.
        raise ToolError("page is empty or blocked (too little text)")

    return {"url": str(response.url), "title": title, "text": text[: config.MAX_PAGE_CHARS]}


def html_to_text(html: str) -> tuple[str, str]:
    """Strip menus, scripts and ads from HTML and return (title, readable text)."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    # Remove parts of the page that are not the main content.
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "aside", "form", "svg", "iframe"]):
        tag.decompose()

    # Prefer the <main> or <article> part of the page if there is one.
    main = soup.find("article") or soup.find("main") or soup.body or soup
    lines = (line.strip() for line in main.get_text("\n").splitlines())
    # Keep lines that look like real sentences, not single menu words.
    text = "\n".join(line for line in lines if len(line.split()) > 3)
    return title, text


# ---------------------------------------------------------------------------
# Tool 3: summarise_page (uses the LLM)
# ---------------------------------------------------------------------------
def summarise_page(url: str, focus: str, llm) -> dict:
    """Fetch a page and ask the LLM for a short summary focused on `focus`."""
    page = fetch_page(url)  # may raise ToolError - that's fine, the graph handles it
    prompt = (
        f"Summarise the following web page in 5-8 bullet points, focusing on: {focus}.\n"
        "Only include facts stated in the page. Keep numbers and study details.\n\n"
        f"TITLE: {page['title']}\n\n{page['text']}"
    )
    try:
        reply = llm.invoke(prompt)
    except Exception as exc:
        raise ToolError(f"LLM summary failed: {exc.__class__.__name__}") from exc

    summary = message_text(reply).strip()
    if not summary:
        raise ToolError("LLM returned an empty summary")
    return {
        "url": page["url"],
        "title": page["title"],
        "text": summary,
        "usage": getattr(reply, "usage_metadata", None) or {},
    }


def message_text(message) -> str:
    """Get plain text from an LLM reply (Gemini sometimes returns a list of parts)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, (str, dict))
        )
    return str(content)


# ---------------------------------------------------------------------------
# Wrap the functions as LangChain tools so the LLM can see and call them.
# The docstrings/descriptions below are what the LLM reads to choose a tool.
# ---------------------------------------------------------------------------
def make_tools(llm) -> dict[str, StructuredTool]:
    """Return the four tools, keyed by name."""

    def _web_search(query: str) -> dict:
        return web_search(query)

    def _wikipedia_search(query: str) -> dict:
        return wikipedia_search(query)

    def _fetch_page(url: str) -> dict:
        return fetch_page(url)

    def _summarise_page(url: str, focus: str) -> dict:
        return summarise_page(url, focus, llm)

    tools = [
        StructuredTool.from_function(
            _web_search, name="web_search",
            description="Search the web. Returns titles, URLs and short snippets. "
                        "Snippets cannot be cited - fetch a page to use it as a source.",
        ),
        StructuredTool.from_function(
            _wikipedia_search, name="wikipedia_search",
            description="Search English Wikipedia. Returns article titles, URLs and "
                        "snippets. Good for background facts and definitions, and a "
                        "backup when web_search fails. Fetch a page to cite it.",
        ),
        StructuredTool.from_function(
            _fetch_page, name="fetch_page",
            description="Download a URL and return its main text. The page becomes a "
                        "numbered source you can cite.",
        ),
        StructuredTool.from_function(
            _summarise_page, name="summarise_page",
            description="Download a URL and return a short summary focused on `focus`. "
                        "Use for long pages. The page becomes a numbered source.",
        ),
    ]
    return {t.name: t for t in tools}
