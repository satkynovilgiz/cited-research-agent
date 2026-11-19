"""Make sure every citation in the answer points to a page we really fetched.

The LLM is told to cite only fetched sources, but LLMs sometimes make things
up. So after the answer is written we check it in plain Python:
  * any [n] whose number is not a fetched source is deleted
  * we count sentences that have no citation at all (useful for evaluation)
"""

import re

CITATION = re.compile(r"\[(\d+)\]")


def check_citations(answer: str, sources: list[dict]) -> dict:
    """Remove invalid citations and report what was cited.

    `sources` is the list of fetched pages, each with an "id" (1, 2, 3...).
    Returns {answer, cited_ids, removed_ids, uncited_sentences}.
    """
    valid_ids = {s["id"] for s in sources}
    removed: set[int] = set()

    def keep_or_remove(match: re.Match) -> str:
        n = int(match.group(1))
        if n in valid_ids:
            return match.group(0)
        removed.add(n)
        return ""  # delete the made-up citation

    cleaned = CITATION.sub(keep_or_remove, answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)  # tidy "word ." -> "word."

    cited = sorted({int(n) for n in CITATION.findall(cleaned)})
    return {
        "answer": cleaned,
        "cited_ids": cited,
        "removed_ids": sorted(removed),
        "uncited_sentences": count_uncited_sentences(cleaned),
    }


def count_uncited_sentences(text: str) -> int:
    """Count sentences (longer than a few words) that contain no [n] citation."""
    count = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("note:"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if len(sentence.split()) >= 6 and not CITATION.search(sentence):
                count += 1
    return count
