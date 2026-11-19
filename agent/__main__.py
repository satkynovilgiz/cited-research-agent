"""Command line interface.

    python -m agent "What are the health effects of intermittent fasting?"
    python -m agent "..." --max-steps 5 --json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from agent import config
from agent.graph import run_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the research agent a question.")
    parser.add_argument("question", help="the research question, in quotes")
    parser.add_argument("--max-steps", type=int, default=config.DEFAULT_MAX_STEPS,
                        help=f"maximum tool calls (default {config.DEFAULT_MAX_STEPS})")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--quiet", action="store_true", help="hide the step-by-step log")
    args = parser.parse_args()

    # Step logs go to stderr so they don't mix with the answer on stdout.
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="  [log] %(message)s", stream=sys.stderr)
    # Keep other libraries' chatter out of the log.
    for noisy in ("httpx", "httpcore", "primp", "ddgs", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    result = run_agent(args.question, max_steps=args.max_steps)

    # Save every run as JSON so it can be evaluated later.
    runs = Path("runs")
    runs.mkdir(exist_ok=True)
    path = runs / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print("\n" + result["answer"] + "\n")
    cited = [s for s in result["sources"] if s["cited"]]
    if cited:
        print("Sources:")
        for s in cited:
            print(f"  [{s['id']}] {s['title']} - {s['url']}")
    usage = result["token_usage"]
    print(f"\n(tool calls: {result['steps']}/{result['max_steps']}, "
          f"errors: {len(result['errors'])}, tokens: {usage.get('total_tokens', 0)}, "
          f"log: {path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
