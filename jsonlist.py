"""
Tolerant extraction of a JSON list from an LLM reply.

Models mostly return a clean `[ {...}, {...} ]`, but sometimes wrap it in ```json
fences, add a sentence before or after, or emit one object per line with no
enclosing brackets. All of those carry the same data — this recovers the list
rather than raising and dropping a score.
"""

from __future__ import annotations

import json
import re


def extract_json_list(text: str) -> list:
    text = text.strip()

    # 1. strip a ```json ... ``` fence if present
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 2. a bracketed array anywhere in the text (prose before/after is fine)
    bracket = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket:
        try:
            value = json.loads(bracket.group(0))
            if isinstance(value, list):
                return value
        except json.JSONDecodeError:
            pass

    # 3. a stream of top-level {...} objects (compact or pretty-printed, no [ ])
    objs: list = []
    depth = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(text[start : i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    if objs:
        return objs

    raise ValueError(f"no JSON list found in model reply:\n{text}")
