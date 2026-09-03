"""
Tool-Use Correctness dimension — checklist generation.

Separate from task completion. Task completion asks "did the user get the right outcome?".
This dimension asks "did the agent USE ITS TOOLS correctly on the way there?" — right
tool, right parameters, results read correctly, no needless calls, and any verification
call the situation required.

The checklist always has the same five checks (one per tool-use failure type), but each
is filled in with the specifics of THIS request. It is generated from the user's opening
message and the domain config alone, before seeing the agent's actual calls.

The LLM call is the single function `generate_tooluse_checklist`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from config import DomainConfig

MODEL = "claude-opus-5"

# The five criterion ids, in order. Each maps to one failure type from PRD Section 4.
CRITERION_IDS = (
    "correct_tools_selected",           # <- "wrong tool"
    "parameters_correct",               # <- "incorrect parameters"
    "tool_results_interpreted_correctly",  # <- "tool result misinterpreted"
    "no_unnecessary_tool_calls",        # <- "unnecessary tool call"
    "required_verification_call_made",  # <- "missing required tool call"
)

SYSTEM_PROMPT = """\
You evaluate an AI agent's TOOL USE: whether it called the right tools, with the right
parameters, read the results correctly, avoided needless calls, and made the calls the
situation required — including any verification call that was available.

Right now your ONLY job is to write the tool-use checklist for one request, from the
user's opening message and the system's tools alone. You have NOT seen the agent's actual
tool calls. Do not guess what it did.

The checklist ALWAYS has exactly these five criteria, in this order, with these ids. Fill
each one in with the concrete detail for THIS situation:

1. correct_tools_selected — name the tool(s) the situation calls for, and, if helpful, the
   kind of tool that would be wrong here.
2. parameters_correct — state what the key parameters should be: which record, which
   item(s), and how any value should be derived (e.g. from a field in the retrieved
   record, not from an unrelated total).
3. tool_results_interpreted_correctly — state what the agent must correctly read out of
   the tool results in order to act well (e.g. which item the issue concerns, the current
   state, a value).
4. no_unnecessary_tool_calls — name tools that would NOT be needed for this request, so an
   extra call can be flagged.
5. required_verification_call_made — if a read tool exists that should be called to verify
   the situation BEFORE any write action, name it and say it applies. If NO verification
   tool could apply to this request (nothing available could confirm the relevant fact),
   say so explicitly — a verification that was never possible must not be treated as a miss.

Keep each criterion to one or two plain sentences a reviewer could mark yes or no.
Respond with ONLY a JSON array. Each element:
  {"id": "<one of the five ids above>", "criterion": "<the filled-in criterion>"}"""


@dataclass
class ToolUseCriterion:
    id: str
    criterion: str


def _build_user_prompt(opening_message: str, cfg: DomainConfig) -> str:
    return f"""\
{cfg.summary_for_prompt()}

THE USER'S OPENING MESSAGE:
{opening_message}

Write the five-item tool-use checklist for this request."""


def _extract_json_array(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    bare = re.search(r"\[.*\]", text, re.DOTALL)
    if bare:
        return bare.group(0)
    raise ValueError(f"no JSON array found in model reply:\n{text}")


def generate_tooluse_checklist(opening_message: str, cfg: DomainConfig) -> list[ToolUseCriterion]:
    """Ask Claude for the tool-use checklist. Raises if credentials aren't configured."""
    from anthropic import Anthropic

    client = Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(opening_message, cfg)}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    rows = json.loads(_extract_json_array(text))
    return [ToolUseCriterion(id=row["id"], criterion=row["criterion"]) for row in rows]


def tooluse_checklist_path_for(trajectory_path: str) -> str:
    name = os.path.splitext(os.path.basename(trajectory_path))[0]
    return os.path.join("tooluse_checklists", f"{name}.json")


def save_tooluse_checklist(items: list[ToolUseCriterion], path: str, opening_message: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "generated_from_opening_message": opening_message,
        "items": [{"id": c.id, "criterion": c.criterion} for c in items],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_tooluse_checklist(path: str) -> list[ToolUseCriterion]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [ToolUseCriterion(id=r["id"], criterion=r["criterion"]) for r in payload["items"]]


def display_tooluse_checklist(items: list[ToolUseCriterion]) -> None:
    print("=" * 70)
    print(f"TOOL-USE CORRECTNESS CHECKLIST — {len(items)} criteria")
    print("(from the opening message + tool config only, before seeing the agent's calls)")
    print("=" * 70)
    for n, item in enumerate(items, start=1):
        print(f"{n}. [{item.id}]")
        print(f"   {item.criterion}")
    print("=" * 70)
