"""
Task Completion dimension — step 1: generate the success checklist from the user's
opening message.

Before looking at anything the agent did, read ONLY the user's first message and write
down the concrete things that must be true for THIS specific request to count as "done".
Doing it first keeps the checklist an honest read of what was asked, not something bent
to match whatever the agent happened to do.

Domain-neutral: the system under evaluation (its tools, states, rules) comes from the
config, not from this file. The LLM call is kept in ONE function, `generate_checklist`.
"""

from __future__ import annotations
from jsonlist import extract_json_list

import json
import os
import re
from dataclasses import dataclass

from config import DomainConfig

MODEL = "claude-opus-5"

# What we ask the model to be, every time. It is deliberately strict about NOT
# assuming what the agent did — at this point we have not shown it the transcript.
SYSTEM_PROMPT = """\
You judge whether an AI agent fully completed what a user asked for.

Right now your ONLY job is to write the success checklist for one request: the list of
concrete, yes/no-checkable things that must be true for THIS specific request to count as
done. You are given the user's opening message and the system's capabilities and rules.
You have NOT seen what the agent did. Do not guess or assume what it did.

How to write a good checklist:
- One item per distinct thing the user is asking for. Two problems in one message means
  at least two items — do not collapse them.
- Match what they actually said. If they asked for a specific resolution, the item is
  that specific resolution. If they only reported a problem without naming one, the item
  is "an appropriate resolution is provided", not a specific one.
- Include an item for the user being told the outcome — and any timeline too, when the
  agent took a consequential action (a payment, a cancellation, an irreversible change).
- Include a guard item: nothing beyond what was asked should be acted on, changed, or
  reversed.
- If the user asks for something the system's rules do not allow in the current state,
  the correct completion is a clear explanation plus an alternative — write the item that
  way, not as the impossible action.
- If the problem is caused by something outside the agent's control that the system
  cannot fix directly, completing the task means giving an accurate status, not making
  the underlying problem go away.
- Do NOT write criteria about whether the agent verified the situation with a tool before
  acting (e.g. "checked X before doing Y", "confirmed the problem was real"). Whether the
  right process/tools were followed is scored by the separate tool-use correctness
  dimension. Task completion is only about whether the user ended up with the right
  outcome. Identifying the correct record and the specific item(s) the user flagged is
  fine to include — that is part of the outcome, not a process check.

Keep each criterion to one plain sentence a reviewer could mark yes or no."""


@dataclass
class ChecklistItem:
    id: str
    criterion: str


def _build_user_prompt(opening_message: str, cfg: DomainConfig) -> str:
    return f"""\
{cfg.summary_for_prompt()}

THE USER'S OPENING MESSAGE:
{opening_message}

Write the checklist for this request. Respond with ONLY a JSON array. Each element:
  {{"id": "<short_snake_case_id>", "criterion": "<one sentence, yes/no checkable>"}}"""




def generate_checklist(opening_message: str, cfg: DomainConfig) -> list[ChecklistItem]:
    """Ask Claude for the success checklist. Raises if credentials aren't configured."""
    from anthropic import Anthropic  # imported here so the rest of the tool runs without the SDK

    client = Anthropic()  # reads ANTHROPIC_API_KEY (or an `ant auth login` profile)
    message = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(opening_message, cfg)}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    raw = extract_json_list(text)
    return [ChecklistItem(id=row["id"], criterion=row["criterion"]) for row in raw]


def checklist_path_for(trajectory_path: str) -> str:
    """Where the saved checklist for a given trajectory lives: checklists/<name>.json."""
    name = os.path.splitext(os.path.basename(trajectory_path))[0]
    return os.path.join("checklists", f"{name}.json")


def save_checklist(items: list[ChecklistItem], path: str, opening_message: str) -> None:
    """Write the checklist to disk so scoring reuses the exact same list."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "generated_from_opening_message": opening_message,
        "items": [{"id": c.id, "criterion": c.criterion} for c in items],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_checklist(path: str) -> list[ChecklistItem]:
    """Read a checklist saved by save_checklist."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [ChecklistItem(id=row["id"], criterion=row["criterion"]) for row in payload["items"]]


def display_checklist(items: list[ChecklistItem]) -> None:
    print("=" * 70)
    print(f"TASK-COMPLETION CHECKLIST — {len(items)} criteria")
    print("(generated from the opening message only, before seeing the agent's work)")
    print("=" * 70)
    for n, item in enumerate(items, start=1):
        print(f"{n}. [{item.id}]")
        print(f"   {item.criterion}")
    print("=" * 70)
