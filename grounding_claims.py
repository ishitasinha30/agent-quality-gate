"""
Grounding dimension — step 1: claim extraction.

Pull the discrete factual claims out of the agent's FINAL message — each separate thing
the agent asserts as true (an amount, a status, a date or timeline, a fact about a
record, a policy statement). Apologies, pleasantries, and questions like "anything else I
can help with?" are not claims.

DESIGN DECISION (resolved): option (a), strict. General policy / disclaimer statements
ARE extracted as claims. If no tool result supports them they come back "ungrounded" in
the check and count against the score. They are not excluded at extraction time.

No grounding check here. That is the next step. This just isolates what will be checked.

The LLM call is the single function `extract_claims`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from trajectory import Trajectory
from jsonlist import extract_json_list

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You extract the factual claims from an AI agent's FINAL message to the user — each
discrete piece of information the agent asserts as true: an amount, a status, a date or
timeline, a fact about a record or account, or a policy statement.

Rules:
- Extract from the final agent message ONLY. Earlier turns are context for resolving what
  a pronoun or "it" refers to; do not pull claims from them.
- One claim per discrete assertion. If one sentence asserts three things, that is three
  claims.
- Exclude non-claims: apologies ("sorry about that"), pleasantries, offers of further
  help, and questions ("anything else I can help with?").
- Write each claim as a short standalone statement a checker could look up (e.g. "The
  amount involved is 450"), and include the exact span of the message it came from.
- Do NOT judge whether a claim is true or supported — that is a later step.

Respond with ONLY a JSON array. Each element:
  {"id": "<short_snake_case_id>", "claim": "<standalone statement>", "quote": "<exact text from the message>"}"""


@dataclass
class Claim:
    id: str
    claim: str
    quote: str


def _transcript_text(traj: Trajectory) -> str:
    return "\n".join(f"{t.role}: {t.text}" for t in traj.turns)


def _final_agent_message(traj: Trajectory) -> str | None:
    for turn in reversed(traj.turns):
        if turn.role == "AGENT":
            return turn.text
    return None




def extract_claims(traj: Trajectory) -> list[Claim]:
    """Ask Claude to list the factual claims in the final agent message. Raises if no creds."""
    from anthropic import Anthropic

    final_message = _final_agent_message(traj)
    if not final_message:
        return []

    user_prompt = f"""\
FULL TRANSCRIPT (context only — do not extract from here):
{_transcript_text(traj)}

THE AGENT'S FINAL MESSAGE (extract claims from THIS only):
{final_message}

Extract the factual claims."""

    client = Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    rows = extract_json_list(text)
    return [Claim(id=r["id"], claim=r["claim"], quote=r["quote"]) for r in rows]


def claims_path_for(trajectory_path: str) -> str:
    name = os.path.splitext(os.path.basename(trajectory_path))[0]
    return os.path.join("grounding_claims", f"{name}.json")


def save_claims(claims: list[Claim], path: str, final_message: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "final_message": final_message,
        "claims": [{"id": c.id, "claim": c.claim, "quote": c.quote} for c in claims],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_claims(path: str) -> list[Claim]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [Claim(id=r["id"], claim=r["claim"], quote=r["quote"]) for r in payload["claims"]]


def display_claims(claims: list[Claim]) -> None:
    print("=" * 70)
    print(f"GROUNDING — EXTRACTED CLAIMS FROM FINAL MESSAGE — {len(claims)} claims")
    print("(claims only; no grounding check yet)")
    print("=" * 70)
    for n, c in enumerate(claims, start=1):
        print(f"{n}. [{c.id}]")
        print(f"   claim: {c.claim}")
        print(f"   quote: \"{c.quote}\"")
    print("=" * 70)
