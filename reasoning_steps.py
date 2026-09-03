"""
Reasoning Quality dimension — step 1: identify the reasoning steps in a trajectory.

A "reasoning step" is a point where the agent had to WORK SOMETHING OUT rather than read
it off: an inference, a calculation, or a judgment call (PRD Section 3). Reading a single
value straight from a tool result is not a reasoning step.

No correctness judgement here — that is the next stage. If a trajectory has no reasoning
steps, this dimension is NOT APPLICABLE to it (PRD Sections 3, 6) and it is excluded from
the reasoning-quality score rather than forced to 1.00 / 0.00.

DESIGN DECISION (PRD Section 5 vs Section 6, resolved): option (a). A judgment/inference
step whose PREMISE is ungrounded (the agent asserted something with no tool result or
config rule behind it) is identified here for transparency but is NOT scored by reasoning
quality — it is a grounding failure. Reasoning quality only scores how the agent operated
on facts it actually had (arithmetic, logic applied to real inputs). Such a step carries
"scoring_note" and is skipped by the scoring stage.

The LLM call is the single function `identify_reasoning_steps`.
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
You identify the REASONING STEPS in an AI agent's trajectory — every point where the agent
draws a conclusion, performs a calculation, or makes a judgment call based on the
information available to it.

A reasoning step is anything the agent had to WORK OUT, not just read off:
- an inference — e.g. concluding which of several things a report refers to, by combining
  what the user said with what a record shows
- a calculation — e.g. combining two or more values from a record to produce a total
- a judgment call — e.g. deciding a rule applies, or choosing between two valid courses of
  action

NOT a reasoning step: reading a single value straight out of a tool result with nothing to
combine or infer (e.g. "your status is 'active'", "the amount on file is 450").

For each step record:
- kind: "inference" | "calculation" | "judgment"
- step: what the agent was working out
- inputs: the specific data it used
- conclusion: what the agent concluded or produced

Additionally, whenever a step is a calculation, add one "guard" item with kind "guard"
and id "figure_follows_from_inputs": it checks that the agent's stated figure actually
corresponds to a real combination of the input values (catches numbers that came from
nowhere).

Do NOT judge whether any step is correct — that is the next stage.
If the trajectory contains NO reasoning steps, return an empty array [].

Respond with ONLY a JSON array. Each element:
  {"id": "<short_snake_case_id>", "kind": "inference|calculation|judgment|guard",
   "step": "<what was being worked out>", "inputs": "<data used>", "conclusion": "<result>"}"""


@dataclass
class ReasoningStep:
    id: str
    kind: str
    step: str
    inputs: str
    conclusion: str
    scoring_note: str = ""   # non-empty => not scored by reasoning quality (handled elsewhere)


def _transcript_text(traj: Trajectory) -> str:
    return "\n".join(f"{t.role}: {t.text}" for t in traj.turns)




def identify_reasoning_steps(traj: Trajectory) -> list[ReasoningStep]:
    """Ask Claude to list the reasoning steps. Empty list == dimension not applicable. Raises if no creds."""
    from anthropic import Anthropic

    user_prompt = f"""\
FULL TRANSCRIPT:
{_transcript_text(traj)}

Identify the reasoning steps."""

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
    return [
        ReasoningStep(
            id=r["id"],
            kind=r.get("kind", "inference"),
            step=r["step"],
            inputs=r.get("inputs", ""),
            conclusion=r.get("conclusion", ""),
            scoring_note=r.get("scoring_note", ""),
        )
        for r in rows
    ]


def reasoning_steps_path_for(trajectory_path: str) -> str:
    name = os.path.splitext(os.path.basename(trajectory_path))[0]
    return os.path.join("reasoning_steps", f"{name}.json")


def save_reasoning_steps(steps: list[ReasoningStep], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "applicable": len(steps) > 0,
        "steps": [
            {
                "id": s.id, "kind": s.kind, "step": s.step,
                "inputs": s.inputs, "conclusion": s.conclusion,
                **({"scoring_note": s.scoring_note} if s.scoring_note else {}),
            }
            for s in steps
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_reasoning_steps(path: str) -> list[ReasoningStep]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [
        ReasoningStep(
            id=s["id"], kind=s["kind"], step=s["step"], inputs=s["inputs"],
            conclusion=s["conclusion"], scoring_note=s.get("scoring_note", ""),
        )
        for s in payload["steps"]
    ]


def display_reasoning_steps(steps: list[ReasoningStep]) -> None:
    print("=" * 70)
    if not steps:
        print("REASONING QUALITY — NOT APPLICABLE")
        print("No reasoning steps in this trajectory (nothing to infer or compute).")
        print("This trajectory is excluded from the reasoning-quality score.")
        print("=" * 70)
        return
    print(f"REASONING QUALITY — IDENTIFIED REASONING STEPS — {len(steps)} steps")
    print("(steps only; no correctness check yet)")
    print("=" * 70)
    for n, s in enumerate(steps, start=1):
        print(f"{n}. [{s.id}]  ({s.kind})")
        print(f"   step:       {s.step}")
        print(f"   inputs:     {s.inputs}")
        print(f"   conclusion: {s.conclusion}")
    print("=" * 70)
