"""
Grounding dimension — step 2: check each extracted claim against the trajectory's tool results.

Internal consistency only: for each claim, does a specific tool result
earlier in THIS trajectory support it? Not: is it true in the real world.

Label per claim:
  grounded     — a specific tool result directly supports it
  contradicted — a specific tool result directly conflicts with it
  ungrounded   — no tool result addresses it either way (a general policy / disclaimer
                 with nothing in the data to check against)

Grounding score = grounded / total. Contradicted and ungrounded both count as not-grounded,
but are labelled differently because a contradiction is the more severe failure.

The judgement is one isolated LLM call: `check_grounding`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from grounding_claims import Claim
from trajectory import Trajectory
from jsonlist import extract_json_list

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You check whether each claim in an AI support agent's final message is supported by the
TOOL RESULTS earlier in the same trajectory. This is an internal-consistency check only:
you are NOT checking real-world truth, only whether the trajectory's own tool results back
the claim.

For each claim, assign one label:
- "grounded"     — a specific tool result in the trajectory directly supports the claim.
- "contradicted" — a specific tool result directly conflicts with the claim.
- "ungrounded"   — no tool result in the trajectory addresses the claim either way (e.g. a
                   general policy or disclaimer with nothing in the data to check).

Rules:
- Point to the specific tool result and field that decided it.
- "contradicted" requires an actual conflict with returned data, not merely its absence.
- A general statement not tied to this order's data (a standard policy, a disclaimer), with
  no supporting field, is "ungrounded" — not "grounded".
- Do not use outside knowledge about how this kind of system usually works.

Respond with ONLY a JSON array. Each element:
  {"id": "<claim id>", "label": "grounded" | "contradicted" | "ungrounded", "reason": "<one sentence citing the tool result>"}"""


@dataclass
class ClaimVerdict:
    id: str
    claim: str
    label: str            # grounded | contradicted | ungrounded
    reason: str


@dataclass
class GroundingReport:
    verdicts: list[ClaimVerdict]

    @property
    def grounded(self) -> int:
        return sum(1 for v in self.verdicts if v.label == "grounded")

    @property
    def contradicted(self) -> int:
        return sum(1 for v in self.verdicts if v.label == "contradicted")

    @property
    def ungrounded(self) -> int:
        return sum(1 for v in self.verdicts if v.label == "ungrounded")

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def fraction(self) -> float:
        return self.grounded / self.total if self.total else 0.0


def _tool_results_text(traj: Trajectory) -> str:
    lines = []
    for t in traj.turns:
        if t.role == "TOOL":
            lines.append(f"CALL: {t.text}")
        elif t.role == "RESULT":
            lines.append(f"RESULT: {t.text}")
    return "\n".join(lines) if lines else "(no tool calls in this trajectory)"




def check_grounding(claims: list[Claim], traj: Trajectory) -> GroundingReport:
    """Ask Claude to label each claim against the trajectory's tool results. Raises if no creds."""
    from anthropic import Anthropic

    claims_json = json.dumps(
        [{"id": c.id, "claim": c.claim, "quote": c.quote} for c in claims], indent=2
    )
    user_prompt = f"""\
TOOL CALLS AND RESULTS IN THIS TRAJECTORY (the only evidence you may use):
{_tool_results_text(traj)}

CLAIMS FROM THE AGENT'S FINAL MESSAGE:
{claims_json}

Label every claim."""

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

    by_id = {c.id: c.claim for c in claims}
    verdicts = [
        ClaimVerdict(
            id=row["id"],
            claim=by_id.get(row["id"], "(unknown claim)"),
            label=row["label"],
            reason=row["reason"],
        )
        for row in rows
    ]
    return GroundingReport(verdicts=verdicts)


def save_grounding_score(
    report: GroundingReport, path: str, claims_file: str, trajectory_file: str
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "trajectory_file": trajectory_file,
        "claims_file": claims_file,
        "verdicts": [
            {"id": v.id, "claim": v.claim, "label": v.label, "reason": v.reason}
            for v in report.verdicts
        ],
        "grounded": report.grounded,
        "contradicted": report.contradicted,
        "ungrounded": report.ungrounded,
        "total": report.total,
        "score": round(report.fraction, 4),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


_MARK = {
    "grounded": "✓ grounded    ",
    "contradicted": "✗ contradicted",
    "ungrounded": "· ungrounded  ",
}


def display_grounding(report: GroundingReport) -> None:
    print("=" * 70)
    print("GROUNDING CHECK")
    print("=" * 70)
    for n, v in enumerate(report.verdicts, start=1):
        print(f"{n}. {_MARK.get(v.label, v.label)}  [{v.id}]")
        print(f"   claim:  {v.claim}")
        print(f"   reason: {v.reason}")
    print("-" * 70)
    print(
        f"Grounded: {report.grounded} of {report.total}   "
        f"(contradicted: {report.contradicted}, ungrounded: {report.ungrounded})"
    )
    print(f"Grounding score = {report.grounded}/{report.total} = {report.fraction:.2f}")
    print("(contradicted and ungrounded both count as not-grounded)")
    print("=" * 70)
