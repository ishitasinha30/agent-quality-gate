"""
Tool-Use Correctness dimension — scoring.

Walk the five tool-use criteria against the agent's ACTUAL tool calls in the trajectory:
selection, parameters, result interpretation, unnecessary calls, and any required
verification call. Only "met" counts. Score = criteria met / 5.

The judgement is one isolated LLM call: `score_tooluse`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from config import DomainConfig
from tooluse_checklist import ToolUseCriterion
from trajectory import Trajectory
from jsonlist import extract_json_list

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You check an AI agent's ACTUAL tool calls against a fixed five-item tool-use checklist.
The checklist was written beforehand from the user's opening message and the tool config.
Judge only what the transcript shows.

For each criterion return one verdict:
- "met"     — the transcript clearly satisfies it.
- "partial" — the right kind of thing was done but part of it is wrong (correct tool, one
              parameter off; a verification attempted on the wrong tool; a result read
              half-right).
- "not_met" — a required call or parameter is wrong or missing, or a result was misread.

Rules:
- parameters_correct: check parameters against the tool RESULTS, not just for
  plausibility — e.g. a value the agent passes must match the corresponding field in the
  record a read tool returned, and identifiers must match the retrieved record.
- tool_results_interpreted_correctly: what the agent said or did must actually follow
  from what the tools returned.
- no_unnecessary_tool_calls: a call whose result the agent never used, or that the
  request plainly did not need, is an unnecessary call.
- required_verification_call_made: if the criterion names a verification tool that
  applies, the agent must have called it BEFORE the relevant write action — if not, this
  is not_met. If the criterion states no verification tool could apply, mark this met.
- A technically-successful write call does not excuse a wrong parameter or a skipped
  verification.

Give a one-sentence reason each, citing the specific call or result. Respond with ONLY a
JSON array. Each element:
  {"id": "<criterion id>", "verdict": "met" | "partial" | "not_met", "reason": "<one sentence>"}"""


@dataclass
class ToolUseResult:
    id: str
    criterion: str
    verdict: str
    reason: str


@dataclass
class ToolUseReport:
    results: list[ToolUseResult]

    @property
    def met(self) -> int:
        return sum(1 for r in self.results if r.verdict == "met")

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def fraction(self) -> float:
        return self.met / self.total if self.total else 0.0


def _transcript_text(traj: Trajectory) -> str:
    return "\n".join(f"{t.role}: {t.text}" for t in traj.turns)


def _tool_calls_text(traj: Trajectory) -> str:
    if not traj.tool_calls:
        return "(no tool calls were made)"
    return "\n".join(
        f"- {t.tool_name}({t.tool_args_text or ''})" for t in traj.tool_calls
    )




def score_tooluse(
    checklist: list[ToolUseCriterion],
    traj: Trajectory,
    cfg: DomainConfig,
) -> ToolUseReport:
    """Ask Claude to mark each tool-use criterion against the actual calls. Raises if no creds."""
    from anthropic import Anthropic

    checklist_json = json.dumps(
        [{"id": c.id, "criterion": c.criterion} for c in checklist], indent=2
    )
    user_prompt = f"""\
{cfg.summary_for_prompt()}

TOOL-USE CHECKLIST (written beforehand):
{checklist_json}

TOOL CALLS THE AGENT ACTUALLY MADE:
{_tool_calls_text(traj)}

FULL TRANSCRIPT:
{_transcript_text(traj)}

Mark every criterion."""

    client = Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    rows = extract_json_list(text)

    by_id = {c.id: c.criterion for c in checklist}
    results = [
        ToolUseResult(
            id=row["id"],
            criterion=by_id.get(row["id"], "(unknown criterion)"),
            verdict=row["verdict"],
            reason=row["reason"],
        )
        for row in rows
    ]
    return ToolUseReport(results=results)


def save_tooluse_score(
    report: ToolUseReport, path: str, checklist_file: str, trajectory_file: str
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "trajectory_file": trajectory_file,
        "checklist_file": checklist_file,
        "results": [
            {"id": r.id, "criterion": r.criterion, "verdict": r.verdict, "reason": r.reason}
            for r in report.results
        ],
        "met": report.met,
        "total": report.total,
        "score": round(report.fraction, 4),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


_MARK = {"met": "✓ met    ", "partial": "~ partial", "not_met": "✗ not met"}


def display_tooluse_score(report: ToolUseReport) -> None:
    print("=" * 70)
    print("TOOL-USE CORRECTNESS SCORING")
    print("=" * 70)
    for n, r in enumerate(report.results, start=1):
        print(f"{n}. {_MARK.get(r.verdict, r.verdict)}  [{r.id}]")
        print(f"   reason: {r.reason}")
    print("-" * 70)
    print(f"Criteria met: {report.met} of {report.total}")
    print(f"Tool-use correctness score = {report.met}/{report.total} = {report.fraction:.2f}")
    print("(only 'met' counts; 'partial' is not a pass)")
    print("=" * 70)
