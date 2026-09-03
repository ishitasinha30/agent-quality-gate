"""
Task Completion dimension — step 2: score a trajectory against its checklist.

Walk each checklist criterion, look for evidence in the transcript, and mark it:
  met      — the transcript clearly satisfies this criterion
  partial  — the agent did something adjacent but not what the criterion requires
             (e.g. delivered a different resolution than the one asked for)
  not_met  — no evidence, or the agent did the wrong thing

Only "met" counts toward the score (a partial fulfilment is not a pass). Score = criteria
met / total criteria — a plain fraction a reviewer can audit line by line.

The judgement is one isolated LLM call: `score_trajectory`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from checklist import ChecklistItem
from config import DomainConfig
from trajectory import Trajectory
from jsonlist import extract_json_list

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You check a COMPLETED agent transcript against a fixed checklist of success criteria. The
checklist was written beforehand from the user's opening message; do not second-guess it.
Your job is only to decide, for each criterion, whether the transcript satisfies it.

For each criterion return one verdict:
- "met"     — the transcript clearly and fully satisfies it, with evidence you can point to.
- "partial" — the agent addressed the same topic but not in the way the criterion requires
              (wrong resolution type, right action but wrong scope, outcome stated but no
              timeline when a consequential action was taken, only one of two issues
              handled, etc.).
- "not_met" — no evidence in the transcript, or the agent did the wrong thing.

Rules:
- Judge ONLY on what the transcript shows. A confident closing message is not evidence
  that the underlying action was correct — check the tool calls and their results.
- Be strict about scope. Acting on more than the user asked for fails the guard criterion
  even if everything else went well.
- If a criterion is about explaining why something impossible can't be done, a clear
  explanation plus an alternative counts as "met".
- Give a one-sentence reason that cites what in the transcript decided it.

Respond with ONLY a JSON array — the whole reply must be one `[ ... ]`, no prose before
or after, no markdown fences, not one object per line. Each element:
  {"id": "<criterion id>", "verdict": "met" | "partial" | "not_met", "reason": "<one sentence>"}"""


@dataclass
class CriterionResult:
    id: str
    criterion: str
    verdict: str          # "met" | "partial" | "not_met"
    reason: str


@dataclass
class ScoreReport:
    results: list[CriterionResult]

    @property
    def met(self) -> int:
        return sum(1 for r in self.results if r.verdict == "met")

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def fraction(self) -> float:
        return self.met / self.total if self.total else 0.0


def _trajectory_as_text(traj: Trajectory) -> str:
    return "\n".join(f"{t.role}: {t.text}" for t in traj.turns)




def score_trajectory(
    checklist: list[ChecklistItem],
    traj: Trajectory,
    cfg: DomainConfig,
) -> ScoreReport:
    """Ask Claude to mark each criterion against the transcript. Raises if no credentials."""
    from anthropic import Anthropic

    checklist_json = json.dumps(
        [{"id": c.id, "criterion": c.criterion} for c in checklist], indent=2
    )
    user_prompt = f"""\
{cfg.summary_for_prompt()}

CHECKLIST (written beforehand from the user's opening message):
{checklist_json}

FULL TRANSCRIPT:
{_trajectory_as_text(traj)}

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
        CriterionResult(
            id=row["id"],
            criterion=by_id.get(row["id"], "(unknown criterion)"),
            verdict=row["verdict"],
            reason=row["reason"],
        )
        for row in rows
    ]
    return ScoreReport(results=results)


def save_score(report: ScoreReport, path: str, checklist_file: str, trajectory_file: str) -> None:
    """Write the score to disk, recording which checklist file it was scored against."""
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


def display_score(report: ScoreReport) -> None:
    print("=" * 70)
    print("TASK-COMPLETION SCORING")
    print("=" * 70)
    for n, r in enumerate(report.results, start=1):
        print(f"{n}. {_MARK.get(r.verdict, r.verdict)}  [{r.id}]")
        print(f"   criterion: {r.criterion}")
        print(f"   reason:    {r.reason}")
    print("-" * 70)
    print(f"Criteria met: {report.met} of {report.total}")
    print(f"Task completion score = {report.met}/{report.total} = {report.fraction:.2f}")
    print("(only 'met' counts; 'partial' is not a pass — PRD Section 8)")
    print("=" * 70)
