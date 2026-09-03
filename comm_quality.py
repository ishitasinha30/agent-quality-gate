"""
Communication Quality dimension — scoring.

Rates how well the agent's FINAL message is written, independent of whether the action was
correct and independent of whether any fact is missing (PRD Sections 1-2).

Unlike the other dimensions there is NO per-trajectory checklist generation: the same five
fixed criteria apply to every message (PRD Section 3).

  1. plain_language           — no jargon / abbreviations / internal shorthand
  2. complete_coherent_response — full sentences, not fragments; also not bloated/repetitive
  3. professional_tone        — not rude, not robotic
  4. internally_consistent    — no self-contradiction
  5. information_present       — CROSS-CHECK ONLY; a missing fact is task completion's
                                concern, not a communication-quality failure

Score = (met + 0.5 * partial) / total. The judgement is one isolated LLM call.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from trajectory import Trajectory
from jsonlist import extract_json_list

MODEL = "claude-opus-5"

CRITERIA = (
    "plain_language",
    "complete_coherent_response",
    "professional_tone",
    "internally_consistent",
    "information_present",
)

SYSTEM_PROMPT = """\
You rate the COMMUNICATION QUALITY of an AI agent's final message to the user: how well it
is written. You are NOT judging whether the underlying action was correct, and NOT judging
whether a fact is missing (that belongs to another dimension).

Score the message against these five fixed criteria:

1. plain_language — Plain, user-facing language. No unexplained jargon, abbreviations, or
   internal shorthand (e.g. "TAT", "bd", "adjustment processed", raw status codes).
2. complete_coherent_response — Reads as a complete, coherent reply in full sentences, not
   clipped fragments. Also NOT bloated or repetitive — restating the same fact several
   times is a failure too.
3. professional_tone — Appropriate and professional: not rude, not robotic; acknowledges
   the user's situation where that is warranted.
4. internally_consistent — Nothing in the message contradicts anything else in it.
5. information_present — CROSS-CHECK ONLY. Is the information the message is trying to
   convey actually stated (the outcome, any amount, any timeframe)? If something is
   missing, note it in the reason, but do NOT hold it against communication quality —
   mark this "met" unless the message is so gutted there is nothing to judge.

Label each "met", "partial", or "not_met", with a one-sentence reason.
Do NOT penalise a message for being short if it is clear (brevity is fine). Do NOT
penalise it for omitting a fact that was never required.

Respond with ONLY a JSON array. Each element:
  {"id": "<criterion id>", "verdict": "met" | "partial" | "not_met", "reason": "<one sentence>"}"""


@dataclass
class CommResult:
    id: str
    verdict: str
    reason: str


@dataclass
class CommReport:
    results: list[CommResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def points(self) -> float:
        return sum(
            1.0 if r.verdict == "met" else 0.5 if r.verdict == "partial" else 0.0
            for r in self.results
        )

    @property
    def fraction(self) -> float:
        return self.points / self.total if self.total else 0.0


def final_agent_message(traj: Trajectory) -> str | None:
    for turn in reversed(traj.turns):
        if turn.role == "AGENT":
            return turn.text
    return None




def score_comm_quality(final_message: str) -> CommReport:
    """Ask Claude to rate the final message on the five fixed criteria. Raises if no creds."""
    from anthropic import Anthropic

    client = Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f'FINAL MESSAGE:\n"{final_message}"\n\nRate it.'}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    rows = extract_json_list(text)
    return CommReport(results=[CommResult(id=r["id"], verdict=r["verdict"], reason=r["reason"]) for r in rows])


def comm_score_path_for(trajectory_path: str) -> str:
    name = os.path.splitext(os.path.basename(trajectory_path))[0]
    return os.path.join("comm_scores", f"{name}.json")


def save_comm_score(report: CommReport, path: str, trajectory_file: str, final_message: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "trajectory_file": trajectory_file,
        "final_message": final_message,
        "results": [{"id": r.id, "verdict": r.verdict, "reason": r.reason} for r in report.results],
        "points": report.points,
        "total": report.total,
        "score": round(report.fraction, 4),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


_MARK = {"met": "✓ met    ", "partial": "~ partial", "not_met": "✗ not met"}


def display_comm_score(report: CommReport) -> None:
    print("=" * 70)
    print("COMMUNICATION QUALITY SCORING")
    print("=" * 70)
    for n, r in enumerate(report.results, start=1):
        print(f"{n}. {_MARK.get(r.verdict, r.verdict)}  [{r.id}]")
        print(f"   reason: {r.reason}")
    print("-" * 70)
    print(f"Points: {report.points} of {report.total}")
    print(f"Communication quality score = {report.points}/{report.total} = {report.fraction:.2f}")
    print("(met = 1, partial = 0.5, not_met = 0; criterion 5 is a cross-check, not a penalty)")
    print("=" * 70)
