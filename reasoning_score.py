"""
Reasoning Quality dimension — step 2: check whether each identified reasoning step was done
correctly.

For each step from reasoning_steps.py:
  - inference   -> does the conclusion follow from the stated inputs?
  - calculation -> is the arithmetic right?
  - guard       -> does the agent's stated figure correspond to a real combination of the
                   input values?

Steps carrying a "scoring_note" (e.g. premise ungrounded -> grounding's job) are SKIPPED
here, not scored. If no scorable steps remain, the dimension is not applicable and the
trajectory is excluded from the reasoning-quality score.

Score = correct / scorable_total. The judgement is one isolated LLM call.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from reasoning_steps import ReasoningStep, load_reasoning_steps
from trajectory import Trajectory
from jsonlist import extract_json_list

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You check whether each REASONING STEP an AI agent took was performed correctly. You are
given the steps (already identified) and the trajectory. Judge only the operation in each
step, using the inputs listed for that step.

Per kind:
- inference   -> mark "correct" if the conclusion follows from the stated inputs, else "incorrect".
- calculation -> do the arithmetic yourself from the stated inputs; "correct" only if the
                 agent's figure matches.
- guard       -> "correct" if the agent's stated figure equals some real combination of the
                 listed input values; "incorrect" if it matches no combination (a number
                 from nowhere).

Also watch for: a multi-step chain where an early step is wrong but a later step happens to
land on a right-looking number — call the early step incorrect anyway and say so.

Do NOT re-judge a step whose premise is unsupported by trajectory data — those are handled
by grounding. (They will not be sent to you.)

Respond with ONLY a JSON array. Each element:
  {"id": "<step id>", "verdict": "correct" | "incorrect", "reason": "<one sentence, show the check>"}"""


@dataclass
class StepResult:
    id: str
    kind: str
    verdict: str            # correct | incorrect
    reason: str


@dataclass
class ReasoningReport:
    results: list[StepResult]
    skipped: list[str]      # step ids not scored (premise ungrounded)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.verdict == "correct")

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def applicable(self) -> bool:
        return self.total > 0

    @property
    def fraction(self) -> float | None:
        return self.correct / self.total if self.total else None


def _transcript_text(traj: Trajectory) -> str:
    return "\n".join(f"{t.role}: {t.text}" for t in traj.turns)




def _scorable(steps: list[ReasoningStep]) -> list[ReasoningStep]:
    # A step carrying a scoring_note is handled by another dimension (e.g. its premise is
    # ungrounded -> grounding's job). Not scored here.
    return [s for s in steps if not s.scoring_note]


def score_reasoning(steps: list[ReasoningStep], traj: Trajectory) -> ReasoningReport:
    """Check each scorable reasoning step. Raises if no credentials."""
    from anthropic import Anthropic

    scorable = _scorable(steps)
    skipped = [s.id for s in steps if s not in scorable]
    if not scorable:
        return ReasoningReport(results=[], skipped=skipped)

    steps_json = json.dumps(
        [{"id": s.id, "kind": s.kind, "step": s.step, "inputs": s.inputs, "conclusion": s.conclusion}
         for s in scorable],
        indent=2,
    )
    user_prompt = f"""\
TRANSCRIPT:
{_transcript_text(traj)}

REASONING STEPS TO CHECK:
{steps_json}

Check every step."""

    client = Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    rows = extract_json_list(text)
    by_kind = {s.id: s.kind for s in scorable}
    results = [
        StepResult(id=r["id"], kind=by_kind.get(r["id"], "?"), verdict=r["verdict"], reason=r["reason"])
        for r in rows
    ]
    return ReasoningReport(results=results, skipped=skipped)


def reasoning_score_path_for(trajectory_path: str) -> str:
    name = os.path.splitext(os.path.basename(trajectory_path))[0]
    return os.path.join("reasoning_scores", f"{name}.json")


def save_reasoning_score(report: ReasoningReport, path: str, steps_file: str, trajectory_file: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "trajectory_file": trajectory_file,
        "steps_file": steps_file,
        "applicable": report.applicable,
        "skipped_steps": report.skipped,
        "results": [{"id": r.id, "kind": r.kind, "verdict": r.verdict, "reason": r.reason} for r in report.results],
        "correct": report.correct,
        "total": report.total,
        "score": round(report.fraction, 4) if report.fraction is not None else None,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


_MARK = {"correct": "✓ correct  ", "incorrect": "✗ incorrect"}


def display_reasoning_score(report: ReasoningReport) -> None:
    print("=" * 70)
    print("REASONING QUALITY SCORING")
    print("=" * 70)
    if not report.applicable:
        print("NOT APPLICABLE — no scorable reasoning steps.")
        if report.skipped:
            print(f"(skipped, handled by grounding: {', '.join(report.skipped)})")
        print("=" * 70)
        return
    for n, r in enumerate(report.results, start=1):
        print(f"{n}. {_MARK.get(r.verdict, r.verdict)}  [{r.id}]  ({r.kind})")
        print(f"   {r.reason}")
    if report.skipped:
        print(f"\nskipped (grounding's job): {', '.join(report.skipped)}")
    print("-" * 70)
    print(f"Correct: {report.correct} of {report.total}")
    print(f"Reasoning quality score = {report.correct}/{report.total} = {report.fraction:.2f}")
    print("=" * 70)
