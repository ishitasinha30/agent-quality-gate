"""
Decision Engine — fold the six dimension scores into one verdict.

Reads the per-dimension score files already on disk (scores/, tooluse_scores/,
grounding_scores/, reasoning_scores/, comm_scores/, policy_scores/). It does NOT re-run
any evaluator or touch their code.

  1. renormalized weighted average
  2. critical override rules
  3. the PASS / RETRY / REPAIR / HUMAN_REVIEW label   (plus INCOMPLETE, see below)

NOT-APPLICABLE vs NOT-RUN
  Two dimensions can legitimately have nothing to judge and are excluded from the maths
  when so marked:
    - reasoning_quality   (no inference / calculation in the trajectory)
    - policy_compliance   (no policy rule was triggered)
  The other four ALWAYS produce a score. A missing file for one of those is a gap in the
  evaluation, not a not-applicable — the trajectory is reported INCOMPLETE and cannot PASS
  until it is scored.

Critical overrides (PRD Section 4): regardless of the weighted average, a trajectory is
BLOCKED FROM PASS if any of these hold —
  - a policy-compliance violation occurred (score 0.00 on an applicable policy rule)
  - grounding found a contradicted claim (an outright false statement, not merely an
    unconfirmed one)
  - task completion was a complete failure (0.00)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

# name -> (folder holding <trajectory>.json, weight %, may_legitimately_be_not_applicable)
DIMENSIONS: dict[str, tuple[str, int, bool]] = {
    "task_completion":       ("scores",           25, False),
    "reasoning_quality":     ("reasoning_scores",  25, True),
    "tool_use_correctness":  ("tooluse_scores",    15, False),
    "grounding":             ("grounding_scores",  15, False),
    "policy_compliance":     ("policy_scores",     15, True),
    "communication_quality": ("comm_scores",        5, False),
}

# Label thresholds (PRD Section 6).
PASS_MIN = 0.85          # no override + avg >= this -> PASS
SALVAGEABLE_MIN = 0.60   # override fired: avg >= this -> REPAIR, else HUMAN_REVIEW
RETRY_FLOOR = 0.50       # no override: avg below this -> HUMAN_REVIEW instead of RETRY
REVIEW_BAND = (0.55, 0.70)  # override + avg in here: the REPAIR/HUMAN_REVIEW call is close


def _load(folder: str, traj_name: str) -> dict | None:
    path = os.path.join(folder, f"{traj_name}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


@dataclass
class DimRow:
    name: str
    weight: int
    optional: bool
    score: float | None
    state: str               # "scored" | "not_applicable" | "missing"
    note: str

    @property
    def included(self) -> bool:
        return self.state == "scored"


@dataclass
class WeightedResult:
    rows: list[DimRow]
    weighted_average: float | None
    total_weight: int
    missing_required: list[str] = field(default_factory=list)

    @property
    def weighted_sum(self) -> float:
        return sum(r.score * r.weight for r in self.rows if r.included and r.score is not None)

    @property
    def incomplete(self) -> bool:
        return bool(self.missing_required) or self.weighted_average is None


def _read_row(name: str, folder: str, weight: int, optional: bool, traj_name: str) -> DimRow:
    data = _load(folder, traj_name)
    if data is None:
        state = "not_applicable" if optional else "missing"
        note = "no score file — assumed not applicable" if optional else "REQUIRED — not scored yet"
        return DimRow(name, weight, optional, None, state, note)
    if data.get("applicable") is False:
        return DimRow(name, weight, optional, None, "not_applicable", "marked not applicable")
    score = data.get("score")
    if score is None:
        return DimRow(name, weight, optional, None, "not_applicable", "score is null")
    return DimRow(name, weight, optional, float(score), "scored", "")


def weighted_average(traj_name: str) -> WeightedResult:
    rows = [_read_row(n, f, w, opt, traj_name) for n, (f, w, opt) in DIMENSIONS.items()]
    total_weight = sum(r.weight for r in rows if r.included)
    wsum = sum(r.score * r.weight for r in rows if r.included and r.score is not None)
    avg = (wsum / total_weight) if total_weight else None
    missing_required = [r.name for r in rows if r.state == "missing"]
    return WeightedResult(rows, avg, total_weight, missing_required)


# ---------------------------------------------------------------- overrides (PRD Section 4)

@dataclass
class Override:
    id: str
    triggered: bool
    detail: str


def check_overrides(traj_name: str) -> list[Override]:
    out: list[Override] = []

    pol = _load("policy_scores", traj_name)
    if pol is None:
        out.append(Override("policy_violation", False, "no policy score on disk"))
    elif not pol.get("applicable"):
        out.append(Override("policy_violation", False, "no policy rule applied to this trajectory"))
    elif float(pol.get("score") or 0.0) == 0.0:
        rules = [r["id"] for r in pol.get("results", []) if r.get("verdict") == "violation"]
        which = ", ".join(rules) or "an applicable rule"
        out.append(Override("policy_violation", True, f"policy compliance 0.00 — violated: {which}"))
    else:
        out.append(Override("policy_violation", False, f"policy compliance {pol.get('score')}, no violation"))

    grd = _load("grounding_scores", traj_name)
    if grd is None:
        out.append(Override("contradicted_claim", False, "no grounding score on disk"))
    else:
        n = int(grd.get("contradicted", 0))
        if n > 0:
            claims = [v["claim"] for v in grd.get("verdicts", []) if v.get("label") == "contradicted"]
            example = f' e.g. "{claims[0]}"' if claims else ""
            out.append(Override("contradicted_claim", True, f"{n} contradicted claim(s){example}"))
        else:
            ung = int(grd.get("ungrounded", 0))
            out.append(Override("contradicted_claim", False,
                                f"0 contradicted ({ung} merely ungrounded — does not trigger)"))

    tc = _load("scores", traj_name)
    if tc is None:
        out.append(Override("task_completion_zero", False, "no task-completion score on disk"))
    elif float(tc.get("score") or 0.0) == 0.0:
        out.append(Override("task_completion_zero", True, "task completion 0.00 — complete failure"))
    else:
        out.append(Override("task_completion_zero", False, f"task completion {tc.get('score')}"))

    return out


def blocked_from_pass(overrides: list[Override]) -> bool:
    return any(o.triggered for o in overrides)


# ------------------------------------------------------------------- label (PRD Section 6)

@dataclass
class Decision:
    label: str               # PASS | RETRY | REPAIR | HUMAN_REVIEW | INCOMPLETE
    reason: str
    needs_review: bool        # True => the REPAIR/HUMAN_REVIEW split here is a close call


def decide(res: WeightedResult, overrides: list[Override]) -> Decision:
    if res.incomplete:
        why = ", ".join(res.missing_required) or "no applicable dimensions"
        return Decision("INCOMPLETE",
                        f"required dimension(s) not scored: {why} — score them, then re-run", False)

    avg = res.weighted_average
    fired = [o.id for o in overrides if o.triggered]

    if fired:
        near_line = REVIEW_BAND[0] <= avg <= REVIEW_BAND[1]
        if avg >= SALVAGEABLE_MIN:
            return Decision("REPAIR",
                            f"override(s) {', '.join(fired)} fired but weighted average {avg:.3f} "
                            f"is above {SALVAGEABLE_MIN:.2f} — largely salvageable", near_line)
        return Decision("HUMAN_REVIEW",
                        f"override(s) {', '.join(fired)} fired and weighted average {avg:.3f} "
                        f"is below {SALVAGEABLE_MIN:.2f} — broadly weak", near_line)

    if avg >= PASS_MIN:
        return Decision("PASS", f"weighted average {avg:.3f} >= {PASS_MIN:.2f}, no override", False)
    if avg >= RETRY_FLOOR:
        return Decision("RETRY",
                        f"weighted average {avg:.3f} in [{RETRY_FLOOR:.2f}, {PASS_MIN:.2f}), no "
                        f"override — recoverable gap in an otherwise sound trajectory", False)
    return Decision("HUMAN_REVIEW",
                    f"weighted average {avg:.3f} < {RETRY_FLOOR:.2f}, no override fired — broadly "
                    f"weak, not a recoverable gap", False)


# ---------------------------------------------------------------------------------- display

def display(traj_name: str, res: WeightedResult, overrides: list[Override], dec: Decision) -> None:
    print("=" * 72)
    print(f"DECISION ENGINE — {traj_name}")
    print("=" * 72)
    print(f"{'dimension':<24} {'weight':>7} {'score':>7}   note")
    print("-" * 72)
    for r in res.rows:
        if r.included and r.score is not None:
            print(f"{r.name:<24} {r.weight:>6}% {r.score:>7.2f}   contributes {r.score * r.weight:.2f}")
        else:
            tag = "MISSING" if r.state == "missing" else "n/a"
            print(f"{r.name:<24} {r.weight:>6}% {tag:>7}   {r.note}")
    print("-" * 72)
    if res.weighted_average is None:
        print("weighted average : undefined (no applicable dimensions scored)")
    else:
        print(f"weighted average : {res.weighted_sum:.2f} / {res.total_weight} = {res.weighted_average:.3f}")

    print()
    print("CRITICAL OVERRIDES (PRD Section 4)")
    print("-" * 72)
    for o in overrides:
        print(f"{'TRIGGERED' if o.triggered else 'clear    ':<11}  [{o.id}]  {o.detail}")

    print()
    print("-" * 72)
    print(f"DECISION: {dec.label}")
    print(f"  {dec.reason}")
    if dec.needs_review:
        print("  ** REPAIR / HUMAN_REVIEW split is a close call here — flag for a human. **")
    print("=" * 72)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python decision_engine.py <trajectory-name-or-path>")
        return 1
    traj_name = os.path.splitext(os.path.basename(argv[1]))[0]
    res = weighted_average(traj_name)
    overrides = check_overrides(traj_name)
    display(traj_name, res, overrides, decide(res, overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
