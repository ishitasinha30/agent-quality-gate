"""
Policy Compliance dimension — checking.

Checks whether the agent's actions and statements stay within the FIXED BUSINESS RULES in
the domain config's [[policy_rules]] section — checked against the rules, not against tool
results (that is grounding's job). A claim can be fully grounded and still break a rule.

GENERAL RULE (locked decision): policy compliance fires ONLY when a specific named
[[policy_rules]] entry is actually violated. It is never a generic "nothing confirms this
statement" check — an unsupported claim with no specific rule against it is grounding's
job, not this dimension's. If in doubt, mark not_applicable. (Same call already made for
reasoning quality on cancel_after_dispatch_good's unbacked refund promise.)

Most trajectories touch no policy rule -> the dimension is NOT APPLICABLE for them and they
are excluded from the score (same handling as reasoning quality with no calculation).

Score = compliant / applicable. The judgement is one isolated LLM call.

Policy rules live in the [[policy_rules]] section of the domain config, loaded via config.py
like every other domain fact.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from config import DEFAULT_CONFIG_PATH, PolicyRule, load_config
from trajectory import Trajectory
from jsonlist import extract_json_list

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You check whether an AI agent's actions and statements comply with the FIXED BUSINESS
RULES (policy rules) in the domain config. You check against the rules, not against tool
results — a statement can be fully supported by a tool result and still break a rule.

You are given the trajectory and a list of policy rules. For each rule:
1. Decide whether it APPLIES to this trajectory, using its "applies_when" condition and
   what actually happens in the transcript. Match the rule's SPECIFIC prohibition — do not
   stretch a rule to cover a loosely related situation.
2. If it does not apply, return verdict "not_applicable" for that rule.
3. If it applies, return "compliant" or "violation":
   - "violation" — the agent did the thing the rule forbids, or failed to do what it
     requires.
   - "compliant" — the agent stayed within the rule. Correctly escalating to a human when
     no rule covers the situation is compliant, not a failure.
Give a one-sentence reason citing the specific agent action or statement.

NEVER flag a statement just because nothing in the config or tool results confirms it.
An unsupported claim with no specific rule against it is grounding's concern, not yours.
If no listed rule squarely fits, every rule is "not_applicable".

Respond with ONLY a JSON array, one element per rule:
  {"id": "<rule id>", "verdict": "compliant" | "violation" | "not_applicable", "reason": "<one sentence>"}"""


@dataclass
class PolicyResult:
    id: str
    verdict: str            # compliant | violation | not_applicable
    reason: str


@dataclass
class PolicyReport:
    results: list[PolicyResult]

    @property
    def applicable(self) -> list[PolicyResult]:
        return [r for r in self.results if r.verdict in ("compliant", "violation")]

    @property
    def is_applicable(self) -> bool:
        return len(self.applicable) > 0

    @property
    def compliant(self) -> int:
        return sum(1 for r in self.applicable if r.verdict == "compliant")

    @property
    def total(self) -> int:
        return len(self.applicable)

    @property
    def fraction(self) -> float | None:
        return self.compliant / self.total if self.total else None


def load_policy_rules(config_path: str = DEFAULT_CONFIG_PATH) -> list[PolicyRule]:
    """Thin wrapper: policy rules live in the [[policy_rules]] section of the domain config."""
    return load_config(config_path).policy_rules


def _transcript_text(traj: Trajectory) -> str:
    return "\n".join(f"{t.role}: {t.text}" for t in traj.turns)




def check_policy_compliance(traj: Trajectory, rules: list[PolicyRule]) -> PolicyReport:
    """Ask Claude which policy rules apply and whether the agent complied. Raises if no creds."""
    from anthropic import Anthropic

    if not rules:
        return PolicyReport(results=[])

    rules_json = json.dumps(
        [{"id": r.id, "rule": r.rule, "applies_when": r.applies_when, "violation_example": r.violation_example}
         for r in rules],
        indent=2,
    )
    user_prompt = f"""\
POLICY RULES:
{rules_json}

TRANSCRIPT:
{_transcript_text(traj)}

Check every rule."""

    client = Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    rows = extract_json_list(text)
    return PolicyReport(results=[PolicyResult(id=r["id"], verdict=r["verdict"], reason=r["reason"]) for r in rows])


def policy_score_path_for(trajectory_path: str) -> str:
    name = os.path.splitext(os.path.basename(trajectory_path))[0]
    return os.path.join("policy_scores", f"{name}.json")


def save_policy_score(report: PolicyReport, path: str, trajectory_file: str, config_file: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "trajectory_file": trajectory_file,
        "config_file": config_file,
        "applicable": report.is_applicable,
        "results": [{"id": r.id, "verdict": r.verdict, "reason": r.reason} for r in report.results],
        "compliant": report.compliant,
        "total": report.total,
        "score": round(report.fraction, 4) if report.fraction is not None else None,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


_MARK = {"compliant": "✓ compliant   ", "violation": "✗ violation   ", "not_applicable": "· n/a         "}


def display_policy_score(report: PolicyReport) -> None:
    print("=" * 70)
    print("POLICY COMPLIANCE")
    print("=" * 70)
    for n, r in enumerate(report.results, start=1):
        print(f"{n}. {_MARK.get(r.verdict, r.verdict)}  [{r.id}]")
        print(f"   {r.reason}")
    print("-" * 70)
    if not report.is_applicable:
        print("NOT APPLICABLE — no policy rule is triggered by this trajectory.")
        print("(excluded from the policy-compliance score)")
    else:
        print(f"Compliant: {report.compliant} of {report.total} applicable rule(s)")
        print(f"Policy compliance score = {report.compliant}/{report.total} = {report.fraction:.2f}")
    print("=" * 70)
