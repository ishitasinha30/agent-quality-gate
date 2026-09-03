"""
Domain config loading.

The evaluators never hardcode a domain. Everything specific to the system under
evaluation — its tools, the state machine of whatever entity it acts on, its business
constraints and policy rules — lives in a .toml config. Point the evaluators at a
different config and the same code evaluates a different product.

See config/TEMPLATE.toml for the full schema and config/quick_commerce.toml for a worked
example. Set the AQG_CONFIG env var to change the default config path.

TOML is a plain-text config format. Python 3.11+ reads it with the built-in `tomllib`.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field

# Default config path. Override with the AQG_CONFIG env var or a positional CLI arg.
DEFAULT_CONFIG_PATH = os.environ.get("AQG_CONFIG") or "config/quick_commerce.toml"


@dataclass
class Tool:
    name: str
    effect: str          # "read" (looks something up) or "write" (changes state / money / people)
    description: str


@dataclass
class Constraint:
    id: str
    rule: str            # a plain-English sentence the agent must respect


@dataclass
class PolicyRule:
    id: str
    rule: str
    applies_when: str
    violation_example: str = ""


@dataclass
class DomainConfig:
    name: str
    description: str
    tools: list[Tool]
    states: list[str]                       # lifecycle states of the entity the agent acts on
    states_label: str = "states"            # e.g. "order states", "ticket states", "case states"
    constraints: list[Constraint] = field(default_factory=list)
    policy_rules: list[PolicyRule] = field(default_factory=list)

    # Back-compat alias — older code referred to cfg.order_states.
    @property
    def order_states(self) -> list[str]:
        return self.states

    def tool(self, name: str) -> Tool | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    @property
    def write_tools(self) -> list[Tool]:
        return [t for t in self.tools if t.effect == "write"]

    def states_arrow(self) -> str:
        return " -> ".join(self.states) if self.states else "(no lifecycle states defined)"

    def summary_for_prompt(self) -> str:
        """Compact, domain-neutral block every evaluator can drop into its prompt."""
        lines = [
            f"SYSTEM UNDER EVALUATION: {self.name}",
            f"WHAT IT DOES: {self.description}" if self.description else "",
            "",
            "TOOLS (name [read/write]: what it does):",
            *(f"- {t.name} [{t.effect}]: {t.description}" for t in self.tools),
        ]
        if self.states:
            lines += ["", f"{self.states_label.upper()}: {self.states_arrow()}"]
        if self.constraints:
            lines += ["", "CONSTRAINTS:", *(f"- {c.rule}" for c in self.constraints)]
        return "\n".join(l for l in lines if l is not None)


def load_config(path: str = DEFAULT_CONFIG_PATH) -> DomainConfig:
    with open(path, "rb") as f:          # tomllib wants the file opened in binary mode
        data = tomllib.load(f)

    tools = [
        Tool(name=t["name"], effect=t.get("effect", "read"), description=t.get("description", ""))
        for t in data.get("tools", [])
    ]
    constraints = [Constraint(id=c["id"], rule=c["rule"]) for c in data.get("constraints", [])]
    policy_rules = [
        PolicyRule(
            id=r["id"],
            rule=r["rule"],
            applies_when=r.get("applies_when", ""),
            violation_example=r.get("violation_example", ""),
        )
        for r in data.get("policy_rules", [])
    ]

    return DomainConfig(
        name=data["name"],
        description=data.get("description", ""),
        tools=tools,
        states=data.get("states", data.get("order_states", [])),   # accept legacy key
        states_label=data.get("states_label", "states"),
        constraints=constraints,
        policy_rules=policy_rules,
    )


def describe(cfg: DomainConfig) -> None:
    print("=" * 70)
    print(f"DOMAIN CONFIG — {cfg.name}")
    print("=" * 70)
    if cfg.description:
        print(cfg.description)
        print()

    print(f"Tools ({len(cfg.tools)}):")
    name_width = max((len(t.name) for t in cfg.tools), default=0)
    for t in cfg.tools:
        print(f"  - {t.name.ljust(name_width)}  [{t.effect:>5}]  {t.description}")
    print()

    if cfg.states:
        print(f"{cfg.states_label.capitalize()} (in sequence):")
        print("  " + "  ->  ".join(cfg.states))
        print()

    print(f"Constraints ({len(cfg.constraints)}):")
    for c in cfg.constraints:
        print(f"  - ({c.id})")
        print(f"      {c.rule}")

    if cfg.policy_rules:
        print()
        print(f"Policy rules ({len(cfg.policy_rules)}):")
        for r in cfg.policy_rules:
            print(f"  - ({r.id})")
            print(f"      {r.rule}")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    describe(load_config(sys.argv[1] if len(sys.argv) == 2 else DEFAULT_CONFIG_PATH))
