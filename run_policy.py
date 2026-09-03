"""
Policy Compliance dimension — CLI.

Reads [[policy_rules]] from the domain config and checks one trajectory against them.
Does not touch any other dimension's code (config.py included).

usage:
    python run_policy.py <trajectory-file.txt> [config-file.toml]

Writes policy_scores/<name>.json. A trajectory that triggers no rule is marked not
applicable and excluded from the score. Needs an Anthropic API key.
"""

from __future__ import annotations

import sys

from config import DEFAULT_CONFIG_PATH
from policy_check import (
    check_policy_compliance,
    display_policy_score,
    load_policy_rules,
    policy_score_path_for,
    save_policy_score,
)
from trajectory import parse_trajectory

DEFAULT_CONFIG = DEFAULT_CONFIG_PATH


def main(argv: list[str]) -> int:
    args = argv[1:]
    if len(args) not in (1, 2):
        print(__doc__)
        return 1

    traj_path = args[0]
    config_path = args[1] if len(args) == 2 else DEFAULT_CONFIG

    try:
        rules = load_policy_rules(config_path)
        with open(traj_path, "r", encoding="utf-8") as f:
            traj = parse_trajectory(f.read())
    except OSError as e:
        print(f"could not read a file: {e}")
        return 1

    print(f"Loaded {len(rules)} policy rule(s) from {config_path}.")
    print("Checking the trajectory against them (asking Claude)...\n")
    try:
        report = check_policy_compliance(traj, rules)
    except Exception as e:  # noqa: BLE001
        print(f"Could not run the policy check: {e}")
        print(
            "\nMost likely there's no API key set. Add one with:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "then run this command again."
        )
        return 1

    display_policy_score(report)
    sc_path = policy_score_path_for(traj_path)
    save_policy_score(report, sc_path, trajectory_file=traj_path, config_file=config_path)
    print(f"\nSaved score to {sc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
