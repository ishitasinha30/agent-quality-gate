"""
Tool-Use Correctness dimension — CLI.

Reuses the existing domain config and trajectories. Does not touch the task-completion
code. Right now it does checklist generation only; scoring is the next slice.

usage:
    python run_tooluse.py <trajectory-file.txt> [config-file.toml]
    python run_tooluse.py <trajectory-file.txt> --score

Generates the tool-use checklist from the opening user message and saves it to
tooluse_checklists/<name>.json. --score also scores the agent's actual tool calls against
that saved checklist and writes tooluse_scores/<name>.json. Needs an Anthropic API key.
"""

from __future__ import annotations

import os
import sys

from config import DEFAULT_CONFIG_PATH, load_config
from tooluse_checklist import (
    display_tooluse_checklist,
    generate_tooluse_checklist,
    load_tooluse_checklist,
    save_tooluse_checklist,
    tooluse_checklist_path_for,
)
from trajectory import parse_trajectory

DEFAULT_CONFIG = DEFAULT_CONFIG_PATH


def main(argv: list[str]) -> int:
    args = argv[1:]
    want_score = "--score" in args
    args = [a for a in args if a != "--score"]
    if len(args) not in (1, 2):
        print(__doc__)
        return 1

    traj_path = args[0]
    config_path = args[1] if len(args) == 2 else DEFAULT_CONFIG

    try:
        cfg = load_config(config_path)
        with open(traj_path, "r", encoding="utf-8") as f:
            traj = parse_trajectory(f.read())
    except OSError as e:
        print(f"could not read a file: {e}")
        return 1

    opening = traj.opening_user_message
    if not opening:
        print("No opening user message found — cannot build a checklist.")
        return 1

    cl_path = tooluse_checklist_path_for(traj_path)
    if os.path.exists(cl_path):
        print(f"Using existing tool-use checklist: {cl_path}")
        print("(delete it to regenerate)")
        items = load_tooluse_checklist(cl_path)
    else:
        print("Generating tool-use checklist from the opening message (asking Claude)...")
        try:
            items = generate_tooluse_checklist(opening, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"\nCould not generate the checklist: {e}")
            print(
                "\nMost likely there's no API key set. Add one with:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                "then run this command again."
            )
            return 1
        save_tooluse_checklist(items, cl_path, opening)
        print(f"Saved checklist to {cl_path}")

    print()
    display_tooluse_checklist(items)

    if want_score:
        from tooluse_score import display_tooluse_score, save_tooluse_score, score_tooluse

        print("\nScoring the agent's actual tool calls against the SAVED checklist (asking Claude)...")
        try:
            report = score_tooluse(items, traj, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"\nCould not score: {e}")
            return 1
        print()
        display_tooluse_score(report)
        sc_path = os.path.join("tooluse_scores", os.path.basename(cl_path))
        save_tooluse_score(report, sc_path, checklist_file=cl_path, trajectory_file=traj_path)
        print(f"\nSaved score to {sc_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
