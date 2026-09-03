"""
Communication Quality dimension — CLI.

Runs on any existing trajectory's final agent message. No per-trajectory checklist
generation — the five criteria are fixed. Does not touch other dimensions' code.

usage:
    python run_comm.py <trajectory-file.txt>

Scores the final agent message on the five fixed criteria and writes
comm_scores/<name>.json. Needs an Anthropic API key.
"""

from __future__ import annotations

import os
import sys

from comm_quality import (
    comm_score_path_for,
    display_comm_score,
    final_agent_message,
    save_comm_score,
    score_comm_quality,
)
from trajectory import parse_trajectory


def main(argv: list[str]) -> int:
    args = argv[1:]
    if len(args) != 1:
        print(__doc__)
        return 1

    traj_path = args[0]
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            traj = parse_trajectory(f.read())
    except OSError as e:
        print(f"could not read trajectory {traj_path}: {e}")
        return 1

    final_message = final_agent_message(traj)
    if not final_message:
        print("No final agent message found — nothing to score.")
        return 1

    print(f'FINAL MESSAGE:\n  "{final_message}"\n')
    print("Scoring communication quality (asking Claude)...")
    try:
        report = score_comm_quality(final_message)
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not score: {e}")
        print(
            "\nMost likely there's no API key set. Add one with:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "then run this command again."
        )
        return 1

    print()
    display_comm_score(report)
    sc_path = comm_score_path_for(traj_path)
    save_comm_score(report, sc_path, trajectory_file=traj_path, final_message=final_message)
    print(f"\nSaved score to {sc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
