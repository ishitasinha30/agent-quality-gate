"""
Reasoning Quality dimension — CLI.

usage:
    python run_reasoning.py <trajectory-file.txt>            # step 1: identify reasoning steps
    python run_reasoning.py <trajectory-file.txt> --score    # + step 2: check each step

Step 1 identifies the reasoning steps (inferences, calculations, judgment calls) and saves
them to reasoning_steps/<name>.json. --score then checks each scorable step and writes
reasoning_scores/<name>.json. An empty step list means the dimension is not applicable to
that trajectory. Needs an Anthropic API key.
"""

from __future__ import annotations

import os
import sys

from reasoning_steps import (
    display_reasoning_steps,
    identify_reasoning_steps,
    load_reasoning_steps,
    reasoning_steps_path_for,
    save_reasoning_steps,
)
from trajectory import parse_trajectory


def main(argv: list[str]) -> int:
    args = argv[1:]
    want_score = "--score" in args
    args = [a for a in args if a != "--score"]
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

    steps_path = reasoning_steps_path_for(traj_path)
    if os.path.exists(steps_path):
        print(f"Using existing reasoning steps: {steps_path}")
        print("(delete it to re-identify)")
        steps = load_reasoning_steps(steps_path)
    else:
        print("Identifying reasoning steps (asking Claude)...")
        try:
            steps = identify_reasoning_steps(traj)
        except Exception as e:  # noqa: BLE001
            print(f"\nCould not identify reasoning steps: {e}")
            print(
                "\nMost likely there's no API key set. Add one with:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                "then run this command again."
            )
            return 1
        save_reasoning_steps(steps, steps_path)
        print(f"Saved reasoning steps to {steps_path}")

    print()
    display_reasoning_steps(steps)

    if want_score:
        from reasoning_score import (
            display_reasoning_score,
            reasoning_score_path_for,
            save_reasoning_score,
            score_reasoning,
        )

        print("\nChecking each reasoning step against the SAVED steps (asking Claude)...")
        try:
            report = score_reasoning(steps, traj)
        except Exception as e:  # noqa: BLE001
            print(f"\nCould not score: {e}")
            return 1
        print()
        display_reasoning_score(report)
        sc_path = reasoning_score_path_for(traj_path)
        save_reasoning_score(report, sc_path, steps_file=steps_path, trajectory_file=traj_path)
        print(f"\nSaved score to {sc_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
