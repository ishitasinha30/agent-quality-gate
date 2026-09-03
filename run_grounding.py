"""
Grounding dimension — CLI.

Reuses the existing config and trajectories. Does not touch task-completion or tool-use
code. Right now it does claim extraction only; grounding scoring is the next slice.

usage:
    python run_grounding.py <trajectory-file.txt>
    python run_grounding.py <trajectory-file.txt> --check

Extracts the factual claims from the agent's final message and saves them to
grounding_claims/<name>.json. --check also labels each claim against the trajectory's tool
results (grounded / contradicted / ungrounded) and writes grounding_scores/<name>.json.
Needs an Anthropic API key.
"""

from __future__ import annotations

import os
import sys

from grounding_claims import (
    claims_path_for,
    display_claims,
    extract_claims,
    load_claims,
    save_claims,
)
from grounding_claims import _final_agent_message
from trajectory import parse_trajectory


def main(argv: list[str]) -> int:
    args = argv[1:]
    want_check = "--check" in args
    args = [a for a in args if a != "--check"]
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

    final_message = _final_agent_message(traj)
    if not final_message:
        print("No final agent message found — nothing to extract.")
        return 1

    cl_path = claims_path_for(traj_path)
    if os.path.exists(cl_path):
        print(f"Using existing extracted claims: {cl_path}")
        print("(delete it to re-extract)")
        claims = load_claims(cl_path)
    else:
        print("Extracting claims from the agent's final message (asking Claude)...")
        try:
            claims = extract_claims(traj)
        except Exception as e:  # noqa: BLE001
            print(f"\nCould not extract claims: {e}")
            print(
                "\nMost likely there's no API key set. Add one with:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                "then run this command again."
            )
            return 1
        save_claims(claims, cl_path, final_message)
        print(f"Saved claims to {cl_path}")

    print()
    print(f'FINAL MESSAGE:\n  "{final_message}"\n')
    display_claims(claims)

    if want_check:
        from grounding_check import check_grounding, display_grounding, save_grounding_score

        print("\nChecking each claim against the trajectory's tool results (asking Claude)...")
        try:
            report = check_grounding(claims, traj)
        except Exception as e:  # noqa: BLE001
            print(f"\nCould not run the grounding check: {e}")
            return 1
        print()
        display_grounding(report)
        sc_path = os.path.join("grounding_scores", os.path.basename(cl_path))
        save_grounding_score(report, sc_path, claims_file=cl_path, trajectory_file=traj_path)
        print(f"\nSaved grounding score to {sc_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
