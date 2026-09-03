"""
Load a domain config AND a trajectory, print both. Optionally generate the checklist.

  - the domain config (what this product is: tools, states, rules)  -> from a .toml file
  - the trajectory (one transcript to evaluate)                     -> from a .txt file

usage:
    python run.py <trajectory-file.txt> [config-file.toml]
    python run.py <trajectory-file.txt> --checklist
    python run.py <trajectory-file.txt> --score

--checklist generates the task-completion checklist from the opening user message
  and SAVES it to checklists/<name>.json.
--score reads that saved checklist (generating + saving it once if it doesn't exist yet)
  and scores the transcript against it. It never regenerates a checklist that already
  exists — delete the file or re-run --checklist to make a fresh one.
Both need an Anthropic API key (see README).

If you leave the config off, it uses $AQG_CONFIG or config/quick_commerce.toml.
"""

from __future__ import annotations

import sys

from config import DEFAULT_CONFIG_PATH, describe, load_config
from trajectory import parse_trajectory, display

DEFAULT_CONFIG = DEFAULT_CONFIG_PATH


def main(argv: list[str]) -> int:
    args = argv[1:]
    want_score = "--score" in args
    want_checklist = want_score or "--checklist" in args
    args = [a for a in args if a not in ("--checklist", "--score")]

    if len(args) not in (1, 2):
        print(__doc__)
        return 1

    traj_path = args[0]
    config_path = args[1] if len(args) == 2 else DEFAULT_CONFIG

    try:
        cfg = load_config(config_path)
    except OSError as e:
        print(f"could not read config {config_path}: {e}")
        return 1

    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        print(f"could not read trajectory {traj_path}: {e}")
        return 1

    describe(cfg)
    print()
    traj = parse_trajectory(raw)
    display(traj)

    # A tiny cross-check: which tools in this trajectory are known to the config?
    print()
    print("Tool calls in this trajectory, checked against the config:")
    for t in traj.tool_calls:
        known = cfg.tool(t.tool_name) if t.tool_name else None
        if known:
            print(f"  ✓ {t.tool_name}  — known, {known.effect} action")
        else:
            print(f"  ? {t.tool_name or t.text}  — NOT in this config")

    if want_checklist:
        import os

        from checklist import (
            checklist_path_for,
            display_checklist,
            generate_checklist,
            load_checklist,
            save_checklist,
        )

        opening = traj.opening_user_message
        if not opening:
            print("\nNo opening user message found — cannot build a checklist.")
            return 1

        cl_path = checklist_path_for(traj_path)
        force_new = "--checklist" in argv[1:] and not want_score

        if os.path.exists(cl_path) and not force_new:
            print(f"\nUsing existing checklist: {cl_path}")
            print("(delete it or run with --checklist to regenerate)")
            items = load_checklist(cl_path)
        else:
            print("\nGenerating checklist from the opening message (asking Claude)...")
            try:
                items = generate_checklist(opening, cfg)
            except Exception as e:  # noqa: BLE001 — surface any failure plainly for a newcomer
                print(f"\nCould not generate the checklist: {e}")
                print(
                    "\nMost likely there's no API key set. Add one with:\n"
                    "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                    "then run this command again."
                )
                return 1
            save_checklist(items, cl_path, opening)
            print(f"Saved checklist to {cl_path}")

        print()
        display_checklist(items)

        if want_score:
            from score import display_score, save_score, score_trajectory

            print("\nScoring the transcript against the SAVED checklist (asking Claude)...")
            try:
                report = score_trajectory(items, traj, cfg)
            except Exception as e:  # noqa: BLE001
                print(f"\nCould not score: {e}")
                return 1
            print()
            display_score(report)
            sc_path = os.path.join("scores", os.path.basename(cl_path))
            save_score(report, sc_path, checklist_file=cl_path, trajectory_file=traj_path)
            print(f"\nSaved score to {sc_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
