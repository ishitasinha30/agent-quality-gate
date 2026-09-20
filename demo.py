"""
demo.py — one command to show a trajectory end-to-end, for live demos.

Prints the transcript, then the full decision-engine breakdown (weighted average,
overrides, final label). Reads only already-saved score files — no API key needed,
nothing live, nothing that can hang or fail on a call.

usage:
    python3 demo.py --list             # which trajectories are fully scored and demo-ready
    python3 demo.py <trajectory-name>  # show that one (name only, no .txt, no folder)
"""

from __future__ import annotations

import os
import sys

from decision_engine import DIMENSIONS, check_overrides, decide, display, weighted_average


def list_ready() -> None:
    print(f"{'trajectory':<34} {'label':<14} dimensions scored")
    print("-" * 80)
    for fn in sorted(os.listdir("trajectories")):
        if not fn.endswith(".txt"):
            continue
        name = fn[:-4]
        res = weighted_average(name)
        dec = decide(res, check_overrides(name))
        scored = sum(1 for r in res.rows if r.included)
        flag = "  <- ready" if dec.label != "INCOMPLETE" else ""
        print(f"{name:<34} {dec.label:<14} {scored}/{len(DIMENSIONS)}{flag}")


def show(name: str) -> int:
    path = os.path.join("trajectories", f"{name}.txt")
    if not os.path.exists(path):
        print(f"No such trajectory: {path}")
        print("Run 'python3 demo.py --list' to see what's available.")
        return 1

    print("#" * 72)
    print(f"# {name}")
    print("#" * 72)
    print(open(path, encoding="utf-8").read().strip())
    print()

    res = weighted_average(name)
    overrides = check_overrides(name)
    display(name, res, overrides, decide(res, overrides))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 1
    if argv[1] == "--list":
        list_ready()
        return 0
    return show(argv[1])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
