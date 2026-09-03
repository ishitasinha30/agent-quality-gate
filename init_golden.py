"""
Scaffold golden_dataset.json from the trajectories/ folder.

Run this after adding transcripts. For every .txt in trajectories/ it ensures there is an
entry in golden_dataset.json with all six dimensions present, each set to status
"pending" / score null. You then fill in the numbers and reasons by hand — deciding those
independently of the evaluator is the whole point of a golden set.

Existing entries and any already-filled dimension values are never modified; this only
adds what is missing.

    python3 init_golden.py [golden_dataset.json] [trajectories/]
"""

from __future__ import annotations

import json
import os
import sys

DIMENSIONS = (
    "task_completion",
    "tool_use_correctness",
    "grounding",
    "reasoning_quality",
    "communication_quality",
    "policy_compliance",
)

README = (
    "Hand-verified verdicts per trajectory and dimension, decided independently of what "
    "the evaluator produces. Not written by the evaluator. status 'pending' means you have "
    "not reviewed it yet: set score + a reason and change status to 'verified' once you "
    "have, or to 'not_applicable' with score null when the dimension has nothing to judge "
    "for that trajectory. compare_to_golden.py diffs the evaluator's live output against "
    "this file and ignores 'pending' rows."
)


def blank_dimension() -> dict:
    return {"score": None, "status": "pending", "reason": None}


def main(argv: list[str]) -> int:
    golden_path = argv[1] if len(argv) > 1 else "golden_dataset.json"
    traj_dir = argv[2] if len(argv) > 2 else "trajectories"

    if not os.path.isdir(traj_dir):
        print(f"no such directory: {traj_dir}/")
        return 1

    if os.path.exists(golden_path):
        with open(golden_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data.setdefault("_readme", README)
    entries = data.setdefault("trajectories", [])
    by_id = {e["id"]: e for e in entries}

    txts = sorted(f for f in os.listdir(traj_dir) if f.endswith(".txt"))
    if not txts:
        print(f"no .txt files in {traj_dir}/")
        return 1

    added: list[str] = []
    extended: list[str] = []
    for fn in txts:
        tid = os.path.splitext(fn)[0]
        rel = os.path.join(traj_dir, fn)
        entry = by_id.get(tid)
        if entry is None:
            entries.append(
                {
                    "id": tid,
                    "trajectory_file": rel,
                    "intent": "",
                    "dimensions": {d: blank_dimension() for d in DIMENSIONS},
                }
            )
            added.append(tid)
        else:
            dims = entry.setdefault("dimensions", {})
            new = [d for d in DIMENSIONS if d not in dims]
            for d in new:
                dims[d] = blank_dimension()
            if new:
                extended.append(f"{tid} (+{', '.join(new)})")

    txt_ids = {os.path.splitext(f)[0] for f in txts}
    orphans = [e["id"] for e in entries if e["id"] not in txt_ids]

    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    pending = sum(
        1
        for e in entries
        for d in e.get("dimensions", {}).values()
        if d.get("status") == "pending"
    )
    print(f"{golden_path}: {len(entries)} trajectory entries")
    print(f"  new skeletons     : {', '.join(added) if added else '(none)'}")
    print(f"  dimensions filled : {', '.join(extended) if extended else '(none)'}")
    if orphans:
        print(f"  in golden, no .txt: {', '.join(orphans)}  (left untouched)")
    print(f"  {pending} dimension rows are 'pending' — fill in score + status + reason")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
