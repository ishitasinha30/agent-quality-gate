"""
Compare each dimension's evaluator output against golden_dataset.json.

golden_dataset.json holds hand-verified verdicts. This script runs (or reads) the
evaluator's own output for the same trajectory + dimension and reports where they agree
and where they disagree — the accuracy check the dimension PRDs call for.

usage:
    python compare_to_golden.py                      # every scored golden entry, from saved output
    python compare_to_golden.py --only vbcr          # only status == verified_by_claude_reasoning
    python compare_to_golden.py --only verified      # only status == verified
    python compare_to_golden.py --live               # regenerate evaluator output (needs API key)

Default reads the saved score artifact for each dimension (scores/, tooluse_scores/,
grounding_scores/, reasoning_scores/, comm_scores/). --live re-runs the evaluators first.
"""

from __future__ import annotations

import json
import os
import sys

GOLDEN = "golden_dataset.json"
TOL = 0.02  # |golden - live| within this is an AGREE (absorbs 0.3333 vs 0.33 rounding)

# golden trajectory_file paths are best-guess; map the ones that differ from the repo.
PATH_OVERRIDES = {
    "trajectories/multi_issue.txt": "trajectories/multi_issue_missing_and_damaged.txt",
}

# dimension -> where its saved score lives
DIM_DIR = {
    "task_completion": "scores",
    "tool_use_correctness": "tooluse_scores",
    "grounding": "grounding_scores",
    "reasoning_quality": "reasoning_scores",
    "communication_quality": "comm_scores",
    "policy_compliance": "policy_scores",
}

STATUS_ALIASES = {
    "vbcr": "verified_by_claude_reasoning",
    "vcr": "verified_by_claude_reasoning",
    "verified_by_claude_reasoning": "verified_by_claude_reasoning",
    "verified": "verified",
}


def resolve_traj(entry: dict) -> str:
    p = entry["trajectory_file"]
    return PATH_OVERRIDES.get(p, p)


def traj_name(traj_path: str) -> str:
    return os.path.splitext(os.path.basename(traj_path))[0]


def live_run(dimension: str, traj_path: str) -> None:
    """Regenerate the saved score artifact for one dimension by calling its evaluator."""
    from trajectory import parse_trajectory

    with open(traj_path, "r", encoding="utf-8") as f:
        traj = parse_trajectory(f.read())
    name = traj_name(traj_path)

    if dimension == "communication_quality":
        from comm_quality import comm_score_path_for, final_agent_message, save_comm_score, score_comm_quality
        msg = final_agent_message(traj)
        rep = score_comm_quality(msg)
        save_comm_score(rep, comm_score_path_for(traj_path), traj_path, msg)
    elif dimension == "grounding":
        from grounding_claims import claims_path_for, extract_claims, load_claims, save_claims, _final_agent_message
        from grounding_check import check_grounding, save_grounding_score
        cp = claims_path_for(traj_path)
        claims = load_claims(cp) if os.path.exists(cp) else extract_claims(traj)
        if not os.path.exists(cp):
            save_claims(claims, cp, _final_agent_message(traj))
        rep = check_grounding(claims, traj)
        save_grounding_score(rep, os.path.join("grounding_scores", f"{name}.json"), cp, traj_path)
    elif dimension == "tool_use_correctness":
        from config import load_config
        from tooluse_checklist import generate_tooluse_checklist, load_tooluse_checklist, save_tooluse_checklist, tooluse_checklist_path_for
        from tooluse_score import save_tooluse_score, score_tooluse
        cfg = load_config()
        cp = tooluse_checklist_path_for(traj_path)
        items = load_tooluse_checklist(cp) if os.path.exists(cp) else generate_tooluse_checklist(traj.opening_user_message, cfg)
        if not os.path.exists(cp):
            save_tooluse_checklist(items, cp, traj.opening_user_message)
        rep = score_tooluse(items, traj, cfg)
        save_tooluse_score(rep, os.path.join("tooluse_scores", f"{name}.json"), cp, traj_path)
    elif dimension == "reasoning_quality":
        from reasoning_steps import identify_reasoning_steps, load_reasoning_steps, reasoning_steps_path_for, save_reasoning_steps
        from reasoning_score import save_reasoning_score, score_reasoning
        sp = reasoning_steps_path_for(traj_path)
        steps = load_reasoning_steps(sp) if os.path.exists(sp) else identify_reasoning_steps(traj)
        if not os.path.exists(sp):
            save_reasoning_steps(steps, sp)
        rep = score_reasoning(steps, traj)
        save_reasoning_score(rep, os.path.join("reasoning_scores", f"{name}.json"), sp, traj_path)
    elif dimension == "task_completion":
        from config import load_config
        from checklist import checklist_path_for, generate_checklist, load_checklist, save_checklist
        from score import save_score, score_trajectory
        cfg = load_config()
        cp = checklist_path_for(traj_path)
        items = load_checklist(cp) if os.path.exists(cp) else generate_checklist(traj.opening_user_message, cfg)
        if not os.path.exists(cp):
            save_checklist(items, cp, traj.opening_user_message)
        rep = score_trajectory(items, traj, cfg)
        save_score(rep, os.path.join("scores", f"{name}.json"), checklist_file=cp, trajectory_file=traj_path)


def read_live_score(dimension: str, traj_path: str) -> tuple[str, float | None]:
    """Return (state, score) from the saved artifact. state in {ok, missing, not_applicable}."""
    path = os.path.join(DIM_DIR[dimension], f"{traj_name(traj_path)}.json")
    if not os.path.exists(path):
        return "missing", None
    data = json.load(open(path, encoding="utf-8"))
    # reasoning_quality and policy_compliance can be genuinely not-applicable to a trajectory
    if data.get("applicable") is False or data.get("score") is None:
        return "not_applicable", None
    return "ok", data.get("score")


def classify(golden_status: str, golden_score, live_state: str, live_score):
    if golden_status == "not_applicable":
        if live_state in ("not_applicable", "missing"):
            return "N/A"          # golden says nothing to score; evaluator has nothing either
        return "DISAGREE"         # golden says N/A but the evaluator produced a score
    if live_state == "missing":
        return "MISSING"
    if live_state == "not_applicable":
        return "DISAGREE"  # golden has a score, evaluator says N/A
    if golden_score is None or live_score is None:
        return "MISSING"
    return "AGREE" if abs(golden_score - live_score) <= TOL else "DISAGREE"


def main(argv: list[str]) -> int:
    only = None
    live = "--live" in argv
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            only = STATUS_ALIASES.get(argv[i + 1].lower())
            if only is None:
                print(f"unknown --only value: {argv[i + 1]} (use verified | vbcr)")
                return 1

    golden = json.load(open(GOLDEN, encoding="utf-8"))
    rows = []
    for entry in golden["trajectories"]:
        traj_path = resolve_traj(entry)
        traj_ok = os.path.exists(traj_path)
        for dim, gd in entry["dimensions"].items():
            status = gd.get("status")
            if status == "pending":
                continue
            if only and status != only:
                continue
            if status not in ("verified", "verified_by_claude_reasoning", "not_applicable"):
                continue

            if not traj_ok:
                rows.append((entry["id"], dim, status, gd.get("score"), "NO TRAJ", None, "MISSING"))
                continue
            if live and status != "not_applicable":
                try:
                    live_run(dim, traj_path)
                except Exception as e:  # noqa: BLE001
                    rows.append((entry["id"], dim, status, gd.get("score"), "ERROR", None, f"ERROR: {e}"))
                    continue
            state, lscore = read_live_score(dim, traj_path)
            verdict = classify(status, gd.get("score"), state, lscore)
            rows.append((entry["id"], dim, status, gd.get("score"), state, lscore, verdict))

    # ---- report ----
    print("=" * 100)
    print(f"{'trajectory':<26} {'dimension':<22} {'golden':>7} {'live':>7}  {'status':<26} verdict")
    print("-" * 100)
    tally = {"AGREE": 0, "DISAGREE": 0, "MISSING": 0, "N/A": 0}
    disagreements = []
    for tid, dim, gstatus, gscore, lstate, lscore, verdict in rows:
        gs = "n/a" if gscore is None else f"{gscore:.2f}"
        ls = "n/a" if lscore is None else (f"{lscore:.2f}" if lstate == "ok" else lstate)
        print(f"{tid:<26} {dim:<22} {gs:>7} {ls:>7}  {gstatus:<26} {verdict}")
        tally[verdict.split()[0]] = tally.get(verdict.split()[0], 0) + 1
        if verdict.startswith("DISAGREE"):
            disagreements.append((tid, dim, gscore, lscore))
    print("-" * 100)
    scored = tally.get("AGREE", 0) + tally.get("DISAGREE", 0)
    acc = f"{tally.get('AGREE',0)}/{scored} = {tally.get('AGREE',0)/scored:.0%}" if scored else "n/a"
    print(
        f"AGREE: {tally.get('AGREE',0)}   DISAGREE: {tally.get('DISAGREE',0)}   "
        f"MISSING/ERROR: {tally.get('MISSING',0)}   N/A: {tally.get('N/A',0)}   "
        f"|  agreement on scored entries: {acc}"
    )

    if disagreements:
        print("\nDISAGREEMENTS:")
        gmap = {e["id"]: e["dimensions"] for e in golden["trajectories"]}
        for tid, dim, gscore, lscore in disagreements:
            print(f"\n  {tid} / {dim}")
            print(f"    golden: {gscore}   live: {lscore}")
            print(f"    golden reason: {gmap[tid][dim].get('reason')}")
            lp = os.path.join(DIM_DIR[dim], f"{tid}.json")
            if os.path.exists(lp):
                ld = json.load(open(lp, encoding="utf-8"))
                print(f"    live file:     {lp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
