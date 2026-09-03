"""
Read a trajectory file and turn it into structured turns.

A "trajectory" is a transcript of an AI agent handling ONE user request. Each line
starts with a role tag:

    USER:   what the user said
    AGENT:  what the agent said
    TOOL:   a tool/action the agent invoked, e.g. get_order(order_id=8341)
    RESULT: what that tool returned

Lines without a tag continue the previous turn (so a multi-line message pasted from
somewhere still parses).

Domain-neutral: this parser knows nothing about any particular product.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

# The role tags we recognise at the start of a line.
KNOWN_ROLES = ("USER", "AGENT", "TOOL", "RESULT")


@dataclass
class Turn:
    """One step in the trajectory: one role, one block of text."""

    role: str            # USER / AGENT / TOOL / RESULT
    text: str            # everything after the tag, joined across continuation lines
    line_number: int     # where this turn started in the file (1-based), for reference

    # For TOOL turns only: the parsed call, e.g. name="get_order", args_text="order_id=8341"
    tool_name: str | None = None
    tool_args_text: str | None = None


@dataclass
class Trajectory:
    """A whole transcript = an ordered list of turns."""

    turns: list[Turn] = field(default_factory=list)

    @property
    def opening_user_message(self) -> str | None:
        """The first thing the user said. The checklist is built from this."""
        for turn in self.turns:
            if turn.role == "USER":
                return turn.text
        return None

    @property
    def tool_calls(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "TOOL"]


def _split_tag(line: str) -> tuple[str, str] | None:
    """
    If the line begins with a known tag like 'AGENT:', return (ROLE, rest-of-line).
    Otherwise return None (meaning: this is a continuation of the previous turn).
    """
    if ":" not in line:
        return None
    head, rest = line.split(":", 1)
    if head.strip().upper() in KNOWN_ROLES:
        return head.strip().upper(), rest.strip()
    return None


def _parse_tool_call(text: str) -> tuple[str | None, str | None]:
    """
    Turn 'get_order(order_id=8341)' into ('get_order', 'order_id=8341').
    If it doesn't look like name(...), just return (None, None) and keep the raw text.
    """
    text = text.strip()
    if "(" in text and text.endswith(")"):
        name, args = text.split("(", 1)
        return name.strip(), args[:-1].strip()
    return None, None


def parse_trajectory(raw: str) -> Trajectory:
    traj = Trajectory()
    current: Turn | None = None

    for i, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue  # blank lines are just spacing

        tagged = _split_tag(stripped)
        if tagged is not None:
            role, content = tagged
            current = Turn(role=role, text=content, line_number=i)
            if role == "TOOL":
                current.tool_name, current.tool_args_text = _parse_tool_call(content)
            traj.turns.append(current)
        elif current is not None:
            # continuation line: append to the turn in progress
            current.text = (current.text + " " + stripped).strip()
        else:
            # text before any tag — treat it as a stray USER line so nothing is lost
            current = Turn(role="USER", text=stripped, line_number=i)
            traj.turns.append(current)

    return traj


def display(traj: Trajectory) -> None:
    label_width = max(len(t.role) for t in traj.turns) if traj.turns else 6

    print("=" * 70)
    print(f"TRAJECTORY — {len(traj.turns)} turns, {len(traj.tool_calls)} tool call(s)")
    print("=" * 70)

    for n, turn in enumerate(traj.turns, start=1):
        tag = turn.role.ljust(label_width)
        print(f"{n:>2}. {tag} | {turn.text}")
        if turn.role == "TOOL" and turn.tool_name:
            print(f"    {' ' * label_width} |   ↳ tool: {turn.tool_name}   args: {turn.tool_args_text or '(none)'}")

    print("-" * 70)
    opening = traj.opening_user_message
    print("Opening user message (the basis for the checklist):")
    print(f"  {opening!r}" if opening else "  (none found)")
    print("=" * 70)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python trajectory.py <trajectory-file.txt>")
        return 1

    path = argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        print(f"could not read {path}: {e}")
        return 1

    traj = parse_trajectory(raw)
    display(traj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
