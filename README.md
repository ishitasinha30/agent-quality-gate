# Agent Quality Gate

You built an AI agent — a support bot, an assistant, something that takes a request and
acts on it with tools. You want to know whether a given run was actually *good*, in a way
you can inspect and defend — not just "the judge model said 7/10".

This is that check. You give it a recorded run (a transcript). It scores that run on six
independent quality dimensions, shows you criterion by criterion what passed and what
failed and why, and (via the decision engine) folds the six into one verdict —
`PASS` / `RETRY` / `REPAIR` / `HUMAN_REVIEW`. It runs after the fact on recorded
transcripts; it does not sit in your agent's live path, and by itself it does not block
anything — it produces the numbers and the verdict you would put a release gate behind.

**Domain-generic.** No product-specific knowledge lives in the evaluator code. The system
under evaluation — its tools, the states its records move through, its rules — is described
in one `.toml` config. Point it at a different config and the same six dimensions evaluate
a different product. `config/quick_commerce.toml` is a filled-in example;
`config/TEMPLATE.toml` is the annotated blank.

## The six dimensions

| Dimension | Question it answers |
|---|---|
| **Task completion** | Did the user get the outcome they asked for? |
| **Tool-use correctness** | Right tools, right parameters, results read correctly, no needless or missing calls? |
| **Grounding** | Is every factual claim in the agent's final message backed by a tool result in this transcript? |
| **Reasoning quality** | Was the agent's own inference / arithmetic / judgment correct? (N/A if it did none) |
| **Communication quality** | Is the final message well written — plain, complete, consistent? |
| **Policy compliance** | Do the agent's decisions stay within the named rules in the config? (N/A if no rule is triggered) |

Each dimension runs in two steps: first it works out what "done" *means* for this specific
request (a checklist, or a list of claims / reasoning steps) and saves that to disk; then
it scores the transcript against that saved artifact. Splitting it keeps the standard
fixed while the transcript is judged, and makes every run reproducible and inspectable.

## What a run looks like

```
$ python run_tooluse.py trajectories/example_missing_item.txt --score

======================================================================
TOOL-USE CORRECTNESS SCORING
======================================================================
1. ✓ met      [correct_tools_selected]
   reason: Called get_order to retrieve the order and issue_refund to remedy it; no
           unrelated write tools.
2. ✓ met      [parameters_correct]
   reason: issue_refund used item="eggs", amount=60 — the eggs' price from the get_order
           result, not the order total.
3. ✓ met      [tool_results_interpreted_correctly]
   reason: The agent read the returned item list correctly and identified eggs (60) as
           the undelivered item.
4. ✓ met      [no_unnecessary_tool_calls]
   reason: Only get_order and issue_refund were called, and both results were used.
5. ✗ not met  [required_verification_call_made]
   reason: get_delivery_status was available but never called before the refund.
----------------------------------------------------------------------
Criteria met: 4 of 5
Tool-use correctness score = 4/5 = 0.80
======================================================================
```

Every dimension produces this shape — a verdict and a one-line reason per criterion, then
a fraction. The same breakdown is written to a JSON file so you can diff it later.

(This is the bundled quick-commerce example; `get_order`, `eggs`, `amount=60` etc. come
from that config and transcript, not from the tool. The five criterion IDs shown *are*
fixed for tool-use correctness — the other dimensions build their criteria per trajectory.)

## Transcript format

A trajectory is a plain-text file, one tagged line per turn. The last `AGENT:` line is the
"final message" that grounding and communication quality inspect.

```
USER: <what the user said>
AGENT: <what the agent said>
TOOL: <tool_name(arg=value, ...)>
RESULT: <what the tool returned>
```

Untagged lines continue the previous turn. See `trajectories/` for examples.

## Quick start

```bash
pip install -r requirements.txt          # Python 3.11+
export ANTHROPIC_API_KEY=sk-ant-...       # get one at https://console.anthropic.com/settings/keys
```

The repo ships with a runnable **quick-commerce** example — this works with nothing else
set up:

```bash
python run.py trajectories/example_missing_item.txt --score
```

## Running it on your own transcripts

Point the evaluators at your config (once per shell):

```bash
export AQG_CONFIG=config/your_domain.toml   # defaults to config/quick_commerce.toml
```

Then one command per dimension. `<transcript>` is any file in `trajectories/`:

```bash
python run.py           <transcript>.txt --score   # task completion
python run_tooluse.py   <transcript>.txt --score   # tool-use correctness
python run_grounding.py <transcript>.txt --check   # grounding
python run_reasoning.py <transcript>.txt --score   # reasoning quality
python run_comm.py      <transcript>.txt           # communication quality
python run_policy.py    <transcript>.txt           # policy compliance
```

The trailing verb (`--score` / `--check`) means "also run step 2 and score", not just the
identify step. `run_comm.py` and `run_policy.py` are single-step, so they take no verb.
Any `run_*.py` also accepts a config path as its last argument, overriding `$AQG_CONFIG`.

Each step writes a JSON file to a folder named for the dimension (`scores/`,
`tooluse_scores/`, `grounding_scores/`, `reasoning_scores/`, `comm_scores/`,
`policy_scores/`), plus the step-1 artifact (`checklists/`, `grounding_claims/`, etc.).

## Combining the six: the decision engine

Once a trajectory has all six scores on disk, `decision_engine.py` folds them into one
verdict. It only reads the saved score files — it never calls a model.

```bash
python decision_engine.py <transcript>.txt
```

**1. Weighted average.** Fixed weights: task completion 25, reasoning quality 25, tool-use
15, grounding 15, policy compliance 15, communication quality 5. A dimension that is
legitimately not-applicable (reasoning quality with no calculation, policy compliance with
no rule triggered) is dropped and the remaining weights are renormalized — it is not
scored zero.

**2. Critical overrides.** Regardless of the average, a trajectory is blocked from PASS if
any of these hold: a policy-compliance violation (0.00 on an applicable rule), a
*contradicted* grounding claim (an outright false statement — merely ungrounded does not
count), or task completion 0.00.

**3. Label.**

| Condition | Label |
|---|---|
| a required dimension has no score yet (task / tool-use / grounding / communication — these never legitimately N/A) | `INCOMPLETE` |
| no override, average ≥ 0.85 | `PASS` |
| no override, 0.50 ≤ average < 0.85 | `RETRY` |
| no override, average < 0.50 | `HUMAN_REVIEW` |
| override fired, average ≥ 0.60 | `REPAIR` |
| override fired, average < 0.60 | `HUMAN_REVIEW` |

When an override fired and the average is between 0.55 and 0.70, the `REPAIR` / `HUMAN_REVIEW`
call is close — the output flags it for a human. The `0.60` / `0.50` cut-offs are the
current defaults, meant to be tuned against your own trajectories.

Worked example, again from the bundled quick-commerce demo (`wrong_item_disposition` is a
policy rule from *its* config):

```
$ python decision_engine.py trajectories/wrong_item_delivered.txt

weighted average : 54.50 / 75 = 0.727
CRITICAL OVERRIDES
  TRIGGERED  [policy_violation]  policy compliance 0.00 — violated: wrong_item_disposition
DECISION: REPAIR
  override(s) policy_violation fired but weighted average 0.727 is above 0.60 — largely salvageable
```

## Measuring accuracy

An LLM-based evaluator is only useful if it agrees with a human. To measure that on your
domain: write down the scores *you* think are right, then check how often the evaluator
matches.

`golden_dataset.json` holds those hand-verified verdicts, one per trajectory + dimension.
It is **not** written by the evaluator. Each verdict has a `status`:

| status | meaning | counted by `compare_to_golden.py`? |
|---|---|---|
| `pending` | not reviewed yet (a bare skeleton, or an AI first pass awaiting your check) | no — skipped |
| `verified` | a human decided this score | yes |
| `not_applicable` | this dimension has nothing to judge for this trajectory (`score: null`) | yes — expects the evaluator to also say N/A |

```bash
python init_golden.py           # add a pending skeleton (all 6 dimensions) for every trajectory
python compare_to_golden.py      # compare the score files you've already generated to golden
python compare_to_golden.py --live   # re-run the evaluators fresh first (needs a key), then compare
```

`init_golden.py` only adds what's missing — it never overwrites an entry you've filled in.
To draft the verdicts with an AI first pass instead of writing them from scratch, see
**Bootstrapping with an AI** below.

## Writing a config for a new domain

1. Copy `config/TEMPLATE.toml`.
2. Fill in `name`, `description`, the `states` list (or `[]`), the `[[tools]]` blocks
   (each `effect = "read"` or `"write"`), and any `[[constraints]]`.
3. Add `[[policy_rules]]` blocks only for real, specific rules the business enforces on the
   agent's *decisions*. Policy compliance fires only when a named rule like this is
   violated — never as a generic "the agent said something unverified" check (that is
   grounding's job).
4. `export AQG_CONFIG=config/your_domain.toml`.

No evaluator code changes.

## Bootstrapping with an AI

Writing the config and the golden verdicts by hand is tedious. In practice you hand an LLM
(Claude, or any capable model) a description of your system and your transcripts and have
it draft both — then you review. Two prompts:

### Prompt 1 — generate `config/<domain>.toml`

```
You are writing a domain config for the "Agent Quality Gate" evaluator. Output ONE
TOML file and nothing else.

SCHEMA
- Top-level keys (ALL must appear BEFORE the first [[...]] block):
  name          string  - short name of the system under evaluation
  description   string  - one sentence: what the agent does, what requests it handles
  states        array of strings - lifecycle states of the record the agent acts on,
                in order; use [] if there is no meaningful state machine
  states_label  string  - what to call those in prose, e.g. "order states", "ticket states"
- [[tools]] block per tool the agent can call:
  name          exact tool name as it appears in transcripts
  effect        "read" (only looks something up) or "write" (changes state / money /
                commitments, or hands off to a person)
  description   one line: what it does and what it returns
- [[constraints]] block per hard rule the agent must respect:
  id            short_snake_case
  rule          one plain sentence - a hard limit and what correct handling looks like
                when it applies (e.g. "action X is only valid in state Y; otherwise
                explain why and offer an alternative")
- [[policy_rules]] block - OPTIONAL. Only for a real, NAMED business rule on the agent's
  DECISIONS. Do not invent these; add one only if the system description names a concrete
  rule. Never write one that just means "don't say unverified things" - that is handled
  by another dimension.
  id                short_snake_case
  rule              what the agent may not do, and what to do instead
  applies_when      the specific situation the rule is in force
  violation_example a concrete phrase that would violate it

INPUTS
System description:
<<< describe your agent: its job, who it serves, the domain >>>

Tools it can call (name + what each does):
<<< list them >>>

2-3 sample transcripts (USER / AGENT / TOOL / RESULT lines):
<<< paste >>>

Produce the TOML.
```

Save the output as `config/<domain>.toml`, sanity-check it against `config/TEMPLATE.toml`,
then `export AQG_CONFIG=config/<domain>.toml`.

### Prompt 2 — draft golden verdicts for one trajectory

Run `python init_golden.py` first for the skeleton, then for each trajectory:

```
You are drafting "golden" verdicts for the Agent Quality Gate evaluator - a first pass a
human will review, NOT the final answer. For the ONE trajectory below, score each of the
six dimensions. Judge only what the transcript and config show; be strict and cite the
exact turn or tool result behind each verdict.

DIMENSIONS (score 0.0-1.0 unless marked N/A)
- task_completion: did the user get the outcome they asked for? Build the list of things
  that would make THIS request "done" from the user's first message, then score met/total.
  Never N/A.
- tool_use_correctness: right tools chosen; parameters correct (checked against tool
  results); results read correctly; no needless calls; any available verification call
  made before acting. Score = criteria met / 5.
- grounding: is every factual claim in the agent's FINAL message backed by a tool result
  in THIS transcript? Label each claim grounded / contradicted / ungrounded; score =
  grounded / total claims. Apologies, questions and pleasantries are not claims.
- reasoning_quality: was the agent's own inference / arithmetic / judgment correct? Score
  = correct steps / total. If the agent did no real inference or calculation (a plain
  lookup-and-report), status = "not_applicable", score = null.
- communication_quality: rate the FINAL message on 5 fixed criteria - plain language;
  complete & coherent (not fragments, not bloated/repetitive); professional tone;
  internally consistent; information present (cross-check only, never a penalty).
  met=1, partial=0.5, not_met=0; score = points / 5. A missing fact is task_completion's
  problem, not this one.
- policy_compliance: does the agent's decision violate a NAMED [[policy_rules]] entry in
  the config? Score = compliant / applicable rules. If no policy rule is triggered,
  status = "not_applicable", score = null. Never flag something just because nothing
  confirms it - that is grounding's job.

OUTPUT - exactly this shape. Use status "pending" for every dimension you scored (a human
still has to verify it); use "not_applicable" with score null where the dimension does not
apply.
{
  "id": "<trajectory filename without .txt>",
  "intent": "<one phrase>",
  "dimensions": {
    "task_completion":       {"score": 0.0, "status": "pending", "reason": "..."},
    "tool_use_correctness":  {"score": 0.0, "status": "pending", "reason": "..."},
    "grounding":             {"score": 0.0, "status": "pending", "reason": "..."},
    "reasoning_quality":     {"score": null, "status": "not_applicable", "reason": "..."},
    "communication_quality": {"score": 0.0, "status": "pending", "reason": "..."},
    "policy_compliance":     {"score": null, "status": "not_applicable", "reason": "..."}
  }
}

CONFIG
<<< paste your config/<domain>.toml >>>

TRANSCRIPT
<<< paste one trajectories/<name>.txt >>>
```

Merge each object into `golden_dataset.json`. The AI-drafted rows stay `pending`, so
`compare_to_golden.py` ignores them until you read one, agree with the score (or fix it),
and change its `status` to `verified`. That review step is the point of a golden set —
don't skip it.

## Design notes

- **Dimensions don't double-count.** A single failure usually surfaces most sharply in one
  dimension; the others stay quiet or mark N/A. e.g. an agent that states an outcome no
  tool result supports is a *grounding* miss — reasoning quality and policy compliance
  leave that same line alone unless it also breaks their specific check.
- **N/A is a real outcome**, distinct from 0.0. Reasoning quality and policy compliance
  drop a trajectory from their score entirely when they have nothing to check.
- Parsing, display and `compare_to_golden.py` (without `--live`) run with no API key. Only
  the identify/score steps call a model. The model is set per module as `MODEL = ...`
  (currently an Anthropic model; swap it for whatever the `anthropic` SDK points at).
