# Agent Quality Gate

Offline evaluation for AI agent transcripts. Given a recorded run of an agent handling
one user request, it scores that run on six independent quality dimensions and produces a
per-criterion, auditable breakdown — not a single opaque number.

**Domain-generic.** Nothing about any particular product lives in the evaluator code. The
system under evaluation — its tools, the lifecycle of whatever it acts on, its rules — is
described in one `.toml` config. Swap the config and the same six dimensions evaluate a
different product. `config/quick_commerce.toml` is a worked example; `config/TEMPLATE.toml`
is the annotated skeleton for your own.

## The six dimensions

| Dimension | Question it answers | Module(s) |
|---|---|---|
| **Task completion** | Did the user get the outcome they asked for? | `checklist.py` → `score.py` |
| **Tool-use correctness** | Right tools, right parameters, results read correctly, no needless or missing calls? | `tooluse_checklist.py` → `tooluse_score.py` |
| **Grounding** | Does every claim in the final message trace to a tool result in this transcript? | `grounding_claims.py` → `grounding_check.py` |
| **Reasoning quality** | Was the agent's own inference / arithmetic / judgment sound? (N/A if it did none) | `reasoning_steps.py` → `reasoning_score.py` |
| **Communication quality** | Is the final message well written — plain, complete, consistent? | `comm_quality.py` |
| **Policy compliance** | Do the agent's decisions stay within named business rules in the config? (N/A if no rule is triggered) | `policy_check.py` |

Each dimension is a two-step pipeline (extract/plan, then score) with the LLM call isolated
in one function. Every step saves its output to disk so scoring always runs against a
fixed, inspectable artifact rather than a fresh generation.

## Transcript format

A trajectory is a plain-text file, one tagged line per turn:

```
USER: <what the user said>
AGENT: <what the agent said>
TOOL: <tool_name(arg=value, ...)>
RESULT: <what the tool returned>
```

Untagged lines continue the previous turn. See `trajectories/` for examples.

## Running it

```bash
pip install -r requirements.txt          # Python 3.11+; only needed for the LLM steps
export ANTHROPIC_API_KEY=sk-ant-...       # only the extract/score steps call the model

# point at your own domain (optional; defaults to config/quick_commerce.toml)
export AQG_CONFIG=config/your_domain.toml

python run.py           trajectories/example_missing_item.txt --score   # task completion
python run_tooluse.py   trajectories/example_missing_item.txt --score   # tool-use correctness
python run_grounding.py trajectories/refund_status_query.txt  --check   # grounding
python run_reasoning.py trajectories/multi_item_refund_wrong.txt        # reasoning quality (step 1)
python run_comm.py      trajectories/example_missing_item.txt           # communication quality
python run_policy.py    trajectories/wrong_item_delivered.txt           # policy compliance
```

Each `run_*.py` takes an optional trailing config path that overrides `$AQG_CONFIG`.
Artifacts land in `checklists/` + `scores/`, `tooluse_checklists/` + `tooluse_scores/`,
`grounding_claims/` + `grounding_scores/`, `reasoning_steps/` + `reasoning_scores/`,
`comm_scores/`, `policy_scores/`.

## Measuring accuracy

`golden_dataset.json` holds hand-verified verdicts per trajectory + dimension, decided
independently of the evaluator. It is **not** written by the evaluator.

```bash
python init_golden.py                  # scaffold a pending skeleton entry for every trajectory
#   -> then fill in score + status + reason by hand for each dimension row
python compare_to_golden.py            # diff every saved score against golden
python compare_to_golden.py --only vbcr   # only entries marked verified_by_claude_reasoning
python compare_to_golden.py --live        # regenerate evaluator output first (needs a key)
```

`init_golden.py` only adds what's missing — it never touches an entry you've already
filled in, and is safe to re-run after adding trajectories. To fill the entries with an
AI first pass instead of from scratch, see **Bootstrapping with an AI** below.

## Writing a config for a new domain

1. Copy `config/TEMPLATE.toml`.
2. Fill in `name`, `description`, the `states` list (or `[]`), the `[[tools]]` blocks
   (each with `effect = "read"` or `"write"`), and any `[[constraints]]`.
3. Add `[[policy_rules]]` blocks only for real, specific rules the business enforces on
   the agent's decisions — policy compliance fires only on a named rule, never as a
   generic "nothing confirms this" check.
4. Point the tools at it: `export AQG_CONFIG=config/your_domain.toml`.

No evaluator code changes.

## Bootstrapping with an AI

Writing the config and the golden verdicts by hand is tedious. In practice you hand an
LLM (Claude, or any capable model) a description of your system and your transcripts and
have it draft both — then you review. Two prompts:

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
You are drafting "golden" verdicts for the Agent Quality Gate evaluator - a reference a
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
- grounding: does every factual claim in the agent's FINAL message trace to a tool result
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

OUTPUT - exactly this shape, status "draft" for every dimension you actually scored:
{
  "id": "<trajectory filename without .txt>",
  "intent": "<one phrase>",
  "dimensions": {
    "task_completion":       {"score": 0.0, "status": "draft", "reason": "..."},
    "tool_use_correctness":  {"score": 0.0, "status": "draft", "reason": "..."},
    "grounding":             {"score": 0.0, "status": "draft", "reason": "..."},
    "reasoning_quality":     {"score": null, "status": "not_applicable", "reason": "..."},
    "communication_quality": {"score": 0.0, "status": "draft", "reason": "..."},
    "policy_compliance":     {"score": null, "status": "not_applicable", "reason": "..."}
  }
}

CONFIG
<<< paste your config/<domain>.toml >>>

TRANSCRIPT
<<< paste one trajectories/<name>.txt >>>
```

Merge each object into `golden_dataset.json`. **`compare_to_golden.py` ignores any row
whose `status` is not `verified` (or `not_applicable`)** — so drafts sit inert until you
read one, agree or fix the score, and change `status` to `verified`. That review step is
the point of a golden set; don't skip it.

## Design notes

- **Dimensions don't double-count.** A single failure usually surfaces most sharply in one
  dimension; the others stay quiet or mark N/A. e.g. an agent that promises an unbacked
  refund process is a *grounding* miss — reasoning quality and policy compliance both
  leave it alone.
- **N/A is a real outcome**, distinct from 0.0. Reasoning quality and policy compliance
  exclude a trajectory from their score entirely when they have nothing to check.
- Parsing, display and `compare_to_golden.py` run with no API key. Only the
  extract/score steps call the model (`claude-opus-5`, adaptive thinking).
