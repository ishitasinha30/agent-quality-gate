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
filled in, and is safe to re-run after adding trajectories.

## Writing a config for a new domain

1. Copy `config/TEMPLATE.toml`.
2. Fill in `name`, `description`, the `states` list (or `[]`), the `[[tools]]` blocks
   (each with `effect = "read"` or `"write"`), and any `[[constraints]]`.
3. Add `[[policy_rules]]` blocks only for real, specific rules the business enforces on
   the agent's decisions — policy compliance fires only on a named rule, never as a
   generic "nothing confirms this" check.
4. Point the tools at it: `export AQG_CONFIG=config/your_domain.toml`.

No evaluator code changes.

## Design notes

- **Dimensions don't double-count.** A single failure usually surfaces most sharply in one
  dimension; the others stay quiet or mark N/A. e.g. an agent that promises an unbacked
  refund process is a *grounding* miss — reasoning quality and policy compliance both
  leave it alone.
- **N/A is a real outcome**, distinct from 0.0. Reasoning quality and policy compliance
  exclude a trajectory from their score entirely when they have nothing to check.
- Parsing, display and `compare_to_golden.py` run with no API key. Only the
  extract/score steps call the model (`claude-opus-5`, adaptive thinking).
