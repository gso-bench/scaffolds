# GSO Scaffolds

Example “scaffolds” for running AI coding agents on the [GSO benchmark](https://github.com/gso-bench/gso).

A scaffold here is a thin integration layer that:
- **runs an agent** (or an orchestrator) on GSO instances
- **outputs patches** in a standard predictions format
- **relies on the official GSO harness** for evaluation (testing +metrics + scoring)

## Quickstart

Install:

```bash
uv pip install gso-scaffolds
```

Run a scaffold to produce a `predictions.jsonl`, then evaluate it with the official GSO harness.

## Official evaluation (GSO harness)

For details (outputs, metrics, Docker images, timeouts, etc.), see the official harness docs in the GSO repo:
[**`gso-bench/gso` → `src/gso/harness/README.md`**](https://github.com/gso-bench/gso/tree/main/src/gso/harness).

## Common predictions format

All scaffolds should write JSONL with:
- **`instance_id`**
- **`model_patch`** (a `git diff`)
- **`model_name_or_path`** (any identifier string)

## Included scaffolds

### `harbor/` (run Harbor-compatible agents)

Use this when you want to run agents through [Harbor](https://github.com/harbor-ai/harbor) (e.g. Codex, Claude Code, OpenHands, etc.), while keeping **GSO’s official evaluator** inside the task.

- Evaluation: bundled into the task run (the task’s `test.sh` runs the official GSO evaluator in-container).

Convert GSO → Harbor tasks:

```bash
uv run gso-harbor convert --dataset gso-bench/gso --output ./harbor-tasks/
```

Run an agent (example: oracle for validation):

```bash
harbor run --agent oracle --path ./harbor-tasks/<task-name> -n 1
```

Export Harbor results → GSO predictions:

```bash
uv run gso-harbor export-results --harbor-results ./jobs/<job-name> --output ./predictions/
```

### `openhands_gso/` (run the OpenHands engine)

Use this when you want to run OpenHands directly (legacy-style), but **pin any OpenHands version** via a Git tag/release (no long-lived fork).

This scaffold generates patches and writes a `predictions.jsonl`. Evaluate/scoring is a separate step using the official GSO harness (link above).

Pin an OpenHands version and see options:

```bash
uv run \
  --project . \
  --with "openhands-ai @ git+https://github.com/All-Hands-AI/OpenHands.git@v1.3.0" \
  python -m openhands_gso.run_infer --help
```

For usage details (including the example `config.toml`), see `openhands_gso/README.md`.
