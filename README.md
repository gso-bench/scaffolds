# GSO Scaffolds

Example ways to run AI coding agents on the [GSO benchmark](https://github.com/gso-bench/gso).

## Scaffolds

| Scaffold | What it does |
|----------|--------------|
| **harbor/** | Converts GSO tasks to [Harbor](https://github.com/harbor-ai/harbor) format. Run agents like Claude Code CLI, Cursor CLI, etc. |
| **openhands_gso/** | Runs the OpenHands engine on GSO to produce patches. Pin any OpenHands GitHub release/tag with `uv run --with "openhands-ai @ git+…@vX.Y.Z"`. |

## Installation

```bash
uv pip install gso-scaffolds
```

## OpenHands version pinning (GitHub releases)

Use `uv` to run with a specific OpenHands tag/release:

```bash
uv run --project . \
  --with "openhands-ai @ git+https://github.com/All-Hands-AI/OpenHands.git@1.3.0" \
  python -m openhands_gso.run_infer --help
```

## Evaluation

All scaffolds produce patches. Evaluate them with gsobench:

```bash
uv pip install gsobench
gso evaluate --predictions ./predictions.jsonl --dataset gso-bench/gso
```
