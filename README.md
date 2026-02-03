# GSO Scaffolds

Minimal infra and examples for running AI coding agents on the [GSO benchmark](https://github.com/gso-bench/gso), and producing **GSO-compatible predictions** for evaluation. To install, run:

```bash
uv pip install -e .

# This installs two CLI entrypoints:
# `gso-harbor` → `harbor/`
# `gso-openhands` → `openhands_gso/`
```

## GSO Summary

All scaffolds are expected to output a GSO-compatible JSONL with (at minimum):

- **`instance_id`**
- **`model_patch`**: a `git diff` (patch to apply)
- **`model_name_or_path`**: any identifier string

Then, the GSO eval harness can evaluate the predictions (testing in dockerized environments, scoring, etc.) as shown in the [official GSO codebase](https://github.com/gso-bench/gso/tree/main/src/gso/harness). Please refer to the documentation there for more details on how to run the eval harness.

## Scaffolds Overview

This repo currently includes two practical integrations/examples:

- **`harbor/`**: 
  - convert GSO tasks into Harbor tasks
  - run any Harbor-compatible agent (e.g. Codex, Claude Code, OpenHands, etc.)
  - automatically runs the GSO eval harness in-container on the agent's patches and export GSO predictions and evaluation results

- **`openhands_gso/`**: 
  - run any version of the OpenHands engine andexport GSO predictions
  - *manually* run the official [GSO eval harness](https://github.com/gso-bench/gso/tree/main/src/gso/harness) on the agent's generated patches


## Scaffold Details

### Harbor

Use this when you want to run agents through [Harbor](https://github.com/harbor-ai/harbor) (e.g. Codex, Claude Code, OpenHands, etc.) by converting GSO instances into Harbor tasks, then exporting Harbor job results back into a GSO `predictions.jsonl`.

1. Convert GSO → Harbor tasks:

    ```bash
    gso-harbor convert --dataset gso-bench/gso --output ./harbor-tasks/
    ```

2. Run with Harbor (example: oracle for validation):

    ```bash
    harbor run --agent oracle --path ./harbor-tasks/<task-name> -n 1
    ```

3. Export Harbor results → GSO predictions:

    ```bash
    gso-harbor export-results --harbor-results ./jobs/<job-name> --output ./predictions/
    ```

For more on the Harbor task layout, see `harbor/README.md`.


### OpenHands

Use this when you want to run OpenHands directly, but still be able to **pin any OpenHands version** via Git tags/releases. For instance, you may want to fork OpenHands and add custom features. This should also prove as a useful example for other custom agents.

Example: pin OpenHands v1.3.0
```bash
uv run \
  --with "openhands-ai @ git+https://github.com/All-Hands-AI/OpenHands.git@v1.3.0" \
  --project . \
  gso-openhands --help
```

More usage details for OpenHands(including `config.toml`), see `openhands_gso/README.md`.
