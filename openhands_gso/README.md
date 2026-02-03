# OpenHands GSO Runner

This runs the OpenHands *engine* on GSO to produce patches, while letting you pin
**any OpenHands version** via GitHub tags/releases (no fork required).

## Pick an OpenHands version (Git tag / release)

```bash
uv run \
  --with "openhands-ai @ git+https://github.com/All-Hands-AI/OpenHands.git@v1.3.0" \
  --project . \
  python -m openhands_gso.run_infer --help
```

## Run one task (no overwrites)

Copy the example config and edit it:

```bash
cp openhands_gso/config.example.toml ./config.toml
```

It’s an OpenHands-style `config.toml` with an `[llm.*]` group:

```toml
[llm.test]
model = "gpt-4o-mini"
temperature = 0.0
```

```bash
uv run \
  --with "openhands-ai @ git+https://github.com/All-Hands-AI/OpenHands.git@v1.3.0" \
  --project . \
  python -m openhands_gso.run_infer \
    --llm-config llm.test \
    --config-toml ./config.toml \
    --dataset gso-bench/gso \
    --split test \
    --eval-n-limit 1
```

## Outputs

- Default: `./openhands-runs/<timestamp>/output.jsonl`
- If you pass `--output-dir`, it refuses to overwrite existing directories.

## Evaluate

```bash
uv pip install gsobench
gso evaluate --predictions ./openhands-runs/<timestamp>/output.jsonl --dataset gso-bench/gso
```
