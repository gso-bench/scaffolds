# OpenHands GSO Runner

Runs the OpenHands engine on GSO to produce patches. Pin any OpenHands version via `uv run --with`.

## Usage

```bash
# copy and edit config
cp openhands_gso/config.example.toml ./config.toml

# run one instance
uv run \
  --with "openhands-ai @ git+https://github.com/OpenHands/OpenHands.git@1.3.0" \
  --project . \
  python -m openhands_gso.run_infer \
    --llm-config llm.test \
    --config-toml ./config.toml \
    --dataset gso-bench/gso \
    --split test \
    --eval-n-limit 1
```

Options: `--num-workers N` for parallelism, `--output-dir` for resume (skips already-done instances).

## Retry for rate-limited APIs

```bash
until uv run --with "..." --project . python -m openhands_gso.run_infer \
  --llm-config llm.opus-4-6 --config-toml ./config.toml \
  --num-workers 5 --max-iterations 100 \
  --output-dir ./openhands-runs/my-run; do
  sleep 20
done
docker container prune -f
```
