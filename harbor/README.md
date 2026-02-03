# Harbor Integration

Converts GSO tasks to Harbor format so you can run any Harbor-compatible agent.

## Usage

```bash
# Convert GSO to Harbor tasks
uv run gso-harbor convert --dataset gso-bench/gso --output ./harbor-tasks/

# Run with Harbor
harbor run --agent oracle --path ./harbor-tasks/<task-name> -n 1
harbor run --agent claude-code --path ./harbor-tasks/ -n 10

# Export results to GSO format
uv run gso-harbor export-results --harbor-results ./jobs/<job-name> --output ./predictions/
```

## Task Structure

```
task-name/
├── task.toml           # Harbor config
├── instruction.md      # Agent prompt
├── environment/
│   └── Dockerfile      # GSO image + gsobench
├── tests/
│   ├── test.sh         # Evaluation entry point
│   ├── eval.sh         # GSO eval script
│   └── gso_test_*.py   # Test files
└── solution/
    └── solve.sh        # Oracle (ground truth)
```
