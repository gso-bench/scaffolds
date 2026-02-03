"""
Convert Harbor results to GSO prediction format.

Harbor outputs results in:
    jobs/<job-name>/<task-name>__<id>/
        - verifier/model_patch.diff (captured git diff)
        - verifier/result.json (GSO metrics)

GSO expects predictions as:
    {
        "instance_id": "repo__commit",
        "model_patch": "<unified diff>",
        "model_name_or_path": "agent-name"
    }
"""

import json
from pathlib import Path


def harbor_task_name_to_gso_instance_id(task_name: str) -> str:
    """Convert Harbor task name back to GSO instance_id.

    Harbor task: huggingface-datasets-5994036
    GSO instance: huggingface__datasets-5994036
    """
    parts = task_name.split("-", 1)
    if len(parts) == 2:
        return f"{parts[0]}__{parts[1]}"
    return task_name


def extract_prediction_from_task_dir(task_dir: Path) -> dict | None:
    """Extract GSO prediction from a Harbor task result directory."""
    patch_file = task_dir / "verifier" / "model_patch.diff"
    
    if not patch_file.exists():
        return None

    model_patch = patch_file.read_text()

    # Task dir name is like "tornadoweb-tornado-1b464c4__6WZ9VFi"
    # Extract task name (before __<id>)
    dir_name = task_dir.name
    if "__" in dir_name:
        task_name = dir_name.rsplit("__", 1)[0]
    else:
        task_name = dir_name
    
    instance_id = harbor_task_name_to_gso_instance_id(task_name)

    return {
        "instance_id": instance_id,
        "model_patch": model_patch,
    }


def extract_predictions_from_job(
    job_dir: Path, agent_name: str | None = None
) -> list[dict]:
    """Extract all predictions from a Harbor job."""
    predictions = []

    # Get agent name from job config if not provided
    if agent_name is None:
        config_file = job_dir / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
            agent_name = config.get("agent", {}).get("name", "unknown")
        else:
            agent_name = "unknown"

    # Iterate through task directories (format: <task-name>__<id>)
    for item in job_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name.startswith("."):
            continue
        # Skip non-task directories
        if "__" not in item.name:
            continue

        pred = extract_prediction_from_task_dir(item)
        if pred:
            pred["model_name_or_path"] = agent_name
            predictions.append(pred)

    return predictions


def export_predictions(
    harbor_results_dir: Path,
    output_dir: Path,
    agent_name: str | None = None,
) -> Path:
    """Export Harbor job results to GSO prediction JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = extract_predictions_from_job(harbor_results_dir, agent_name)

    if not predictions:
        print(f"Warning: No predictions found in {harbor_results_dir}")
        return None

    if agent_name:
        output_file = output_dir / f"{agent_name}.jsonl"
    else:
        job_name = harbor_results_dir.name
        output_file = output_dir / f"{job_name}.jsonl"

    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Exported {len(predictions)} predictions to {output_file}")
    return output_file


def merge_predictions(prediction_files: list[Path], output_file: Path) -> Path:
    """Merge multiple prediction files into one."""
    all_predictions = []

    for pf in prediction_files:
        with open(pf) as f:
            for line in f:
                all_predictions.append(json.loads(line))

    with open(output_file, "w") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Merged {len(all_predictions)} predictions to {output_file}")
    return output_file
