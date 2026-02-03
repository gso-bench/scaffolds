#!/usr/bin/env python3
"""
CLI for GSO-Harbor integration.

Commands:
- convert: Convert GSO dataset to Harbor task format
- export-results: Convert Harbor results to GSO prediction format
"""

import argparse
from pathlib import Path


def cmd_convert(args):
    """Convert GSO dataset to Harbor tasks."""
    from harbor.convert import HarborTaskConfig, convert_dataset

    config = HarborTaskConfig(
        timeout_sec=args.timeout,
        cpus=args.cpus,
        memory_mb=args.memory_mb,
    )

    instance_ids = args.instance_ids if args.instance_ids else None

    task_dirs = convert_dataset(
        dataset_name=args.dataset,
        output_dir=Path(args.output),
        split=args.split,
        instance_ids=instance_ids,
        config=config,
    )

    print(f"\nConverted {len(task_dirs)} tasks to {args.output}")


def cmd_export_results(args):
    """Export Harbor results to GSO prediction format."""
    from harbor.results import export_predictions

    export_predictions(
        harbor_results_dir=Path(args.harbor_results),
        output_dir=Path(args.output),
    )


def cmd_validate(args):
    """Validate a converted Harbor task."""
    import subprocess

    task_path = Path(args.task)
    if not task_path.exists():
        print(f"Error: Task not found: {task_path}")
        return 1

    result = subprocess.run(
        ["harbor", "tasks", "check", str(task_path), "-m", args.model],
        capture_output=False,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="GSO-Harbor integration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert GSO dataset to Harbor task format"
    )
    convert_parser.add_argument(
        "--dataset",
        "-d",
        default="gso-bench/gso",
        help="GSO dataset name (default: gso-bench/gso)",
    )
    convert_parser.add_argument(
        "--split", "-s", default="test", help="Dataset split (default: test)"
    )
    convert_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for Harbor tasks"
    )
    convert_parser.add_argument(
        "--instance-ids",
        "-i",
        nargs="+",
        help="Specific instance IDs to convert (optional)",
    )
    convert_parser.add_argument(
        "--timeout", "-t", type=int, default=3600, help="Task timeout in seconds"
    )
    convert_parser.add_argument(
        "--cpus", type=int, default=4, help="Number of CPUs per task"
    )
    convert_parser.add_argument(
        "--memory-mb", type=int, default=16384, help="Memory in MB per task"
    )
    convert_parser.set_defaults(func=cmd_convert)

    # export-results command
    export_parser = subparsers.add_parser(
        "export-results", help="Convert Harbor results to GSO prediction format"
    )
    export_parser.add_argument(
        "--harbor-results",
        "-r",
        required=True,
        help="Harbor results directory (jobs/<job-name>)",
    )
    export_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for GSO predictions"
    )
    export_parser.set_defaults(func=cmd_export_results)

    # validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a converted Harbor task"
    )
    validate_parser.add_argument("task", help="Path to Harbor task directory")
    validate_parser.add_argument(
        "--model",
        "-m",
        default="claude-sonnet-4-20250514",
        help="Model to use for validation",
    )
    validate_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
