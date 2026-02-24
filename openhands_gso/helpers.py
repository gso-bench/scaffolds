"""Helper functions for GSO benchmark with OpenHands."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# GSO instance helpers
# ---------------------------------------------------------------------------

def get_gso_instance_docker_image(instance_id: str) -> str:
    prefix = (os.environ.get("GSO_IMAGE_PREFIX") or "docker.io/slimshetty/gso").rstrip("/")
    return f"{prefix}:gso.eval.x86_64.{instance_id}".lower()


def get_gso_workspace_dir_name(instance: dict) -> str:
    return instance["repo"].replace("/", "__")


# ---------------------------------------------------------------------------
# Binary patch utilities
# ---------------------------------------------------------------------------

def remove_binary_diffs(patch_text: str) -> str:
    """Remove binary file diffs from a git patch."""
    lines = patch_text.splitlines()
    cleaned_lines: list[str] = []
    block: list[str] = []
    is_binary_block = False

    for line in lines:
        if line.startswith("diff --git "):
            if block and not is_binary_block:
                cleaned_lines.extend(block)
            block = [line]
            is_binary_block = False
        elif "Binary files" in line:
            is_binary_block = True
            block.append(line)
        else:
            block.append(line)

    if block and not is_binary_block:
        cleaned_lines.extend(block)
    return "\n".join(cleaned_lines)


def remove_binary_files_from_git_cmd() -> str:
    """Bash command to remove staged binary files without external tools."""
    return (
        'for file in $(git diff --cached --numstat | awk \'$1=="-" || $2=="-" {print $3}\'); do\n'
        '    git rm -f --cached "$file" 2>/dev/null || true\n'
        '    rm -f "$file" 2>/dev/null || true\n'
        '    echo "Removed binary: $file"\n'
        "done"
    )


def process_git_patch(patch: str) -> str:
    """Clean a git patch: strip control characters, normalize line endings."""
    if not isinstance(patch, str):
        return ""
    if not patch.strip():
        return ""
    patch = patch.replace("\r\n", "\n")
    # Strip any garbage/control characters before the first real diff line
    lines = patch.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("diff --git"):
            patch = "\n".join(lines[i:])
            break
    patch = patch.rstrip() + "\n"  # ensure trailing newline
    return patch


# ---------------------------------------------------------------------------
# Fatal error detection
# ---------------------------------------------------------------------------

_FATAL_ERROR_NAMES = [
    "AgentRuntimeError",
    "AgentRuntimeBuildError",
    "AgentRuntimeTimeoutError",
    "AgentRuntimeUnavailableError",
    "AgentRuntimeNotReadyError",
    "AgentRuntimeDisconnectedError",
    "AgentRuntimeNotFoundError",
    "ConnectionError",
]


def is_fatal_evaluation_error(error: str | None) -> bool:
    if not error:
        return False
    return any(name in error for name in _FATAL_ERROR_NAMES)


# ---------------------------------------------------------------------------
# Fake user response
# ---------------------------------------------------------------------------

def codeact_user_response(state) -> str:
    from openhands.events.action import MessageAction

    msg = (
        "Please continue working on the task on whatever approach you think is suitable.\n"
        "When you think you have solved the question, please use the finish tool and "
        "include your final answer in the message parameter of the finish tool.\n"
        "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n"
    )

    if state.history:
        user_msgs = [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == "user"
        ]
        if len(user_msgs) >= 2:
            return (
                msg
                + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
            )
    return msg


# ---------------------------------------------------------------------------
# Trajectory path
# ---------------------------------------------------------------------------

def trajectory_path_for_instance(traj_dir: str, instance_id: str) -> str:
    """Return the full file path for an instance's trajectory."""
    return str(Path(traj_dir) / f"{instance_id}.json")


# ---------------------------------------------------------------------------
# Per-instance logging (multiprocessing)
# ---------------------------------------------------------------------------

def reset_logger_for_multiprocessing(logger, instance_id: str, log_dir: str) -> None:
    """Reset the logger for multiprocessing.

    Save logs to a separate file for each process, instead of trying to write
    to the same file/console from multiple processes.
    """
    import logging

    log_file = os.path.join(log_dir, f"instance_{instance_id}.log")

    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler — one info line, then WARNING+ only
    try:
        from openhands.core.logger import get_console_handler
        console_handler = get_console_handler(log_level=logging.INFO)
    except ImportError:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            f"Instance {instance_id} - %(asctime)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(console_handler)
    logger.info(
        f"Starting evaluation for instance {instance_id}.\n"
        f'Hint: run "tail -f {log_file}" to see live logs in a separate shell'
    )
    # Only log WARNING or higher to console after the initial message
    console_handler.setLevel(logging.WARNING)

    # File handler — INFO and above
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# OpenHands compatibility patch
# ---------------------------------------------------------------------------

def patch_openhands_skills_dir() -> None:
    """Create the missing skills/ directory when OpenHands is installed via pip/uv."""
    import openhands

    openhands_source = Path(openhands.__file__).parent
    project_root = openhands_source.parent
    skills_dest = project_root / "skills"
    if skills_dest.exists():
        return

    uv_cache_root = None
    for parent in project_root.parents:
        if parent.name == "uv" or (parent / "git-v0").is_dir():
            uv_cache_root = parent if parent.name == "uv" else parent / "uv"
            break
        if parent.name in ("archive-v0", "git-v0"):
            uv_cache_root = parent.parent
            break

    skills_src = None
    if uv_cache_root is not None:
        git_checkouts = uv_cache_root / "git-v0" / "checkouts"
        if git_checkouts.is_dir():
            for checkout in git_checkouts.rglob("skills"):
                if checkout.is_dir() and (checkout / "github.md").exists():
                    skills_src = checkout
                    break

    if skills_src is not None:
        shutil.copytree(skills_src, skills_dest)
    else:
        skills_dest.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# OpenHands lazy import
# ---------------------------------------------------------------------------

def require_openhands() -> dict:
    """Import OpenHands modules lazily. Raises SystemExit if not installed."""
    try:
        from openhands.core.config import AgentConfig, OpenHandsConfig, SandboxConfig, get_llm_config_arg
        from openhands.core.config.condenser_config import NoOpCondenserConfig
        from openhands.core.main import create_runtime, run_controller
        from openhands.events.action import CmdRunAction, FileReadAction, MessageAction
        from openhands.events.observation import (
            CmdOutputObservation,
            ErrorObservation,
            FileReadObservation,
        )
        from openhands.events.serialization.event import event_to_dict
        from openhands.utils.async_utils import call_async_from_sync
        from openhands.core.logger import openhands_logger as logger
        from openhands.memory.condenser import get_condensation_metadata

        patch_openhands_skills_dir()

        return {
            "AgentConfig": AgentConfig,
            "OpenHandsConfig": OpenHandsConfig,
            "SandboxConfig": SandboxConfig,
            "NoOpCondenserConfig": NoOpCondenserConfig,
            "get_llm_config_arg": get_llm_config_arg,
            "create_runtime": create_runtime,
            "run_controller": run_controller,
            "CmdRunAction": CmdRunAction,
            "FileReadAction": FileReadAction,
            "MessageAction": MessageAction,
            "CmdOutputObservation": CmdOutputObservation,
            "ErrorObservation": ErrorObservation,
            "FileReadObservation": FileReadObservation,
            "event_to_dict": event_to_dict,
            "call_async_from_sync": call_async_from_sync,
            "get_condensation_metadata": get_condensation_metadata,
            "logger": logger,
        }
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "OpenHands is not installed in this environment.\n\n"
            "Use uv to pin any release/tag, for example:\n"
            '  uv run --with "openhands-ai @ git+https://github.com/OpenHands/OpenHands.git@1.3.0" gso-openhands --help\n\n'
            f"Import error: {e}"
        )


# ---------------------------------------------------------------------------
# Output directory / resume logic
# ---------------------------------------------------------------------------

def resolve_output_dir(path: str | None, eval_note: str | None = None) -> tuple[Path, bool]:
    """Return (output_dir, resume). *resume* is True when appending to an existing output.jsonl."""
    if path is not None:
        p = Path(path)
        if p.exists():
            output_file = p / "output.jsonl"
            if output_file.exists():
                return p, True
            return p, False
        p.mkdir(parents=True, exist_ok=True)
        return p, False
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    if eval_note:
        stamp = f"{stamp}-{eval_note}"
    p = Path("openhands-runs") / stamp
    p.mkdir(parents=True, exist_ok=True)
    return p, False


def load_done_ids(output_file: Path) -> set[str]:
    """Read already-completed instance_ids from an output.jsonl file."""
    done = set()
    with output_file.open() as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["instance_id"])
    return done


# ---------------------------------------------------------------------------
# Output / retry helpers
# ---------------------------------------------------------------------------

def rebuild_predictions_from_output(output_file: Path, predictions_file: Path) -> None:
    """Ensure output.gso.jsonl mirrors output.jsonl records."""
    if not output_file.exists():
        return
    tmp_file = predictions_file.with_suffix(".jsonl.tmp")
    with output_file.open() as fin, tmp_file.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            pred = {
                "instance_id": rec["instance_id"],
                "model_patch": rec.get(
                    "model_patch", rec.get("test_result", {}).get("git_patch", "")
                ),
                "model_name_or_path": rec.get("model_name_or_path", "openhands"),
            }
            fout.write(json.dumps(pred) + "\n")
    os.replace(tmp_file, predictions_file)



