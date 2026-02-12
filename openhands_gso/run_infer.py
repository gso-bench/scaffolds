#!/usr/bin/env python3
"""
OpenHands runner for GSO.

Produces patches using OpenHands, supports parallel workers and resume.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import signal
import time
from contextlib import contextmanager

from datasets import load_dataset

from openhands_gso.helpers import (
    codeact_user_response,
    get_gso_instance_docker_image,
    get_gso_workspace_dir_name,
    is_fatal_evaluation_error,
    load_done_ids,
    process_git_patch,
    rebuild_predictions_from_output,
    remove_binary_diffs,
    remove_binary_files_from_git_cmd,
    require_openhands,
    reset_logger_for_multiprocessing,
    resolve_output_dir,
    trajectory_path_for_instance,
)

TIMEOUT_SECONDS = 3 * 60 * 60  # 3 hours per instance
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class _EvalTimeout(Exception):
    pass


@contextmanager
def _timeout(seconds: int):
    def handler(signum, frame):
        raise _EvalTimeout(f"Timed out after {seconds}s")

    prev = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _build_instruction(instance: dict) -> str:
    workspace_dir_name = get_gso_workspace_dir_name(instance)

    filter_set = ["git clean -xfd", "which python", "python --version", "uv venv"]
    install_cmds = instance.get("install_commands", [])
    if isinstance(install_cmds, str):
        install_cmds = [install_cmds]
    filtered_cmds = [
        cmd.replace("git clean -xfd &&", "").strip()
        for cmd in install_cmds
        if not any(cmd.startswith(f) for f in filter_set)
    ]
    install_script = "\n".join(filtered_cmds)

    return (
        "<uploaded_files>\n"
        f"/workspace/{workspace_dir_name}\n"
        "</uploaded_files>\n"
        f"I've uploaded a python code repository in the directory {workspace_dir_name}. "
        "Consider the following test script showing an example usage of the repository:\n\n"
        "<test_script>\n"
        f"{instance['prob_script']}\n"
        "</test_script>\n\n"
        "Can you help me implement the necessary changes to the repository so that "
        "the runtime of the <test_script> is optimized?\n"
        "Basic guidelines:\n"
        "   1. Your task is to make changes to non-tests files in the /workspace directory "
        "to improve the performance of the <test_script>.\n"
        "   2. Make changes while ensuring the repository is functionally equivalent to the original.\n"
        "   3. Do not overoptimize for just the specific inputs in <test_script>. "
        "Make general performance improvements for the usage scenario shown.\n"
        "   4. You may need to rebuild the repo for your changes to take effect before testing. "
        "Some rebuilds may take time to run, so be patient with running them.\n"
        "\nFollow these steps to improve performance:\n"
        "1. As a first step, it might be a good idea to explore the repo to familiarize yourself "
        "with its structure.\n"
        "2. Create a script in the /workspace directory (e.g., /workspace/test_opt.py) to reproduce "
        "and time the example and execute it with `python /workspace/<filename.py>` using the BashTool.\n"
        "3. Edit the sourcecode of the repo to improve the performance\n"
        "4. Rebuild and rerun your script and confirm that the performance has improved!\n"
        "Your thinking should be thorough and so it's fine if it's very long.\n"
        f"\nTo rebuild the repo with your changes at any point, you can use the "
        f"following in the {workspace_dir_name} directory:\n"
        f"```\n{install_script}\n```\n"
    )


# ---------------------------------------------------------------------------
# Runtime setup
# ---------------------------------------------------------------------------

def _initialize_runtime(mod, runtime, instance):
    CmdRunAction = mod["CmdRunAction"]
    CmdOutputObservation = mod["CmdOutputObservation"]
    logger = mod["logger"]
    workspace_dir = get_gso_workspace_dir_name(instance)

    def _run(cmd, error_msg=None):
        action = CmdRunAction(command=cmd)
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        if error_msg and (not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0):
            raise RuntimeError(f"{error_msg}: {obs}")
        return obs

    _run(
        f"echo 'export GSO_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && "
        "echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && "
        "echo \"alias git='git --no-pager'\" >> ~/.bashrc",
        "Failed to set environment variables",
    )
    _run("""export USER=$(whoami); echo USER=${USER} """, "Failed to export USER")
    _run(r"sed -i 's#source .venv/bin/activate#\#source .venv/bin/activate#g' ~/.bashrc",
         "Failed to comment out .venv activation")
    _run("if [ -d /workspace ]; then rm -rf /workspace/*; else mkdir /workspace; fi && "
         f"mkdir -p /workspace && cp -r /testbed /workspace/{workspace_dir}",
         f"Failed to set up workspace /workspace/{workspace_dir}")
    _run(f"cd /workspace/{workspace_dir}", f"Failed to cd to /workspace/{workspace_dir}")
    _run("source .venv/bin/activate", "Failed to activate .venv")
    obs = _run("which python", "Failed to run 'which python'")
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"'which python' failed: {obs}")
    if "testbed" not in getattr(obs, "content", ""):
        raise RuntimeError(
            f"Expected to find python interpreter from testbed, but got: {obs}"
        )


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def _extract_patch(mod, runtime, instance):
    CmdRunAction = mod["CmdRunAction"]
    FileReadAction = mod["FileReadAction"]
    CmdOutputObservation = mod["CmdOutputObservation"]
    ErrorObservation = mod["ErrorObservation"]
    FileReadObservation = mod["FileReadObservation"]
    logger = mod["logger"]
    workspace_dir = get_gso_workspace_dir_name(instance)
    base_commit = instance["base_commit"]

    def _run(cmd, timeout=600):
        action = CmdRunAction(command=cmd)
        action.set_hard_timeout(timeout)
        obs = runtime.run_action(action)
        logger.info(obs, extra={"msg_type": "OBSERVATION"})
        return obs

    obs = _run(f"cd /workspace/{workspace_dir}")
    if isinstance(obs, CmdOutputObservation) and obs.exit_code == -1:
        runtime.run_action(CmdRunAction(command="C-c", is_input=True))
        time.sleep(5)
        obs = _run(f"cd /workspace/{workspace_dir}")
    if isinstance(obs, CmdOutputObservation) and obs.exit_code == -1:
        runtime.run_action(CmdRunAction(command="C-z", is_input=True))
        obs = _run(f"cd /workspace/{workspace_dir}")
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to cd to /workspace/{workspace_dir}: {obs}")

    _run('git config --global core.pager ""')
    _run("git add -A")
    _run(remove_binary_files_from_git_cmd())

    git_patch = None
    for attempt in range(5):
        t = max(300 + 100 * attempt, 600)
        obs = _run(f"git diff --no-color --cached {base_commit} > /tmp/patch.diff", timeout=t)
        if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
            read_action = FileReadAction(path="/tmp/patch.diff")
            read_action.set_hard_timeout(t)
            read_obs = runtime.run_action(read_action)
            logger.info(read_obs, extra={"msg_type": "OBSERVATION"})
            if isinstance(read_obs, FileReadObservation):
                git_patch = read_obs.content
                break
            if isinstance(read_obs, ErrorObservation) and "File could not be decoded as utf-8" in (
                getattr(read_obs, "content", "") or ""
            ):
                cat_obs = _run("cat /tmp/patch.diff", timeout=t)
                if isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0:
                    git_patch = getattr(cat_obs, "content", "") or ""
                    break
        logger.info(f"git diff failed (attempt {attempt + 1}), retrying...")
        time.sleep(10)

    if git_patch is None:
        raise RuntimeError("Failed to get git diff after 5 attempts")
    return remove_binary_diffs(git_patch)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _process_instance(instance: dict, args_dict: dict) -> dict:
    """Run one GSO instance end-to-end. Returns result dict."""
    mod = require_openhands()
    logger = mod["logger"]
    instance_id = instance["instance_id"]

    # Set up per-instance logging so parallel workers don't clobber each other
    log_dir = os.path.join(args_dict["output_dir"], "infer_logs")
    reset_logger_for_multiprocessing(logger, instance_id, log_dir)

    llm_config = mod["get_llm_config_arg"](args_dict["llm_config"], args_dict["config_toml"])
    if llm_config is None:
        raise RuntimeError(f'LLM config "{args_dict["llm_config"]}" not found')

    llm_config.modify_params = False
    llm_config.log_completions = True
    if llm_config.log_completions:
        llm_config.log_completions_folder = os.path.join(
            args_dict["output_dir"], "llm_completions", instance_id
        )
    if str(getattr(llm_config, "model", "")).startswith("vertex_ai/claude"):
        llm_config.top_p = None

    # --- Sandbox config: mirrors get_default_sandbox_config_for_eval() + GSO overrides ---
    sandbox_config = mod["SandboxConfig"](
        use_host_network=False,
        # default is 300; GSO overrides to 20 min for long rebuilds/tests
        timeout=20 * 60,
        api_key=os.environ.get("ALLHANDS_API_KEY", None),
        runtime_startup_env_vars={"NO_CHANGE_TIMEOUT_SECONDS": "30"},
        remote_runtime_api_url=os.environ.get("SANDBOX_REMOTE_RUNTIME_API_URL"),
        keep_runtime_alive=False,
        remote_runtime_init_timeout=3600,
        remote_runtime_api_timeout=120,
        remote_runtime_enable_retries=True,
        remote_runtime_class="sysbox",
    )
    sandbox_config.base_container_image = get_gso_instance_docker_image(instance_id)
    sandbox_config.enable_auto_lint = True
    sandbox_config.platform = "linux/amd64"

    config = mod["OpenHandsConfig"](
        default_agent=args_dict["agent_cls"],
        run_as_openhands=False,
        max_iterations=args_dict["max_iterations"],
        runtime=os.environ.get("RUNTIME", "docker"),
        sandbox=sandbox_config,
        workspace_base=None,
        workspace_mount_path=None,
    )
    traj_dir = args_dict.get("trajectory_dir")
    if traj_dir:
        config.save_trajectory_path = trajectory_path_for_instance(traj_dir, instance_id)
    config.set_llm_config(llm_config)
    config.set_agent_config(mod["AgentConfig"](
        enable_jupyter=False, enable_browsing=False,
        enable_llm_editor=False, enable_prompt_extensions=False,
        condenser=mod["NoOpCondenserConfig"](),
    ))

    runtime = mod["create_runtime"](config)
    mod["call_async_from_sync"](runtime.connect)

    try:
        _initialize_runtime(mod, runtime, instance)
        instruction = _build_instruction(instance)
        state = asyncio.run(
            mod["run_controller"](
                config=config,
                initial_user_action=mod["MessageAction"](content=instruction),
                runtime=runtime,
                fake_user_response_fn=codeact_user_response,
            )
        )
        if state and is_fatal_evaluation_error(getattr(state, "last_error", None)):
            raise RuntimeError(f"Fatal error detected: {state.last_error}")
        git_patch = _extract_patch(mod, runtime, instance)
    finally:
        runtime.close()

    # Build result
    history = [mod["event_to_dict"](event) for event in state.history] if state else None
    metrics = state.metrics.get() if state and getattr(state, "metrics", None) else {}
    # Add condenser metadata to metrics
    try:
        metrics["condenser"] = mod["get_condensation_metadata"](state)
    except Exception:
        pass
    model_name = getattr(llm_config, "model", "openhands")

    # Clean the patch (strip control chars, normalize line endings)
    cleaned_patch = process_git_patch(git_patch)

    return {
        "instance_id": instance_id,
        "test_result": {"git_patch": git_patch},
        "instruction": instruction,
        "metadata": {
            "agent_class": args_dict["agent_cls"],
            "model_name": model_name,
            "max_iterations": args_dict["max_iterations"],
        },
        "history": history,
        "metrics": metrics,
        "error": state.last_error if state and getattr(state, "last_error", None) else None,
        "instance": instance,
        "model_patch": cleaned_patch,
        "model_name_or_path": model_name,
    }


# ---------------------------------------------------------------------------
# Wrapper: timeout + retries
# ---------------------------------------------------------------------------

def _make_error_record(instance: dict, error: str) -> dict:
    return {
        "instance_id": instance["instance_id"],
        "test_result": {"git_patch": ""},
        "instruction": "",
        "metadata": {},
        "history": None,
        "metrics": {},
        "error": error,
        "instance": instance,
        "model_patch": "",
        "model_name_or_path": "openhands",
    }


def _worker_wrapper(args):
    """Wrapper for mp.Pool — handles timeout + retries, always returns a dict."""
    instance, args_dict = args
    instance_id = instance["instance_id"]

    for attempt in range(MAX_RETRIES + 1):
        try:
            with _timeout(TIMEOUT_SECONDS):
                return _process_instance(instance, args_dict)
        except _EvalTimeout:
            return _make_error_record(instance, f"Timeout after {TIMEOUT_SECONDS}s")
        except Exception as e:  # noqa: BLE001
            if attempt == MAX_RETRIES:
                return _make_error_record(instance, f"Failed after {MAX_RETRIES} retries: {e}")
            print(f"[{instance_id}] attempt {attempt + 1} failed: {e}, retrying...")
            time.sleep(5)
            continue

    return _make_error_record(instance, "Unexpected: exhausted retries")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    mod = require_openhands()

    parser = argparse.ArgumentParser(description="Run OpenHands on GSO to produce patches.")
    parser.add_argument("--dataset", default="gso-bench/gso")
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-ids", nargs="*", default=None)
    parser.add_argument("--eval-n-limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--agent-cls", default="CodeActAgent")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--eval-note", default=None)
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--config-toml", default="config.toml")
    args = parser.parse_args()

    output_dir, resume = resolve_output_dir(args.output_dir, args.eval_note)
    output_file = output_dir / "output.jsonl"
    predictions_file = output_dir / "predictions.jsonl"

    ds = load_dataset(args.dataset, split=args.split)
    if args.instance_ids:
        wanted = set(args.instance_ids)
        ds = ds.filter(lambda x: x["instance_id"] in wanted)
    if args.eval_n_limit:
        ds = ds.shuffle(seed=42).select(range(min(args.eval_n_limit, len(ds))))

    if resume:
        rebuild_predictions_from_output(output_file, predictions_file)
        done_ids = load_done_ids(output_file)
        ds = ds.filter(lambda x: x["instance_id"] not in done_ids)
        print(f"Resume: skipping {len(done_ids)} already-done, {len(ds)} remaining.")

    if len(ds) == 0:
        print("Nothing to run.")
        return 0

    llm_config = mod["get_llm_config_arg"](args.llm_config, args.config_toml)
    if llm_config is None:
        raise SystemExit(f'LLM config "{args.llm_config}" not found in {args.config_toml}.')

    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    args_dict = {
        "llm_config": args.llm_config,
        "config_toml": args.config_toml,
        "agent_cls": args.agent_cls,
        "max_iterations": args.max_iterations,
        "trajectory_dir": str(traj_dir),
        "output_dir": str(output_dir),
    }
    instances = [dict(item) for item in ds]

    output_fp = open(output_file, "a")
    pred_fp = open(predictions_file, "a")

    try:
        if args.num_workers > 1:
            with mp.Pool(args.num_workers) as pool:
                work = ((inst, args_dict) for inst in instances)
                for result in pool.imap_unordered(_worker_wrapper, work):
                    output_fp.write(json.dumps(result) + "\n")
                    output_fp.flush()
                    pred_fp.write(json.dumps({
                        "instance_id": result["instance_id"],
                        "model_patch": result["model_patch"],
                        "model_name_or_path": result["model_name_or_path"],
                    }) + "\n")
                    pred_fp.flush()
                    print(f"Done: {result['instance_id']}")
        else:
            for inst in instances:
                result = _worker_wrapper((inst, args_dict))
                output_fp.write(json.dumps(result) + "\n")
                output_fp.flush()
                pred_fp.write(json.dumps({
                    "instance_id": result["instance_id"],
                    "model_patch": result["model_patch"],
                    "model_name_or_path": result["model_name_or_path"],
                }) + "\n")
                pred_fp.flush()
                print(f"Done: {result['instance_id']}")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt. Cleaning up...")
    finally:
        output_fp.close()
        pred_fp.close()

    print(f"Wrote: {output_file}")
    print(f"Wrote: {predictions_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
