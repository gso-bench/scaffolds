#!/usr/bin/env python3
"""
OpenHands runner for GSO.

Goal: produce patches using OpenHands *engine* pinned to any GitHub release/tag.

You select the OpenHands version with uv, e.g.:
  uv run --with "openhands-ai @ git+https://github.com/All-Hands-AI/OpenHands.git@v1.3.0" gso-openhands ...
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

from openhands_gso.helpers import get_gso_instance_docker_image, get_gso_workspace_dir_name


def _require_openhands():
    try:
        from openhands.core.config import AgentConfig, OpenHandsConfig, SandboxConfig, get_llm_config_arg
        from openhands.core.main import create_runtime, run_controller
        from openhands.events.action import CmdRunAction, MessageAction
        from openhands.events.observation import CmdOutputObservation
        from openhands.utils.async_utils import call_async_from_sync
        from openhands.core.logger import openhands_logger as logger

        return {
            "AgentConfig": AgentConfig,
            "OpenHandsConfig": OpenHandsConfig,
            "SandboxConfig": SandboxConfig,
            "get_llm_config_arg": get_llm_config_arg,
            "create_runtime": create_runtime,
            "run_controller": run_controller,
            "CmdRunAction": CmdRunAction,
            "MessageAction": MessageAction,
            "CmdOutputObservation": CmdOutputObservation,
            "call_async_from_sync": call_async_from_sync,
            "logger": logger,
        }
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "OpenHands is not installed in this environment.\n\n"
            "Use uv to pin any release/tag, for example:\n"
            '  uv run --with "openhands-ai @ git+https://github.com/All-Hands-AI/OpenHands.git@v1.3.0" gso-openhands --help\n\n'
            f"Import error: {e}"
        )


def build_instruction(instance: dict) -> str:
    workspace_dir_name = get_gso_workspace_dir_name(instance)
    filter_set = ["git clean -xfd", "which python", "python --version", "uv venv"]
    install_cmds = [
        cmd.replace("git clean -xfd &&", "").strip()
        for cmd in instance["install_commands"]
        if not any(cmd.startswith(f) for f in filter_set)
    ]
    install_script = "\n".join(install_cmds)

    return f"""<uploaded_files>
/workspace/{workspace_dir_name}
</uploaded_files>
I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following test script showing an example usage of the repository:

<test_script>
{instance["prob_script"]}
</test_script>

Can you help me implement the necessary changes to the repository so that the runtime of the <test_script> is optimized?

<important>
Please make sure you produce a non-empty patch. If you're unsure about the best optimization, make at least one small, safe, performance-motivated change (e.g., reduce overhead, avoid redundant work) before you stop.
</important>

To rebuild the repo with your changes:
```
{install_script}
```
"""


def _mkdir_and_link_workspace(mod, runtime, instance: dict) -> None:
    CmdRunAction = mod["CmdRunAction"]
    CmdOutputObservation = mod["CmdOutputObservation"]
    logger = mod["logger"]

    workspace_dir = get_gso_workspace_dir_name(instance)
    cmd = f"mkdir -p /workspace && ln -sf /testbed /workspace/{workspace_dir}"
    action = CmdRunAction(command=cmd)
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})
    if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
        raise RuntimeError(f"Failed to create workspace link: {obs}")


def _extract_patch(mod, runtime, instance: dict) -> str:
    CmdRunAction = mod["CmdRunAction"]

    workspace_dir = get_gso_workspace_dir_name(instance)
    base_commit = instance["base_commit"]

    # stage changes
    action = CmdRunAction(command=f'cd "/workspace/{workspace_dir}" && git add -A')
    action.set_hard_timeout(600)
    runtime.run_action(action)

    # diff from base commit
    action = CmdRunAction(
        command=f'cd "/workspace/{workspace_dir}" && git diff --no-color --cached {base_commit}'
    )
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    return getattr(obs, "content", "") or ""


def _output_dir(path: str | None) -> Path:
    if path is not None:
        p = Path(path)
        if p.exists():
            raise SystemExit(f"Refusing to overwrite existing output dir: {p}")
        p.mkdir(parents=True, exist_ok=True)
        return p
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    p = Path("openhands-runs") / stamp
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> int:
    mod = _require_openhands()

    parser = argparse.ArgumentParser(description="Run OpenHands on GSO to produce patches.")
    parser.add_argument("--dataset", default="gso-bench/gso")
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-ids", nargs="*", default=None)
    parser.add_argument("--eval-n-limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--agent-cls", default="CodeActAgent")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--llm-config", required=True)
    parser.add_argument(
        "--config-toml",
        default="config.toml",
        help="Path to an OpenHands-style config.toml containing [llm.<name>] groups.",
    )
    args = parser.parse_args()

    output_dir = _output_dir(args.output_dir)
    output_file = output_dir / "output.jsonl"

    ds = load_dataset(args.dataset, split=args.split)
    if args.instance_ids:
        wanted = set(args.instance_ids)
        ds = ds.filter(lambda x: x["instance_id"] in wanted)
    if args.eval_n_limit:
        ds = ds.select(range(min(args.eval_n_limit, len(ds))))

    llm_config = mod["get_llm_config_arg"](args.llm_config, args.config_toml)
    if llm_config is None:
        raise SystemExit(
            f'LLM config "{args.llm_config}" not found in {args.config_toml}.\n\n'
            "Create a config file like:\n"
            "[llm.my-llm]\n"
            "model = \"gpt-4o-mini\"\n"
            "temperature = 0.0\n\n"
            "Then pass: --llm-config llm.my-llm --config-toml /path/to/config.toml\n"
        )

    # write outputs incrementally; refuse overwrite of directory already handled above
    with output_file.open("x") as f:
        for item in ds:
            instance = dict(item)

            # Build sandbox config similar to the original GSO runner.
            sandbox_config = mod["SandboxConfig"](use_host_network=False, timeout=20 * 60)
            sandbox_config.base_container_image = instance.get(
                "remote_instance_image_key", get_gso_instance_docker_image(instance["instance_id"])
            )
            sandbox_config.enable_auto_lint = True
            # Match fork behavior / common OpenHands docs knobs
            sandbox_config.platform = "linux/amd64"
            runtime_img = os.environ.get("SANDBOX_RUNTIME_CONTAINER_IMAGE")
            if runtime_img:
                sandbox_config.runtime_container_image = runtime_img

            config = mod["OpenHandsConfig"](
                default_agent=args.agent_cls,
                run_as_openhands=False,
                max_iterations=args.max_iterations,
                runtime=os.environ.get("RUNTIME", "docker"),
                sandbox=sandbox_config,
                workspace_base=None,
                workspace_mount_path=None,
            )
            config.set_llm_config(llm_config)
            config.set_agent_config(
                mod["AgentConfig"](
                    enable_jupyter=False,
                    enable_browsing=False,
                    enable_llm_editor=False,
                    enable_prompt_extensions=False,
                )
            )

            runtime = mod["create_runtime"](config)
            mod["call_async_from_sync"](runtime.connect)

            error = None
            try:
                _mkdir_and_link_workspace(mod, runtime, instance)
                instruction = build_instruction(instance)
                state = __import__("asyncio").run(
                    mod["run_controller"](
                        config=config,
                        initial_user_action=mod["MessageAction"](content=instruction),
                        runtime=runtime,
                        fake_user_response_fn=lambda _state: "Please continue.",
                    )
                )
                patch = _extract_patch(mod, runtime, instance)
            except Exception as e:  # noqa: BLE001
                error = str(e)
                patch = ""
            finally:
                runtime.close()

            rec = {
                "instance_id": instance["instance_id"],
                "model_patch": patch,
                "model_name_or_path": getattr(llm_config, "model", "openhands"),
            }
            if error:
                rec["error"] = error
            f.write(json.dumps(rec) + "\n")
            f.flush()

    print(f"Wrote: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
