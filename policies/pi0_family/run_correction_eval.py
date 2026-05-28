# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""Evaluate participant corrections by forking from saved simulation states.

For each correction entry (a JSON lines file), this script:
1. Creates the same task environment as the original eval run.
2. Restores the exact world state at the correction timestep.
3. Swaps in the participant's corrected instruction.
4. Runs the policy forward from that point.
5. Records success/failure and saves videos of the forked rollout.

Usage:
    python run_correction_eval.py \\
        --corrections corrections.jsonl \\
        --source-run output/2026-05-26_15-19-18_pi05 \\
        --policy pi05

The corrections JSONL file should have one JSON object per line::

    {"task": "BananaInBowlTask", "timestep": 120, "instruction": "pick up the banana by the stem", "source_episode": 0}
    {"task": "RubiksCubeAndBananaTask", "timestep": 85, "instruction": "grab the cube first", "source_episode": 0}
"""

import argparse
import json
import os
import sys
import traceback

import cv2  # noqa: F401 -- must import before isaaclab
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate participant corrections via forked rollouts."
)
parser.add_argument(
    "--corrections", type=str, required=True,
    help="Path to JSONL file with correction entries.",
)
parser.add_argument(
    "--source-run", "--source_run", type=str, required=True,
    help="Path to the original eval output directory (contains per-task subdirs with HDF5 files).",
)
parser.add_argument(
    "--policy",
    choices=["pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"],
    default="pi05",
    help="Which Pi0-family variant to use.",
)
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000)
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None)
parser.add_argument(
    "--video-mode", "--video_mode", type=str, default="all",
    choices=["all", "viewport", "sensor", "none"],
)
parser.add_argument("--num-envs", "--num_envs", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.constants import PACKAGE_DIR, get_timestamp, set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.logging.results import init_experiment, summarize_experiment_results  # noqa: E402
from robolab.eval.episode import run_forked_episode  # noqa: E402
from robolab.eval.summarize import summarize_run  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from policies.pi0_family.client import Pi0DroidJointposClient  # noqa: E402

auto_register_droid_envs()


def load_corrections(path: str) -> list[dict]:
    """Load correction entries from a JSON-lines file."""
    corrections = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                corrections.append(json.loads(line))
    return corrections


def main() -> None:
    corrections = load_corrections(args_cli.corrections)
    if not corrections:
        print("No corrections found. Exiting.")
        return

    output_dir = os.path.join(
        PACKAGE_DIR, "output",
        get_timestamp() + f"_correction_eval_{args_cli.policy}",
    )
    os.makedirs(output_dir, exist_ok=True)
    episode_results_file, episode_results = init_experiment(output_dir)

    save_videos = args_cli.video_mode != "none"

    client_kwargs = dict(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        policy_variant=args_cli.policy,
    )
    if args_cli.remote_uri is not None:
        client_kwargs["remote_uri"] = args_cli.remote_uri
    client = Pi0DroidJointposClient(**client_kwargs)

    for idx, correction in enumerate(corrections):
        task = correction["task"]
        timestep = correction["timestep"]
        corrected_instruction = correction["instruction"]
        source_episode = correction.get("source_episode", 0)
        source_hdf5 = os.path.join(
            args_cli.source_run, task, f"run_{source_episode}.hdf5",
        )

        if not os.path.exists(source_hdf5):
            print(f"[{idx}] HDF5 not found: {source_hdf5}; skipping.")
            continue

        task_envs = get_envs(task=[task])
        if not task_envs:
            print(f"[{idx}] Task '{task}' not registered; skipping.")
            continue
        task_env = task_envs[0]

        scene_output_dir = os.path.join(output_dir, f"{task}_correction_{idx}")
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        print(
            f"\n\033[96m[{idx}/{len(corrections)}] Forking '{task}' at step {timestep} "
            f"with instruction: '{corrected_instruction}'\033[0m"
        )

        env, env_cfg = create_env(
            task_env,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            policy=args_cli.policy,
        )

        env_results, msgs, timing = run_forked_episode(
            env=env,
            env_cfg=env_cfg,
            episode=0,
            client=client,
            hdf5_path=source_hdf5,
            fork_timestep=timestep,
            fork_instruction=corrected_instruction,
            source_episode=source_episode,
            save_videos=save_videos,
            video_mode=args_cli.video_mode,
            headless=args_cli.headless,
        )

        run_name = f"{task}_correction_{idx}"
        episode_results = summarize_run(
            env_results=env_results,
            msgs=msgs,
            env=env,
            env_cfg=env_cfg,
            num_envs=args_cli.num_envs,
            run_idx=0,
            run_name=run_name,
            task_env=task_env,
            scene_output_dir=scene_output_dir,
            policy=args_cli.policy,
            episode_results=episode_results,
            episode_results_file=episode_results_file,
            extra_fields={
                "correction_index": idx,
                "fork_timestep": timestep,
                "original_instruction": env_cfg.instruction,
                "corrected_instruction": corrected_instruction,
            },
        )

        env.close()

    summarize_experiment_results(episode_results, show_timing=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
