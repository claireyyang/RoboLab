# isort: skip_file

"""Evaluate participant corrections by forking from saved simulation states.

For each text correction in the Prolific feedback CSV, this script:
1. Creates the same task environment as the original eval run.
2. Restores the exact world state at the correction timestep.
3. Swaps in the participant's corrected instruction.
4. Runs the policy forward from that point.
5. Records success/failure and saves videos of the forked rollout.

Usage:
    python run_correction_eval.py \\
        --corrections policies/pi0_family/prolific_deployment_1_feedback_clean.csv \\
        --survey-tasks policies/pi0_family/prolific_1_survey_tasks.csv \\
        --policy pi05 \\
        --num-runs 5 \\
        --output-folder-name my_correction_eval

Results are saved to: output/<output_folder_name>/
A per-correction spreadsheet is written to correction_eval_results.csv.
Re-running with the same ``--output-folder-name`` resumes: completed
spreadsheet rows are kept and those corrections/runs are skipped.

The feedback CSV has columns VideoTimestamp, Video, UserID, Feedback,
InterfaceMode, InterfaceCondition. Only ``main_task_*`` rows are used.
Each ``main_task_*`` video is mapped to a specific original rollout via
``prolific_1_survey_tasks.csv`` (run indices 1, 7, 10, 14 — not 0–3).
Video timestamps (seconds) are converted to policy steps at 15 Hz
(``timestep = round(seconds * 15)``).

Visual click annotations and "No correction needed" rows are skipped because
the forked rollout takes a language instruction.
"""

import argparse
import csv
import json
import os
import re
import sys
import traceback
from collections import defaultdict

import cv2  # noqa: F401 -- must import before isaaclab
from isaaclab.app import AppLauncher

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CORRECTIONS = os.path.join(
    _THIS_DIR, "prolific_deployment_1_feedback_clean.csv"
)
_DEFAULT_SURVEY_TASKS = os.path.join(_THIS_DIR, "prolific_1_survey_tasks.csv")

# Policy / viewport video rate: 1 / (render_interval * dt) = 1 / (8 * 1/120).
VIDEO_FPS = 15
_VIEWPORT_RUN_RE = re.compile(r"_(\d+)_viewport\.mp4$")

SPREADSHEET_FIELDS = [
    "participant_id",
    "survey_video",
    "timestamp_s",
    "fork_timestep",
    "correction",
    "success",
    "output_video",
    "source_video",
    "source_hdf5",
    "source_run",
    "eval_run",
    "interface_mode",
    "interface_condition",
    "task",
    "skip_reason",
]

parser = argparse.ArgumentParser(
    description="Evaluate participant corrections via forked rollouts."
)
parser.add_argument(
    "--corrections", type=str, default=_DEFAULT_CORRECTIONS,
    help="Path to Prolific feedback CSV. Default: prolific_deployment_1_feedback_clean.csv.",
)
parser.add_argument(
    "--survey-tasks", "--survey_tasks", type=str, default=_DEFAULT_SURVEY_TASKS,
    help="CSV mapping survey_video names to original rollout videos/HDF5s.",
)
parser.add_argument(
    "--source-run", "--source_run", type=str, default=None,
    help=(
        "Optional override for the original eval output directory "
        "(contains per-task subdirs with HDF5 files). If omitted, HDF5 paths "
        "are taken from --survey-tasks."
    ),
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
parser.add_argument(
    "--num-runs", "--num_runs", type=int, default=1,
    help="Number of sequential runs per correction entry (default: 1).",
)
parser.add_argument(
    "--output-folder-name", "--output_folder_name", type=str, default=None,
    help="Output folder name under output/. Default is <timestamp>_correction_eval_<policy>.",
)

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
from robolab.eval.episode import cleaned_video_stem, run_forked_episode  # noqa: E402
from robolab.eval.summarize import summarize_run  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from policies.pi0_family.client import Pi0DroidJointposClient  # noqa: E402

auto_register_droid_envs()


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PACKAGE_DIR, path)


def load_survey_rollouts(
    path: str, source_run_root: str | None = None
) -> dict[str, dict]:
    """Map survey video names to the original rollouts they were recorded from."""
    rollouts: dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            video = (row.get("survey_video") or "").strip()
            orig = (row.get("original_video_path") or "").strip()
            task = (row.get("task") or "").strip()
            if not video or not orig:
                continue
            match = _VIEWPORT_RUN_RE.search(orig)
            if match is None:
                print(f"Could not parse run index from {orig}; skipping {video}.")
                continue
            source_run = int(match.group(1))
            orig_dir = os.path.dirname(orig)
            if source_run_root:
                hdf5 = os.path.join(source_run_root, task, f"run_{source_run}.hdf5")
            else:
                hdf5 = os.path.join(orig_dir, f"run_{source_run}.hdf5")
            rollouts[video] = {
                "task": task,
                "source_run": source_run,
                "original_video_path": orig,
                "source_hdf5": _resolve_path(hdf5),
            }
    print(f"Loaded {len(rollouts)} survey rollout mapping(s) from {path}")
    return rollouts


def _text_instruction(feedback_raw: str) -> str | None:
    """Return a language correction, or None for visual / no-correction rows."""
    payload = json.loads(feedback_raw)
    feedback = payload.get("feedback")
    if not isinstance(feedback, str):
        return None
    text = feedback.strip()
    if not text or text == "No correction needed":
        return None
    if payload.get("noCorrectionReason") or payload.get("noInterventionReason"):
        return None
    return text


def load_corrections(path: str, survey_rollouts: dict[str, dict]) -> list[dict]:
    """Load text corrections for main-task videos from a Prolific feedback CSV."""
    corrections = []
    skipped_non_main = 0
    skipped_unmapped = 0
    skipped_no_instruction = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            video = (row.get("Video") or "").strip()
            if not video.startswith("main_task_"):
                skipped_non_main += 1
                continue
            rollout = survey_rollouts.get(video)
            if rollout is None:
                skipped_unmapped += 1
                continue
            try:
                instruction = _text_instruction(row["Feedback"])
            except json.JSONDecodeError:
                skipped_no_instruction += 1
                continue
            if instruction is None:
                skipped_no_instruction += 1
                continue
            timestamp_s = float(row["VideoTimestamp"])
            corrections.append({
                "task": rollout["task"],
                "timestep": int(round(timestamp_s * VIDEO_FPS)),
                "instruction": instruction,
                "source_run": rollout["source_run"],
                "source_env": 0,
                "source_hdf5": rollout["source_hdf5"],
                "original_video_path": rollout["original_video_path"],
                "video": video,
                "timestamp_s": timestamp_s,
                "user_id": row.get("UserID", ""),
                "interface_mode": row.get("InterfaceMode", ""),
                "interface_condition": row.get("InterfaceCondition", ""),
            })
    print(
        f"Loaded {len(corrections)} text correction(s) from {path} "
        f"(skipped {skipped_non_main} non-main-task row(s), "
        f"{skipped_unmapped} unmapped main-task row(s), "
        f"{skipped_no_instruction} visual/no-correction row(s))"
    )
    return corrections


def _output_video_filename(
    instruction: str, run_idx: int, num_envs: int, env_id: int
) -> str:
    cleaned = cleaned_video_stem(instruction)
    suffix = f"_{run_idx}_env{env_id}" if num_envs > 1 else f"_{run_idx}"
    return f"{cleaned}{suffix}_viewport.mp4"


def write_spreadsheet(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPREADSHEET_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_spreadsheet(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _norm_timestamp(value) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _correction_identity(correction: dict) -> tuple:
    return (
        str(correction.get("user_id", "") or ""),
        str(correction.get("video", "") or ""),
        _norm_timestamp(correction.get("timestamp_s")),
        str(correction.get("instruction", "") or ""),
    )


def _row_identity(row: dict) -> tuple:
    return (
        str(row.get("participant_id", "") or ""),
        str(row.get("survey_video", "") or ""),
        _norm_timestamp(row.get("timestamp_s")),
        str(row.get("correction", "") or ""),
    )


def _row_is_complete(row: dict) -> bool:
    if (row.get("skip_reason") or "").strip():
        return True
    return str(row.get("success", "") or "").strip() != ""


def completed_from_spreadsheet(
    rows: list[dict],
) -> tuple[set[tuple], dict[tuple, set[int]]]:
    """Return (fully skipped identities, identity -> completed eval_run indices)."""
    fully_skipped: set[tuple] = set()
    done_runs: dict[tuple, set[int]] = defaultdict(set)
    for row in rows:
        if not _row_is_complete(row):
            continue
        identity = _row_identity(row)
        if (row.get("skip_reason") or "").strip():
            fully_skipped.add(identity)
            continue
        eval_run = row.get("eval_run", "")
        try:
            run_idx = int(float(eval_run))
        except (TypeError, ValueError):
            run_idx = 0
        done_runs[identity].add(run_idx)
    return fully_skipped, done_runs


def _spreadsheet_row(
    correction: dict,
    *,
    success: bool | str = "",
    eval_run: int | str = "",
    output_video: str = "",
    skip_reason: str = "",
) -> dict:
    return {
        "participant_id": correction.get("user_id", ""),
        "survey_video": correction.get("video", ""),
        "timestamp_s": correction.get("timestamp_s", ""),
        "fork_timestep": correction.get("timestep", ""),
        "correction": correction.get("instruction", ""),
        "success": success,
        "output_video": output_video,
        "source_video": correction.get("original_video_path", ""),
        "source_hdf5": correction.get("source_hdf5", ""),
        "source_run": correction.get("source_run", ""),
        "eval_run": eval_run,
        "interface_mode": correction.get("interface_mode", ""),
        "interface_condition": correction.get("interface_condition", ""),
        "task": correction.get("task", ""),
        "skip_reason": skip_reason,
    }


def main() -> None:
    survey_rollouts = load_survey_rollouts(
        args_cli.survey_tasks, source_run_root=args_cli.source_run
    )
    corrections = load_corrections(args_cli.corrections, survey_rollouts)
    if not corrections:
        print("No corrections found. Exiting.")
        return

    if args_cli.output_folder_name is None:
        args_cli.output_folder_name = (
            get_timestamp() + f"_correction_eval_{args_cli.policy}"
        )

    output_dir = os.path.join(PACKAGE_DIR, "output", args_cli.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    episode_results_file, episode_results = init_experiment(output_dir)
    spreadsheet_path = os.path.join(output_dir, "correction_eval_results.csv")
    spreadsheet_rows = load_spreadsheet(spreadsheet_path)
    fully_skipped, done_runs = completed_from_spreadsheet(spreadsheet_rows)

    num_runs = args_cli.num_runs
    print(f"Output directory: {output_dir}")
    print(f"Spreadsheet: {spreadsheet_path}")
    print(f"{num_runs} run(s) per correction entry")
    if spreadsheet_rows:
        print(
            f"Resuming from {len(spreadsheet_rows)} existing spreadsheet row(s); "
            "already-completed corrections/runs will be skipped."
        )

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
        source_run = correction["source_run"]
        source_env = correction.get("source_env", 0)
        source_hdf5 = correction["source_hdf5"]
        identity = _correction_identity(correction)

        if identity in fully_skipped:
            print(f"[{idx}] Previously skipped; leaving existing spreadsheet row.")
            continue

        pending_runs = [
            run_idx for run_idx in range(num_runs)
            if run_idx not in done_runs.get(identity, set())
        ]
        if not pending_runs:
            print(f"[{idx}] Already done; skipping.")
            continue

        if not os.path.exists(source_hdf5):
            print(f"[{idx}] HDF5 not found: {source_hdf5}; skipping.")
            spreadsheet_rows.append(_spreadsheet_row(
                correction, skip_reason=f"HDF5 not found: {source_hdf5}"
            ))
            write_spreadsheet(spreadsheet_path, spreadsheet_rows)
            continue

        task_envs = get_envs(task=[task])
        if not task_envs:
            print(f"[{idx}] Task '{task}' not registered; skipping.")
            spreadsheet_rows.append(_spreadsheet_row(
                correction, skip_reason=f"Task not registered: {task}"
            ))
            write_spreadsheet(spreadsheet_path, spreadsheet_rows)
            continue
        task_env = task_envs[0]

        scene_rel = f"{task}_correction_{idx}"
        scene_output_dir = os.path.join(output_dir, scene_rel)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        env, env_cfg = create_env(
            task_env,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            policy=args_cli.policy,
        )

        for run_idx in pending_runs:
            run_label = f" (run {run_idx + 1}/{num_runs})" if num_runs > 1 else ""
            print(
                f"\n\033[96m[{idx}/{len(corrections)}] Forking '{task}' "
                f"({correction.get('video', '?')} → run_{source_run}.hdf5) "
                f"at step {timestep}{run_label} "
                f"with instruction: '{corrected_instruction}'\033[0m"
            )

            env_results, msgs, timing = run_forked_episode(
                env=env,
                env_cfg=env_cfg,
                episode=run_idx,
                client=client,
                hdf5_path=source_hdf5,
                fork_timestep=timestep,
                fork_instruction=corrected_instruction,
                source_env=source_env,
                save_videos=save_videos,
                video_mode=args_cli.video_mode,
                headless=args_cli.headless,
            )

            run_name = f"{task}_correction_{idx}_{run_idx}"
            episode_results = summarize_run(
                env_results=env_results,
                msgs=msgs,
                env=env,
                env_cfg=env_cfg,
                num_envs=args_cli.num_envs,
                run_idx=run_idx,
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
                    "source_run": source_run,
                    "source_hdf5": source_hdf5,
                    "video": correction.get("video"),
                    "original_video_path": correction.get("original_video_path"),
                    "timestamp_s": correction.get("timestamp_s"),
                    "user_id": correction.get("user_id"),
                    "interface_mode": correction.get("interface_mode"),
                    "interface_condition": correction.get("interface_condition"),
                },
            )

            for env_result in env_results:
                env_id = env_result["env_id"]
                video_name = _output_video_filename(
                    corrected_instruction, run_idx, args_cli.num_envs, env_id
                )
                spreadsheet_rows.append(_spreadsheet_row(
                    correction,
                    success=bool(env_result["success"]),
                    eval_run=run_idx,
                    output_video=os.path.join(scene_rel, video_name),
                ))
            write_spreadsheet(spreadsheet_path, spreadsheet_rows)

        env.close()

    write_spreadsheet(spreadsheet_path, spreadsheet_rows)
    print(f"Wrote {len(spreadsheet_rows)} spreadsheet row(s) to {spreadsheet_path}")
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
