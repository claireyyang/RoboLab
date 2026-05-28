# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Evaluation orchestration: episode runner, inference ABC, summarize helpers,
plus shared argparse / per-task-loop helpers for the per-policy runner scripts
under ``policies/<policy>/run.py``.

Concrete policy clients live in ``policies/<policy>/client.py`` and are imported
directly by their corresponding runner scripts — there is no central registry.

Submodules are loaded lazily via :pep:`562` ``__getattr__`` so that callers can
``from robolab.eval.runner import add_common_eval_args`` *before* ``AppLauncher``
launches without triggering the isaaclab-dependent imports in :mod:`episode`
and :mod:`summarize`. Public names resolve at attribute-access time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_client import InferenceClient
    from .episode import run_episode, run_forked_episode
    from .runner import add_common_eval_args, run_evaluation
    from .state_restoration import load_state_at_timestep, restore_scene_state
    from .summarize import summarize_run

__all__ = [
    "InferenceClient",
    "add_common_eval_args",
    "load_state_at_timestep",
    "restore_scene_state",
    "run_episode",
    "run_evaluation",
    "run_forked_episode",
    "summarize_run",
]


def __getattr__(name: str):
    if name == "InferenceClient":
        from .base_client import InferenceClient

        return InferenceClient
    if name == "run_episode":
        from .episode import run_episode

        return run_episode
    if name == "run_forked_episode":
        from .episode import run_forked_episode

        return run_forked_episode
    if name == "summarize_run":
        from .summarize import summarize_run

        return summarize_run
    if name in ("load_state_at_timestep", "restore_scene_state"):
        from . import state_restoration

        return getattr(state_restoration, name)
    if name in ("add_common_eval_args", "run_evaluation"):
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
