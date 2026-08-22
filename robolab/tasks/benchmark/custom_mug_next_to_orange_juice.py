# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # success = DoneTerm(
    #     func=object_in_container,
    #     params={
    #         "object": "yogurt_cup",
    #         "container": "bowl",
    #         "require_gripper_detached": True
    #     },
    # )


@dataclass
class CustomMugNextToOrangeJuiceTask(Task):
    """Task: Place the ceramic mug next to the orange juice."""
    contact_object_list = [
        "table", "bowl", "banana", "bagel_07", "coffee_can", "banana_01",
        "yogurt_cup", "coffee_pot", "ceramic_mug", "pitcher", "fork_big",
        "spoon_big", "apple_01", "orange2", "milk_carton",
        "orange_juice_carton", "bagel_01", "bagel_02", "plate_small",
        "plate_large"
    ]
    scene = import_scene("breakfast_table.usda", contact_object_list)
    terminations = Terminations
    instruction = {
        "default": "Place the ceramic mug next to the orange juice",
        "vague": "Place the mug next to the orange juice",
        "specific": "Pick up the ceramic mug and place it next to the orange juice",
    }
    episode_length_s: int = 30
    attributes = ['color', 'size']
    subtasks = [
        pick_and_place(
            object=["ceramic_mug"],
            container="table",
            logical="all",
            score=1.0
        )
    ]
