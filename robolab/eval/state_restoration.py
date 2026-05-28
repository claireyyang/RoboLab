"""Restore simulation state from an HDF5 trajectory snapshot.

Used by the correction-evaluation pipeline to fork a rollout from the exact
world state a participant saw when they issued a correction, rather than
replaying from the beginning (which diverges due to physics non-determinism).

The ``states/`` group written by :class:`PostStepStatesRecorder` via
``scene.get_state(is_relative=True)`` contains per-step data for every
articulation (root pose, root velocity, joint positions, joint velocities)
and rigid object (root pose, root velocity).  Positions are stored
**relative** to env origins and must be shifted back to world frame when
writing into the sim.
"""

from __future__ import annotations

import logging

import h5py
import torch

logger = logging.getLogger(__name__)


def load_state_at_timestep(hdf5_path: str, episode: int, timestep: int) -> dict:
    """Load the full scene state dict at a single timestep from an HDF5 run file.

    Args:
        hdf5_path: Path to the run HDF5 file (e.g. ``run_0.hdf5``).
        episode: Demo index inside the file (typically matches env_id).
        timestep: The simulation step index to load.

    Returns:
        Nested dict mirroring the ``states/`` group structure.  Leaf values are
        numpy arrays with the leading ``num_envs`` dimension squeezed to a
        single row (the one corresponding to *episode*).
    """
    state: dict = {}
    with h5py.File(hdf5_path, "r") as f:
        states_grp = f[f"data/demo_{episode}/states"]
        _walk_hdf5_group(states_grp, timestep, state)
    return state


def _walk_hdf5_group(group: h5py.Group, timestep: int, out: dict) -> None:
    """Recursively read all datasets at *timestep* from an HDF5 group."""
    for key in group.keys():
        child = group[key]
        if isinstance(child, h5py.Group):
            out[key] = {}
            _walk_hdf5_group(child, timestep, out[key])
        elif isinstance(child, h5py.Dataset):
            out[key] = child[timestep]


def restore_scene_state(
    env,
    state_dict: dict,
    env_ids: torch.Tensor | list[int] | None = None,
) -> None:
    """Write a previously-saved scene state into the physics simulation.

    This mirrors ``_reset_assets_to_default()`` in ``reset_pose.py`` but reads
    from a snapshot dict (produced by :func:`load_state_at_timestep`) instead
    of the asset's ``default_root_state``.

    After calling this, the caller should step the sim once (or call
    ``env.scene.write_data_to_sim()`` + ``env.sim.step()``) to let PhysX
    settle before starting the policy loop.

    Args:
        env: A :class:`RobolabEnv` (or any ``ManagerBasedRLEnv``).
        state_dict: Nested dict from :func:`load_state_at_timestep`.
            Expected top-level keys match ``scene.get_state()`` output:
            ``articulation/<name>/...`` and ``rigid_object/<name>/...``.
        env_ids: Which parallel envs to restore.  ``None`` → all envs.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, list):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    origins = env.scene.env_origins[env_ids]

    # --- Articulations (robot + any articulated objects) -------------------
    art_states = state_dict.get("articulation", {})
    for name, articulation in env.scene.articulations.items():
        art_data = art_states.get(name)
        if art_data is None:
            logger.warning("No saved state for articulation '%s'; skipping.", name)
            continue

        root_pose = torch.as_tensor(
            art_data["root_pose"], dtype=torch.float32, device=env.device
        )
        root_vel = torch.as_tensor(
            art_data["root_velocity"], dtype=torch.float32, device=env.device
        )
        joint_pos = torch.as_tensor(
            art_data["joint_position"], dtype=torch.float32, device=env.device
        )
        joint_vel = torch.as_tensor(
            art_data["joint_velocity"], dtype=torch.float32, device=env.device
        )

        if root_pose.ndim == 1:
            root_pose = root_pose.unsqueeze(0).expand(len(env_ids), -1).clone()
            root_vel = root_vel.unsqueeze(0).expand(len(env_ids), -1).clone()
            joint_pos = joint_pos.unsqueeze(0).expand(len(env_ids), -1).clone()
            joint_vel = joint_vel.unsqueeze(0).expand(len(env_ids), -1).clone()

        # Saved positions are relative to env origins; convert back to world
        root_pose[:, :3] += origins

        articulation.write_root_pose_to_sim(root_pose[:, :7], env_ids=env_ids)
        articulation.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        articulation.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # --- Rigid objects (banana, bowl, cube, etc.) -------------------------
    rigid_states = state_dict.get("rigid_object", {})
    for name, rigid_object in env.scene.rigid_objects.items():
        obj_data = rigid_states.get(name)
        if obj_data is None:
            logger.warning("No saved state for rigid object '%s'; skipping.", name)
            continue

        root_pose = torch.as_tensor(
            obj_data["root_pose"], dtype=torch.float32, device=env.device
        )
        root_vel = torch.as_tensor(
            obj_data["root_velocity"], dtype=torch.float32, device=env.device
        )

        if root_pose.ndim == 1:
            root_pose = root_pose.unsqueeze(0).expand(len(env_ids), -1).clone()
            root_vel = root_vel.unsqueeze(0).expand(len(env_ids), -1).clone()

        root_pose[:, :3] += origins

        rigid_object.write_root_pose_to_sim(root_pose[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    # --- Deformable objects (cloth, etc.) ---------------------------------
    deformable_states = state_dict.get("deformable", {})
    for name, deformable_object in env.scene.deformable_objects.items():
        deform_data = deformable_states.get(name)
        if deform_data is None:
            continue

        nodal_pos = torch.as_tensor(
            deform_data["nodal_position"], dtype=torch.float32, device=env.device
        )
        nodal_vel = torch.as_tensor(
            deform_data["nodal_velocity"], dtype=torch.float32, device=env.device
        )
        if nodal_pos.ndim == 2:
            nodal_state = (
                torch.cat([nodal_pos, nodal_vel], dim=-1)
                .unsqueeze(0)
                .expand(len(env_ids), -1, -1)
                .clone()
            )
        else:
            nodal_state = torch.cat([nodal_pos, nodal_vel], dim=-1)

        nodal_state[:, :, :3] += origins.unsqueeze(1)
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
