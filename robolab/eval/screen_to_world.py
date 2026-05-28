# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Project a participant's screen click back into the 3D simulation world.

Given a normalized 2D click position on the viewport video (the
``egocentric_mirrored_camera`` frame), this module casts a ray through the
camera's pinhole model and identifies which object (or world point) the
participant was pointing at.

Typical usage
-------------
>>> from robolab.eval.screen_to_world import identify_clicked_object
>>> result = identify_clicked_object(
...     u_frac=0.45, v_frac=0.62,
...     hdf5_path="output/.../run_0.hdf5",
...     episode=0, timestep=120,
... )
>>> result["closest_object"]   # e.g. "banana"
>>> result["world_point"]      # (x, y, z) on the table plane
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default intrinsics for EgocentricMirroredCameraCfg (the viewport camera).
# See robolab/variations/camera.py lines 152-172.
# ---------------------------------------------------------------------------
VIEWPORT_WIDTH = 864
VIEWPORT_HEIGHT = 480
VIEWPORT_FOCAL_LENGTH = 24.0       # mm (USD convention)
VIEWPORT_H_APERTURE = 20.955       # mm
VIEWPORT_V_APERTURE = 15.29        # mm


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics derived from Isaac Sim's USD camera model."""
    width: int = VIEWPORT_WIDTH
    height: int = VIEWPORT_HEIGHT
    focal_length: float = VIEWPORT_FOCAL_LENGTH
    h_aperture: float = VIEWPORT_H_APERTURE
    v_aperture: float = VIEWPORT_V_APERTURE

    @property
    def fx(self) -> float:
        return self.focal_length * self.width / self.h_aperture

    @property
    def fy(self) -> float:
        return self.focal_length * self.height / self.v_aperture

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0


@dataclass
class RaycastResult:
    """Result of projecting a screen click into the 3D world."""
    ray_origin: np.ndarray
    ray_direction: np.ndarray
    table_point: np.ndarray | None = None
    closest_object: str | None = None
    closest_distance: float | None = None
    all_distances: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def screen_to_ray(
    u_frac: float,
    v_frac: float,
    camera_pos: np.ndarray,
    camera_quat_xyzw: np.ndarray,
    intrinsics: CameraIntrinsics | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast a ray from a normalised screen coordinate through a pinhole camera.

    Args:
        u_frac: Horizontal click position as a fraction of image width [0, 1].
            0 = left edge, 1 = right edge.
        v_frac: Vertical click position as a fraction of image height [0, 1].
            0 = top edge, 1 = bottom edge.
        camera_pos: World-frame camera position (3,).
        camera_quat_xyzw: Camera orientation quaternion in **scipy / ROS
            convention** ``(x, y, z, w)``.  This is what Isaac Lab stores as
            ``camera.data.quat_w_ros`` and what gets saved into the HDF5 by
            :class:`InitialCameraExtrinsicsRecorder`.
        intrinsics: Camera intrinsic parameters.  Defaults to the viewport
            camera (``EgocentricMirroredCameraCfg``).

    Returns:
        ``(ray_origin, ray_direction)`` both as ``(3,)`` numpy arrays in
        the world frame.  ``ray_direction`` is unit-length.
    """
    if intrinsics is None:
        intrinsics = CameraIntrinsics()

    u_px = u_frac * intrinsics.width
    v_px = v_frac * intrinsics.height

    # Ray in camera frame (OpenGL: camera looks along -Z, Y is up)
    x_cam = (u_px - intrinsics.cx) / intrinsics.fx
    y_cam = -(v_px - intrinsics.cy) / intrinsics.fy  # pixel Y down → camera Y up
    z_cam = -1.0

    ray_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam)

    # Rotate into world frame
    R_cam2world = Rotation.from_quat(camera_quat_xyzw).as_matrix()
    ray_world = R_cam2world @ ray_cam

    return camera_pos.astype(np.float64), ray_world


def ray_plane_intersect(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    plane_z: float = 0.0,
) -> np.ndarray | None:
    """Intersect a ray with a horizontal plane at ``z = plane_z``.

    Returns the 3D intersection point, or ``None`` if the ray is parallel
    to (or pointing away from) the plane.
    """
    if abs(ray_dir[2]) < 1e-9:
        return None
    t = (plane_z - ray_origin[2]) / ray_dir[2]
    if t < 0:
        return None
    return ray_origin + t * ray_dir


def point_to_centroid_distance(
    point: np.ndarray,
    centroid: np.ndarray,
) -> float:
    """Euclidean distance between a 3D point and an object centroid."""
    return float(np.linalg.norm(point - centroid))


# ---------------------------------------------------------------------------
# HDF5 data helpers
# ---------------------------------------------------------------------------

def load_object_centroids_at_timestep(
    hdf5_path: str,
    episode: int,
    timestep: int,
) -> dict[str, np.ndarray]:
    """Load per-object centroids at a given timestep from an HDF5 run file.

    Reads from ``data/demo_{episode}/bbox/centroid/{object_name}`` which is
    stored as float16 in metres by :class:`PostStepBBoxRecorder`.

    Falls back to reading centroids from ``data/demo_{episode}/states/``
    rigid-object root poses if the bbox group is absent.

    Returns:
        ``{object_name: np.ndarray(3,)}``
    """
    centroids: dict[str, np.ndarray] = {}
    with h5py.File(hdf5_path, "r") as f:
        demo = f.get(f"data/demo_{episode}")
        if demo is None:
            return centroids

        # Preferred source: bbox recorder centroids
        bbox_grp = demo.get("bbox")
        if bbox_grp is not None:
            for key in bbox_grp.keys():
                if key.startswith("centroid/"):
                    obj_name = key.split("/", 1)[1]
                    data = bbox_grp[key]
                    centroids[obj_name] = np.array(data[timestep], dtype=np.float64)
            if centroids:
                return centroids

        # Fallback: rigid-object root poses from states/
        states_grp = demo.get("states")
        if states_grp is None:
            return centroids
        rigid_grp = states_grp.get("rigid_object")
        if rigid_grp is None:
            return centroids
        for obj_name in rigid_grp.keys():
            pose_ds = rigid_grp[obj_name].get("root_pose")
            if pose_ds is not None:
                pose = np.array(pose_ds[timestep], dtype=np.float64)
                centroids[obj_name] = pose[:3]

    return centroids


def load_camera_extrinsics(
    hdf5_path: str,
    episode: int,
    camera_name: str = "egocentric_mirrored_camera",
) -> tuple[np.ndarray, np.ndarray]:
    """Load camera position and orientation from the HDF5 initial state.

    The :class:`InitialCameraExtrinsicsRecorder` saves camera poses under
    ``initial_camera_extrinsics/<camera_name>/position`` and ``orientation``
    (quat_w_ros = x, y, z, w).

    Falls back to ``initial_state/cameras/<camera_name>/...`` (written by
    :class:`InitialStateRecorder` with ``camera_names``).

    Returns:
        ``(position(3,), quat_xyzw(4,))``
    """
    with h5py.File(hdf5_path, "r") as f:
        demo = f.get(f"data/demo_{episode}")
        if demo is None:
            raise ValueError(f"demo_{episode} not found in {hdf5_path}")

        for prefix in ("initial_camera_extrinsics", "initial_state/cameras"):
            grp = demo.get(f"{prefix}/{camera_name}")
            if grp is None:
                continue
            pos = np.array(grp["position"], dtype=np.float64).flatten()[:3]
            quat = np.array(grp["orientation"], dtype=np.float64).flatten()[:4]
            return pos, quat

    # Final fallback: use the static config values
    logger.warning(
        "Camera extrinsics for '%s' not found in HDF5; using static config defaults.",
        camera_name,
    )
    pos = np.array([1.5, 0.0, 1.0], dtype=np.float64)
    # Config rot is OpenGL (w,x,y,z) = (0.653, 0.271, 0.271, 0.653)
    # Convert to scipy (x,y,z,w)
    quat_xyzw = np.array([0.271, 0.271, 0.653, 0.653], dtype=np.float64)
    return pos, quat_xyzw


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def identify_clicked_object(
    u_frac: float,
    v_frac: float,
    hdf5_path: str,
    episode: int,
    timestep: int,
    camera_name: str = "egocentric_mirrored_camera",
    table_z: float = 0.0,
    intrinsics: CameraIntrinsics | None = None,
    max_match_distance: float = 0.15,
) -> RaycastResult:
    """Identify which object a participant clicked on in the viewport video.

    End-to-end: loads camera extrinsics and object centroids from the HDF5
    file, casts a ray through the click position, intersects with the table
    plane, and returns the closest object.

    Args:
        u_frac: Normalised horizontal click position [0, 1].
        v_frac: Normalised vertical click position [0, 1].
        hdf5_path: Path to the run HDF5 (e.g. ``run_0.hdf5``).
        episode: Demo index (typically 0 for single-env runs).
        timestep: Simulation step at which the participant clicked.
        camera_name: Name of the camera the participant was viewing.
        table_z: Z-height of the table surface for plane intersection.
        intrinsics: Camera intrinsics (defaults to viewport camera).
        max_match_distance: Maximum distance (metres) from the table-plane
            intersection to an object centroid for it to be considered a
            match.  Objects farther than this are still reported in
            ``all_distances`` but ``closest_object`` will be ``None``.

    Returns:
        A :class:`RaycastResult` with the ray, table intersection point,
        and the name + distance of the closest object (if any).
    """
    cam_pos, cam_quat_xyzw = load_camera_extrinsics(hdf5_path, episode, camera_name)
    ray_origin, ray_dir = screen_to_ray(
        u_frac, v_frac, cam_pos, cam_quat_xyzw, intrinsics,
    )

    table_point = ray_plane_intersect(ray_origin, ray_dir, plane_z=table_z)

    centroids = load_object_centroids_at_timestep(hdf5_path, episode, timestep)

    result = RaycastResult(
        ray_origin=ray_origin,
        ray_direction=ray_dir,
        table_point=table_point,
    )

    if table_point is None or not centroids:
        return result

    for obj_name, centroid in centroids.items():
        dist = point_to_centroid_distance(table_point, centroid)
        result.all_distances[obj_name] = dist

    if result.all_distances:
        closest = min(result.all_distances, key=result.all_distances.get)
        closest_dist = result.all_distances[closest]
        if closest_dist <= max_match_distance:
            result.closest_object = closest
            result.closest_distance = closest_dist

    return result
