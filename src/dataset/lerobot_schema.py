"""EgoSteer-RealWorld v1.11 field layout and model-input mappings."""

import numpy as np


MOTION_SLICES = {
    "left_arm_joints": slice(0, 7),
    "right_arm_joints": slice(7, 14),
    "left_hand_motors": slice(14, 20),
    "right_hand_motors": slice(20, 26),
    "left_wrist_pose": slice(26, 35),
    "right_wrist_pose": slice(35, 44),
    "left_fingertips": slice(44, 59),
    "right_fingertips": slice(59, 74),
}
CALIBRATION_SHAPES = {
    **{f"calibration.{cam}_intrinsics": (3, 3) for cam in ("head", "chest")},
    **{f"calibration.{cam}_world2cam": (4, 4) for cam in ("head", "chest")},
    **{f"calibration.{cam}_cam_to_{side}_base": (4, 4)
       for cam in ("head", "chest") for side in ("left", "right")},
}
FRAME_COLUMNS = ("observation.state", "action", "timestamp", "frame_index",
                 "episode_index", "index", "task_index")
VIDEO_KEYS = tuple(f"observation.images.{cam}{suffix}"
                   for cam in ("head", "chest") for suffix in ("", "_depth"))
EPISODE_COLUMNS = {
    "episode_index", "length", "tasks", "instructions", "dataset_from_index",
    "dataset_to_index", "data/chunk_index", "data/file_index", *CALIBRATION_SHAPES,
    *(f"videos/{key}/{field}" for key in VIDEO_KEYS
      for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")),
}


def camera_parameters(episode, camera):
    """Return float32 intrinsic4 and flattened world-to-camera extrinsic16."""
    intrinsic = episode[f"calibration.{camera}_intrinsics"]
    extrinsic = episode[f"calibration.{camera}_world2cam"]
    return (
        intrinsic[[0, 1, 0, 1], [0, 1, 2, 2]].astype(np.float32),
        extrinsic.reshape(16).astype(np.float32),
    )


def unpack_motion(values):
    """Map 74D to wrist18/fingertips30; preserve the precomputed world-frame FK.

    Wrist order is [Lxyz, Rxyz, Lrot6d, Rrot6d], not the source's two 9D blocks.
    The 26 joint/motor channels are not used by the fingertip model.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 74:
        raise ValueError(f"expected [T,74] motion, got {values.shape}")
    left = values[:, MOTION_SLICES["left_wrist_pose"]]
    right = values[:, MOTION_SLICES["right_wrist_pose"]]
    wrist = np.concatenate([left[:, :3], right[:, :3], left[:, 3:], right[:, 3:]], axis=-1)
    hand = np.concatenate([
        values[:, MOTION_SLICES["left_fingertips"]],
        values[:, MOTION_SLICES["right_fingertips"]],
    ], axis=-1)
    return wrist, hand
