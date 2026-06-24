'''
Data transformation functions for EgoSteer datasets.
'''

import random
from typing import Optional

# Source: https://albumentations.ai/docs/benchmarks/image-benchmarks/
# ColorJitter on CPU: ~11-13x faster than torchvision (1199 vs 88 img/s).
import albumentations as A
import cv2
import numpy as np
import torch

# cv2 defaults to using all cores for its internal thread pool. With multiple
# dataloader workers each spawning that pool, they would trample each other.
# Force single-threaded so each worker only uses its assigned affinity cores.
cv2.setNumThreads(0)

from src.utils.geometry import (
    transform_wrist_to_target_frame,
    homo_matrix_from_trans_6drot,
    homo_matrix_to_trans_6drot,
    transform_hand_points_to_wrist_frame,
)
from src.model.common.normalizer import LinearNormalizer

# Module-level color augmentation. Stateless: samples new params per __call__.
# Hue intentionally dropped: it harmed color-sensitive VLA/VLM tasks.
# Using albumentations for ~11x CPU speedup vs torchvision on ColorJitter.
COLOR_AUG = A.ColorJitter(
    brightness=(0.7, 1.3),
    contrast=(0.7, 1.3),
    saturation=(0.7, 1.3),
    hue=0,
    p=1.0,
)


def get_relative_action(state, action):
    '''
    Args:
        state: np.ndarray, shape: [wrist_dim + hand_dim]
        action: np.ndarray, shape: [H, wrist_dim + hand_dim]
    Returns:
        action: np.ndarray, shape: [H, wrist_dim + hand_dim]
    '''
    action = action.copy() # avoid modifying the original action
    for idx in range(2):
        wrist_action_homo_mat = homo_matrix_from_trans_6drot(action[..., idx*3 : idx*3+3], action[..., 6+idx*6 : 6+idx*6+6])
        wrist_state_homo_mat = homo_matrix_from_trans_6drot(state[idx*3 : idx*3+3], state[6+idx*6 : 6+idx*6+6])
        wrist_action_homo_mat = np.linalg.pinv(wrist_state_homo_mat) @ wrist_action_homo_mat
        trans, rot_6d = homo_matrix_to_trans_6drot(wrist_action_homo_mat)
        action[..., idx*3 : idx*3+3] = trans
        action[..., 6+idx*6 : 6+idx*6+6] = rot_6d

    action[..., 18:] = action[..., 18:] - state[18:]
    return action

def get_absolute_action(state, relative_action):
    '''
    Convert relative action back to absolute action.
    This is the inverse operation of get_relative_action.

    Args:
        state: torch.Tensor or np.ndarray, shape: [wrist_dim + hand_dim] - current state
            state is in the first frame's camera coordinate system, where wrist is in cam frame and hand is in wrist frame
        relative_action: torch.Tensor or np.ndarray, shape: [H, wrist_dim + hand_dim] - relative action

    Returns:
        absolute_action: torch.Tensor or np.ndarray, shape: [H, wrist_dim + hand_dim] - absolute action
    '''
    # Create a copy of relative_action to store absolute_action
    if isinstance(relative_action, torch.Tensor):
        absolute_action = relative_action.clone()
    else:
        absolute_action = relative_action.copy()

    # For wrist parameters, absolute action = state @ relative action
    # This is the inverse of: relative = pinv(state) @ action
    for idx in range(2):
        wrist_relative_action_homo_mat = homo_matrix_from_trans_6drot(relative_action[..., idx*3 : idx*3+3], relative_action[..., 6+idx*6 : 6+idx*6+6])
        wrist_state_homo_mat = homo_matrix_from_trans_6drot(state[idx*3 : idx*3+3], state[6+idx*6 : 6+idx*6+6])
        wrist_action_homo_mat = wrist_state_homo_mat @ wrist_relative_action_homo_mat
        trans, rot_6d = homo_matrix_to_trans_6drot(wrist_action_homo_mat)
        absolute_action[..., idx*3 : idx*3+3] = trans
        absolute_action[..., 6+idx*6 : 6+idx*6+6] = rot_6d

    # For hand parameters, absolute action = relative action + state
    # This is the inverse of: relative = action - state
    absolute_action[..., 18:] = relative_action[..., 18:] + state[18:]

    return absolute_action


def process_state_action(
    wrist_state,
    hand_state,
    wrist_action,
    hand_action,
    extrinsic,
    hand_ndim,
    normalizer : Optional[LinearNormalizer] = None,
    motion_type = 'mano',
    use_relative_action = False,
):
    '''
    Args:
        wrist_state: np.ndarray, shape: [N_state, wrist_dim]
        hand_state: np.ndarray, shape: [N_state, all_hand_dim]
        wrist_action: np.ndarray, shape: [N_action, wrist_dim]
        hand_action: np.ndarray, shape: [N_action, all_hand_dim]
        extrinsic: np.ndarray, shape: [4, 4]
        hand_ndim: int
        normalizer: Optional[LinearNormalizer]
        motion_type: str, 'mano' or 'keypoint'
        use_relative_action: bool
    Returns:
        state: np.ndarray, shape: [N_state, wrist_dim + hand_dim]
        action: np.ndarray, shape: [N_action, wrist_dim + hand_dim]
    '''
    # use first self.hand_ndim components of hand state and action
    all_hand_ndim = hand_state.shape[-1] // 2 # per hand dims, i.e. 45 in MANO hand params
    hand_state = np.concatenate([
        hand_state[:, :hand_ndim],
        hand_state[:, all_hand_ndim:all_hand_ndim + hand_ndim]
    ], axis=-1)
    hand_action = np.concatenate([
        hand_action[:, :hand_ndim],
        hand_action[:, all_hand_ndim:all_hand_ndim + hand_ndim]
    ], axis=-1)

    if motion_type == 'fingertips':
        # TODO: We can try transform the fingertips to the camera coordinate system or wrist frame coordinate system
        processed_hand_state = transform_hand_points_to_wrist_frame(hand_state, wrist_state)
        processed_hand_state = processed_hand_state.reshape(hand_state.shape)
        processed_hand_action = transform_hand_points_to_wrist_frame(hand_action, wrist_action)
        processed_hand_action = processed_hand_action.reshape(hand_action.shape)
    elif motion_type == 'mano':
        processed_hand_state = hand_state
        processed_hand_action = hand_action
    else:
        raise ValueError(f"Unsupported motion type: {motion_type}")

    # transform the wrist state and action to the camera coordinate system
    processed_wrist_state = transform_wrist_to_target_frame(wrist_state, extrinsic)
    processed_wrist_action = transform_wrist_to_target_frame(wrist_action, extrinsic)

    # use delta of wrist translation and hand mano params as action
    processed_state = np.concatenate([processed_wrist_state, processed_hand_state], axis=-1)
    processed_action = np.concatenate([processed_wrist_action, processed_hand_action], axis=-1)
    if use_relative_action:
        processed_action = get_relative_action(processed_state[-1], processed_action)

    if normalizer is None:
        return processed_state, processed_action
    if use_relative_action:
        return normalizer['states'](processed_state), normalizer['actions'](processed_action)
    return normalizer['motions'](processed_state), normalizer['motions'](processed_action)

def random_resized_crop(images, depth_images, intrinsic, scale_range=(0.9, 1.0)):
    '''Random crop then resize back. Same crop for all frames (temporal consistency).'''
    N, H, W = images.shape[:3]
    scale = random.uniform(*scale_range)
    crop_h, crop_w = int(H * scale), int(W * scale)
    if crop_h >= H and crop_w >= W:
        return images, depth_images, intrinsic

    y0 = random.randint(0, H - crop_h)
    x0 = random.randint(0, W - crop_w)
    sx, sy = W / crop_w, H / crop_h

    # Crop = numpy slicing; resize = cv2 (SIMD-optimized, ~10x faster than PIL).
    cropped = images[:, y0:y0 + crop_h, x0:x0 + crop_w]
    images = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in cropped])

    if depth_images is not None:
        cropped_depth = depth_images[:, y0:y0 + crop_h, x0:x0 + crop_w]
        depth_images = np.stack([
            cv2.resize(f, (W, H), interpolation=cv2.INTER_NEAREST) for f in cropped_depth
        ])

    # Update intrinsic [fx, fy, cx, cy] for the crop-then-resize transform
    if intrinsic is not None:
        intrinsic = intrinsic.copy()
        intrinsic[0] *= sx                      # fx
        intrinsic[1] *= sy                      # fy
        intrinsic[2] = (intrinsic[2] - x0) * sx # cx
        intrinsic[3] = (intrinsic[3] - y0) * sy # cy

    return images, depth_images, intrinsic


def augment_color(images):
    '''Color jitter via albumentations on uint8 numpy array.

    Stack N frames vertically into a single [N*H, W, 3] image so one call to
    A.ColorJitter samples params once and applies them to all frames,
    preserving temporal consistency. ColorJitter is pixel-wise so this is
    strictly equivalent to per-frame apply with shared params, but avoids
    N times the Python dispatch overhead.
    '''
    N, H, W, C = images.shape
    stacked = np.ascontiguousarray(images.reshape(N * H, W, C))
    jittered = COLOR_AUG(image=stacked)["image"]
    return jittered.reshape(N, H, W, C)


def augment_depth(depth_images, noise_scale=0.005, dropout_prob=0.5):
    '''Depth-dependent Gaussian noise + random rectangular dropout (shared across frames).
    Noise sigma = noise_scale * depth; at 1m ≈ 5mm. Dropout area ≤ ~16%.'''
    depth_images = depth_images.copy()
    N, H, W = depth_images.shape

    # Depth-dependent Gaussian noise: only on valid (>0) pixels
    valid = depth_images > 0
    noise = np.random.randn(N, H, W).astype(np.float32)
    depth_images[valid] += noise[valid] * noise_scale * depth_images[valid]
    np.maximum(depth_images, 0, out=depth_images)

    # Random rectangular dropout
    if random.random() < dropout_prob:
        rh = random.randint(1, max(1, int(H * 0.4)))
        rw = random.randint(1, max(1, int(W * 0.4)))
        ry = random.randint(0, H - rh)
        rx = random.randint(0, W - rw)
        depth_images[:, ry:ry + rh, rx:rx + rw] = 0

    return depth_images


def resize_frames(frames, target_hw, interpolation=cv2.INTER_LINEAR):
    '''Resize a batch of frames to target (H, W) via cv2.resize.

    Args:
        frames: np.ndarray, shape [N, H, W, C] (uint8 RGB) or [N, H, W] (depth float32).
        target_hw: (target_H, target_W).
        interpolation: cv2 interpolation flag. Use cv2.INTER_NEAREST for depth
            to avoid interpolation artifacts at depth discontinuities.
    Returns:
        np.ndarray with the same dtype, resized to target spatial dimensions.
    '''
    tH, tW = target_hw
    if frames.shape[1] == tH and frames.shape[2] == tW:
        return frames
    # cv2.resize takes (W, H), not (H, W).
    return np.stack([cv2.resize(f, (tW, tH), interpolation=interpolation) for f in frames])


def process_image(image, depth_image=None, intrinsic=None, aug_transform=None,
                  depth_clip_range=None, target_size=None):
    '''
    Args:
        image: np.ndarray, shape: [N, H, W, 3], uint8
        depth_image: np.ndarray or None, shape: [N, H, W]
        intrinsic: np.ndarray or None, shape: [4] — [fx, fy, cx, cy]
        aug_transform: truthy value enables augmentation (the object itself is not called)
        depth_clip_range: [min, max] in meters, or None
        target_size: (H, W) tuple or None. When set, all frames are resized to
            this resolution before augmentation. Required when world model
            (future frame prediction) is enabled so that temporal attention
            patches align across frames with identical spatial semantics.
    Returns:
        image: np.ndarray, shape: [N, H, W, 3]
        depth_image: np.ndarray or None, shape: [N, H, W], float32 (meters)
        intrinsic: np.ndarray or None, shape: [4]
    '''
    # Resize to target resolution before any augmentation.
    if target_size is not None:
        tH, tW = target_size
        _, H, W, _ = image.shape
        if H != tH or W != tW:
            sx, sy = tW / W, tH / H
            image = resize_frames(image, target_size, interpolation=cv2.INTER_LINEAR)
            if depth_image is not None:
                depth_image = resize_frames(depth_image, target_size, interpolation=cv2.INTER_NEAREST)
            if intrinsic is not None:
                intrinsic = intrinsic.copy()
                intrinsic[0] *= sx  # fx
                intrinsic[1] *= sy  # fy
                intrinsic[2] *= sx  # cx
                intrinsic[3] *= sy  # cy

    # Depth stored as uint16 in millimeters; convert to float32 meters.
    # If already float, assume meters and skip conversion.
    if depth_image is not None and depth_image.dtype == np.uint16:
        depth_image = depth_image.astype(np.float32) / 1000.0

    if aug_transform:
        # image, depth_image, intrinsic = random_resized_crop(image, depth_image, intrinsic)
        image = augment_color(image)
        if depth_image is not None:
            depth_image = augment_depth(depth_image)

    if depth_clip_range is not None and depth_image is not None:
        depth_image = np.clip(depth_image, depth_clip_range[0], depth_clip_range[1])

    return image, depth_image, intrinsic


def compute_relative_motion_padded(
    current_flat16: Optional[np.ndarray],
    future_flat: Optional[np.ndarray],
    n_valid: int,
    K: int,
) -> np.ndarray:
    """Future camera pose expressed in the current camera frame.

    The dataset stores T_world→cam extrinsics, so the transform that takes
    a point in future-cam coordinates back to current-cam coordinates is
    ``T_cam_cur ← cam_fut = T_w2c_cur @ T_c2w_fut = T_w2c_cur @ inv(T_w2c_fut[k])``.

    Invalid steps (k >= n_valid or source missing) are zero-filled;
    downstream mask (frame_valid = arange(K) < n_future_frames) drops them.
    Flattened to 16D (row-major) per future step for loader emission.
    """
    out = np.zeros((K, 16), dtype=np.float32)
    if current_flat16 is None or future_flat is None or n_valid <= 0:
        return out
    n = min(int(n_valid), int(K), int(future_flat.shape[0]))
    if n <= 0:
        return out
    T_cur = current_flat16.reshape(4, 4).astype(np.float32)
    T_fut = future_flat[:n].reshape(n, 4, 4).astype(np.float32)
    T_fut_inv = np.linalg.inv(T_fut)
    # rel[k] = T_cur @ inv(T_fut[k])
    rel = np.einsum("ij,kjl->kil", T_cur, T_fut_inv)
    out[:n] = rel.reshape(n, 16)
    return out
