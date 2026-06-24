"""
Mostly copied from transforms3d library

"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

_FLOAT_EPS = np.finfo(np.float32).eps

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# For testing whether a number is close to zero
_EPS4 = np.finfo(np.float32).eps * 4.0


def mat2euler(mat, axes="sxyz"):
    """Return Euler angles from rotation matrix for specified axis sequence.

    Note that many Euler angle triplets can describe one matrix.

    Parameters
    ----------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> R0 = euler2mat(1, 2, 3, 'syxz')
    >>> al, be, ga = mat2euler(R0, 'syxz')
    >>> R1 = euler2mat(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(mat, dtype=np.float32, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS4:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS4:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def quat2mat(q):
    """Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows quaternions that
    have not been normalized.

    References
    ----------
    Algorithm from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3, dtype=np.float32))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    """
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < _FLOAT_EPS:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ],
        dtype=np.float32,
    )


# Checks if a matrix is a valid rotation matrix.
def isrotation(
    R: np.ndarray,
    thresh=1e-6,
) -> bool:
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    iden = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(iden - shouldBeIdentity)
    return n < thresh


def euler2mat(ai, aj, ak, axes="sxyz"):
    """Return rotation matrix from Euler angles and axis sequence.

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    mat : array (3, 3)
        Rotation matrix or affine.

    Examples
    --------
    >>> R = euler2mat(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.eye(3, dtype=np.float32)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def euler2axangle(ai, aj, ak, axes="sxyz"):
    """Return angle, axis corresponding to Euler angles, axis specification

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    vector : array shape (3,)
       axis around which rotation occurs
    theta : scalar
       angle of rotation

    Examples
    --------
    >>> vec, theta = euler2axangle(0, 1.5, 0, 'szyx')
    >>> np.allclose(vec, [0, 1, 0])
    True
    >>> theta
    1.5
    """
    return quat2axangle(euler2quat(ai, aj, ak, axes))


def euler2quat(ai, aj, ak, axes="sxyz"):
    """Return `quaternion` from Euler angles and axis sequence `axes`

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Examples
    --------
    >>> q = euler2quat(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai = ai / 2.0
    aj = aj / 2.0
    ak = ak / 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,))
    if repetition:
        q[0] = cj * (cc - ss)
        q[i] = cj * (cs + sc)
        q[j] = sj * (cc + ss)
        q[k] = sj * (cs - sc)
    else:
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        q[j] *= -1.0

    return q


def quat2axangle(quat, identity_thresh=None):
    """Convert quaternion to rotation of angle around axis

    Parameters
    ----------
    quat : 4 element sequence
       w, x, y, z forming quaternion.
    identity_thresh : None or scalar, optional
       Threshold below which the norm of the vector part of the quaternion (x,
       y, z) is deemed to be 0, leading to the identity rotation.  None (the
       default) leads to a threshold estimated based on the precision of the
       input.

    Returns
    -------
    theta : scalar
       angle of rotation.
    vector : array shape (3,)
       axis around which rotation occurs.

    Examples
    --------
    >>> vec, theta = quat2axangle([0, 1, 0, 0])
    >>> vec
    array([1., 0., 0.])
    >>> np.allclose(theta, np.pi)
    True

    If this is an identity rotation, we return a zero angle and an arbitrary
    vector:

    >>> quat2axangle([1, 0, 0, 0])
    (array([1., 0., 0.]), 0.0)

    If any of the quaternion values are not finite, we return a NaN in the
    angle, and an arbitrary vector:

    >>> quat2axangle([1, np.inf, 0, 0])
    (array([1., 0., 0.]), nan)

    Notes
    -----
    A quaternion for which x, y, z are all equal to 0, is an identity rotation.
    In this case we return a 0 angle and an arbitrary vector, here [1, 0, 0].

    The algorithm allows for quaternions that have not been normalized.
    """
    quat = np.asarray(quat)
    Nq = np.sum(quat**2)
    if not np.isfinite(Nq):
        return np.array([1.0, 0, 0], dtype=np.float32), np.float32(np.nan)
    if identity_thresh is None:
        try:
            identity_thresh = np.finfo(Nq.type).eps * 3
        except (AttributeError, ValueError):  # Not a numpy type or not float
            identity_thresh = _FLOAT_EPS * 3
    if Nq < _FLOAT_EPS**2:  # Results unreliable after normalization
        return np.array([1.0, 0, 0]), 0.0
    if Nq != 1:  # Normalize if not normalized
        s = math.sqrt(Nq)
        quat = quat / s
    xyz = quat[1:]
    len2 = np.sum(xyz**2)
    if len2 < identity_thresh**2:
        # if vec is nearly 0,0,0, this is an identity rotation
        return np.array([1.0, 0, 0]), 0.0
    # Make sure w is not slightly above 1 or below -1
    theta = 2 * math.acos(max(min(quat[0], 1), -1))
    return xyz / math.sqrt(len2), theta


def quat2euler(quaternion, axes="sxyz"):
    """Euler angles from `quaternion` for specified axis sequence `axes`

    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> angles = quat2euler([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(angles, [0.123, 0, 0])
    True
    """
    return mat2euler(quat2mat(quaternion), axes)


def rot_matrix_from_6drot(rot):
    '''
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    The 6D representation uses two 3D vectors a and b, where:
    - The first vector a represents the first column of the rotation matrix
    - The second vector b represents the second column of the rotation matrix
    - The third column is computed as the cross product of a and b, then normalized
    
    Args:
        rot: torch.Tensor or np.ndarray, shape: [..., 6] or [6], where the first 3 elements are vector a,
             and the last 3 elements are vector b. Supports arbitrary dimensions.
    Returns:
        rot_matrix: torch.Tensor or np.ndarray, shape: [..., 3, 3] or [3, 3]
    '''
    if isinstance(rot, np.ndarray):
        is_numpy = True
        rot = torch.from_numpy(rot)
    else:
        is_numpy = False
    
    # Store original shape for later restoration
    original_shape = rot.shape
    
    # Handle single vector case (shape: [6])
    # Reshape to 2D for easier processing: [..., 6] -> [N, 6]
    rot = rot.reshape(-1, 6)
    
    # Extract the two 3D vectors
    a = rot[..., :3]  # First 3 elements: [N, 3]
    b = rot[..., 3:]  # Last 3 elements: [N, 3]
    
    # Schmidth orthogonalization
    a = F.normalize(a, dim=-1)
    b = b - torch.sum(a * b, dim=-1, keepdim=True) * a
    b = F.normalize(b, dim=-1)
    c = torch.cross(a, b, dim=-1)
    
    # Stack to form the rotation matrix
    rot_matrix = torch.stack([a, b, c], dim=-1)  # [N, 3, 3]
    
    # Reshape back to original dimensions if needed
    if len(original_shape) > 1:
        # Remove the last dimension (6) and add [3, 3] at the end
        new_shape = list(original_shape[:-1]) + [3, 3]
        rot_matrix = rot_matrix.reshape(*new_shape)
    else:
        rot_matrix = rot_matrix.squeeze(0)

    if is_numpy:
        rot_matrix = rot_matrix.cpu().numpy()
    
    return rot_matrix


def rot_matrix_to_6drot(rot_matrix):
    '''
    Convert 3x3 rotation matrix to 6D rotation representation.
    
    Args:
        rot_matrix: torch.Tensor or np.ndarray, shape: [..., 3, 3] or [3, 3]. Supports arbitrary dimensions.
    Returns:
        rot_6d: torch.Tensor or np.ndarray, shape: [..., 6] or [6]
    '''
    if isinstance(rot_matrix, np.ndarray):
        is_numpy = True
        rot_matrix = torch.from_numpy(rot_matrix)
    else:
        is_numpy = False
    
    # Store original shape for later restoration
    original_shape = rot_matrix.shape
    
    # Handle single rotation matrix case (shape: [3, 3])
    # Reshape to 3D for easier processing: [..., 3, 3] -> [N, 3, 3]
    rot_matrix = rot_matrix.reshape(-1, 3, 3)
    
    # Extract the first two columns
    a = rot_matrix[..., :, 0]  # First column: [N, 3]
    b = rot_matrix[..., :, 1]  # Second column: [N, 3]
    
    # Concatenate to form 6D representation
    rot_6d = torch.cat([a, b], dim=-1)  # [N, 6]
    
    # Reshape back to original dimensions if needed
    if len(original_shape) > 2:
        # Remove the last two dimensions (3, 3) and add [6] at the end
        new_shape = list(original_shape[:-2]) + [6]
        rot_6d = rot_6d.reshape(*new_shape)
    else:
        rot_6d = rot_6d.squeeze(0)
    
    if is_numpy:
        rot_6d = rot_6d.cpu().numpy()
    
    return rot_6d


def homo_coordinates_to_cartesian(homo_coordinates, eps=1e-6):
    '''
    Convert homogeneous coordinates to Cartesian coordinates.
    Args:
        homo_coordinates: torch.Tensor or np.ndarray, shape: [..., 4]
    Returns:
        cartesian_coordinates: torch.Tensor or np.ndarray, shape: [..., 3]
    '''
    cartesian_coordinates = homo_coordinates[..., :3] / (homo_coordinates[..., 3:4] + eps)
    return cartesian_coordinates


def homo_coordinates_from_cartesian(cartesian_coordinates):
    '''
    Convert homogeneous coordinates to Cartesian coordinates.
    Args:
        cartesian_coordinates: torch.Tensor or np.ndarray, shape: [..., 3]
    Returns:
        homo_coordinates: torch.Tensor or np.ndarray, shape: [..., 4]
    '''
    if isinstance(cartesian_coordinates, np.ndarray):
        homo_coordinates = np.concatenate([cartesian_coordinates, np.ones_like(cartesian_coordinates[..., :1])], axis=-1)
    else: 
        homo_coordinates = torch.cat([cartesian_coordinates, torch.ones_like(cartesian_coordinates[..., :1])], dim=-1)
    return homo_coordinates


def homo_matrix_to_trans_6drot(homo_matrix):
    '''
    Args:
        homo_matrix: torch.Tensor or np.ndarray, shape: [..., 4, 4]
    Returns:
        trans: torch.Tensor or np.ndarray, shape: [..., 3]
        rot_6d: torch.Tensor or np.ndarray, shape: [..., 6]
    '''
    rot_matrix = homo_matrix[..., :3, :3]
    trans = homo_matrix[..., :3, 3]
    rot_6d = rot_matrix_to_6drot(rot_matrix)
    return trans, rot_6d


def homo_matrix_from_trans_6drot(trans, rot_6d):
    '''
    Args:
        trans: torch.Tensor or np.ndarray, shape: [..., 3]
        rot_6d: torch.Tensor or np.ndarray, shape: [..., 6]
    Returns:
        homo_matrix: torch.Tensor or np.ndarray, shape: [..., 4, 4]
    '''
    assert trans.shape[:-1] == rot_6d.shape[:-1], "trans and rot_6d must have the same shape, except the last dimension"
    if isinstance(trans, np.ndarray):
        is_numpy = True
        trans = torch.from_numpy(trans)
        rot_6d = torch.from_numpy(rot_6d)
    else:
        is_numpy = False
    rot_matrix = rot_matrix_from_6drot(rot_6d)
    homo_matrix = torch.zeros(trans.shape[:-1] + (4, 4), device=trans.device, dtype=trans.dtype)
    homo_matrix[..., :3, :3] = rot_matrix
    homo_matrix[..., :3, 3] = trans
    homo_matrix[..., 3, 3] = 1
    if is_numpy:
        homo_matrix = homo_matrix.cpu().numpy()
    return homo_matrix


def homo_matrix_from_wrist_pose(wrist_pose): 
    '''
    Convert wrist pose to homogeneous matrix.
    Args:
        wrist_pose: torch.Tensor or np.ndarray, shape: [..., 18] 
        including left_trans(3), right_trans(3), left_rot6d(6), right_rot6d(6)
    Returns:
        homo_matrix_left: torch.Tensor or np.ndarray, shape: [..., 4, 4]
        homo_matrix_right: torch.Tensor or np.ndarray, shape: [..., 4, 4]
    '''
    assert wrist_pose.shape[-1] == 18, "wrist_pose must have 18 elements"
    left_trans = wrist_pose[..., :3]  # [..., 3]
    right_trans = wrist_pose[..., 3:6]  # [..., 3]
    left_rot6d = wrist_pose[..., 6:12]  # [..., 6]
    right_rot6d = wrist_pose[..., 12:18]  # [..., 6]
    homo_matrix_left = homo_matrix_from_trans_6drot(left_trans, left_rot6d)  # [..., 4, 4]
    homo_matrix_right = homo_matrix_from_trans_6drot(right_trans, right_rot6d)  # [..., 4, 4]
    return homo_matrix_left, homo_matrix_right


def transform_pose_to_target_frame(pose, target_extrinsic):
    f'''
    Transform the pose to the target frame.
    Args:
        pose: torch.Tensor or np.ndarray, shape: [T, 4, 4] or [B, T, 4, 4] in world frame
        target_extrinsic: torch.Tensor or np.ndarray, shape: [4, 4] or [T, 4, 4] or [B, 4, 4]
        we assume the target_extrinsic is world2cam, and we want to transform the pose in the world frame to the camera frame
        if wrist_pose.ndim > target_extrinsic.ndim, we assume to broadcast the time dimension of target_extrinsic
    Returns:
        pose: torch.Tensor or np.ndarray, shape: [T, 4, 4] or [B, T, 4, 4] in target frame
    '''
    assert pose.dtype == target_extrinsic.dtype, "pose and target_extrinsic must have the same dtype"
    # print(f"pose.shape: {pose.shape}, target_extrinsic.shape: {target_extrinsic.shape}")
    if isinstance(pose, np.ndarray):
        is_numpy = True
        pose = torch.from_numpy(pose)
        target_extrinsic = torch.from_numpy(target_extrinsic)
    else:
        is_numpy = False
    if pose.ndim > target_extrinsic.ndim:
        target_extrinsic = target_extrinsic.unsqueeze(-3) # unsqueeze the time dimension 

    '''
    # use pseudo-inverse to avoid NaN
    # for cam2world, we need to use the inverse of the target_extrinsic
    target_extrinsic_inv = torch.linalg.pinv(target_extrinsic)
    '''

    pose = torch.matmul(target_extrinsic, pose)
    
    # check if the result contains NaN and handle it
    if torch.isnan(pose).any():
        print(f"Warning: NaN detected in pose after transformation")
        pose = torch.where(torch.isnan(pose), torch.zeros_like(pose), pose)
    
    if is_numpy:
        pose = pose.numpy()
    return pose


# TODO: check whether the target_extrinsic is cam2world or world2cam
def transform_wrist_to_target_frame(wrist_pose, target_extrinsic):
    '''
    Transform the wrist action to the target frame.
    Args:
        wrist_pose: torch.Tensor or np.ndarray, shape: [T, 18] or [B, T, 18]
        target_extrinsic: torch.Tensor or np.ndarray, shape: [4, 4] or [T, 4, 4] or [B, 4, 4] or [B, T, 4, 4]
        we assume the target_extrinsic is world2cam, and we want to transform the wrist action in the world frame to the camera frame
        if wrist_pose.ndim > target_extrinsic.ndim, we assume to broadcast the time dimension of target_extrinsic
    Returns:
        wrist_pose: torch.Tensor or np.ndarray, shape: [T, 18] or [B, T, 18]
    '''
    assert wrist_pose.dtype == target_extrinsic.dtype, "wrist_pose and target_extrinsic must have the same dtype"
    
    homo_wrist_pose_left, homo_wrist_pose_right = homo_matrix_from_wrist_pose(wrist_pose)
    homo_wrist_pose_left = transform_pose_to_target_frame(homo_wrist_pose_left, target_extrinsic)
    homo_wrist_pose_right = transform_pose_to_target_frame(homo_wrist_pose_right, target_extrinsic)

    wrist_trans_left, wrist_rot_6d_left = homo_matrix_to_trans_6drot(homo_wrist_pose_left) # [..., 3], [..., 6]
    wrist_trans_right, wrist_rot_6d_right = homo_matrix_to_trans_6drot(homo_wrist_pose_right) # [..., 3], [..., 6]
    if isinstance(wrist_trans_left, np.ndarray):
        wrist_pose = np.concatenate([wrist_trans_left, wrist_trans_right, wrist_rot_6d_left, wrist_rot_6d_right], axis=-1)
    else: 
        wrist_pose = torch.cat([wrist_trans_left, wrist_trans_right, wrist_rot_6d_left, wrist_rot_6d_right], dim=-1)
    return wrist_pose


def transform_hand_points_to_target_frame(points, target_extrinsic):
    '''
    Transform the hand's points to the target frame.
    Args:
        points: torch.Tensor or np.ndarray, shape: [T, D] or [B, T, D]
            where D = (D//3) points * 3D (e.g. 5 left hand fingertip points + 5 right hand fingertip points)
        target_extrinsic: torch.Tensor or np.ndarray, shape: [4, 4] or [T, 4, 4] or [B, 4, 4] or [B, T, 4, 4]
        we assume the target_extrinsic is world2cam, and we want to transform the points in the world frame to the camera frame
        if points.ndim + 1 > target_extrinsic.ndim, we assume to broadcast the time dimension of target_extrinsic
    Returns:
        points: torch.Tensor or np.ndarray, shape: [T, D] or [B, T, D]
    '''
    assert points.dtype == target_extrinsic.dtype, "points and target_extrinsic must have the same dtype"
    # print(f"points.shape: {points.shape}, target_extrinsic.shape: {target_extrinsic.shape}")
    if isinstance(points, np.ndarray):
        is_numpy = True
        points = torch.from_numpy(points)
        target_extrinsic = torch.from_numpy(target_extrinsic)
    else:
        is_numpy = False

    if points.ndim + 1 > target_extrinsic.ndim:
        target_extrinsic = target_extrinsic.unsqueeze(-3) # unsqueeze the time dimension 

    points = rearrange(points, '... (d c) -> ... d c', c=3)
    points = homo_coordinates_from_cartesian(points).unsqueeze(-1) # [..., d, 3] -> [..., d, 4, 1]
    target_extrinsic = target_extrinsic.unsqueeze(-3) # [..., 4, 4] -> [..., 1, 4, 4], broadcast the number of points dimension
    points = torch.matmul(target_extrinsic, points).squeeze(-1) # [..., d, 4, 1] -> [..., d, 4]
    points = homo_coordinates_to_cartesian(points) # [..., d, 3]
    points = rearrange(points, '... d c -> ... (d c)', c=3) # [..., d, 3] -> [..., D]
    if is_numpy:
        points = points.numpy()
    return points


def transform_hand_points_to_wrist_frame(hand_points, wrist_pose):
    '''
    Transform the hand's points to the wrist frame.
    Args:
        hand_points: torch.Tensor or np.ndarray, shape: [T, D] or [B, T, D]
            where D = (D//3) points * 3D (e.g. 5 left hand fingertip points + 5 right hand fingertip points)
            each coordinate is in the base coordinate system
        wrist_pose: torch.Tensor or np.ndarray, shape: [T, wrist_dim] or [B, T, wrist_dim]
            i.e. wrist_dim = 18, [left_trans(3), right_trans(3), left_rot6d(6), right_rot6d(6)]
    Returns:
        hand_points: torch.Tensor or np.ndarray, shape: [T, D] or [B, T, D]
    '''
    assert hand_points.dtype == wrist_pose.dtype, "hand_points and wrist_pose must have the same dtype"
    if isinstance(hand_points, np.ndarray):
        is_numpy = True
        hand_points = torch.from_numpy(hand_points)
        wrist_pose = torch.from_numpy(wrist_pose)
    else:
        is_numpy = False

    D = hand_points.shape[-1]
    # Get left and right hand points
    hand_points_left = hand_points[..., :D//2]
    hand_points_right = hand_points[..., D//2:]
    # Get wrist pose homogeneous matrix, i.e. T_wrist2world
    homo_wrist_pose_left, homo_wrist_pose_right = homo_matrix_from_wrist_pose(wrist_pose)
    # Get T_world2wrist
    homo_world2wrist_left = torch.linalg.pinv(homo_wrist_pose_left)
    homo_world2wrist_right = torch.linalg.pinv(homo_wrist_pose_right)
    # Get T_wrist2world
    hand_points_left = transform_hand_points_to_target_frame(hand_points_left, homo_world2wrist_left)
    hand_points_right = transform_hand_points_to_target_frame(hand_points_right, homo_world2wrist_right)
    hand_points = torch.cat([hand_points_left, hand_points_right], dim=-1)

    if is_numpy:
        hand_points = hand_points.numpy()
    return hand_points


def transform_hand_points_from_wrist_to_camera_frame(hand_points, wrist_pose_in_camera):
    '''
    Transform the hand's points from wrist frame to camera frame.
    Args:
        hand_points: torch.Tensor or np.ndarray, shape: [T, D] or [B, T, D]
            where D = (D//3) points * 3D (e.g. 5 left hand fingertip points + 5 right hand fingertip points)
            each coordinate is in the wrist coordinate system
        wrist_pose_in_camera: torch.Tensor or np.ndarray, shape: [T, wrist_dim] or [B, T, wrist_dim]
            i.e. wrist_dim = 18, [left_trans(3), right_trans(3), left_rot6d(6), right_rot6d(6)]
            wrist pose in camera coordinate system
    Returns:
        hand_points: torch.Tensor or np.ndarray, shape: [T, D] or [B, T, D]
            hand points in camera coordinate system
    '''
    assert hand_points.dtype == wrist_pose_in_camera.dtype, "hand_points and wrist_pose_in_camera must have the same dtype"
    if isinstance(hand_points, np.ndarray):
        is_numpy = True
        hand_points = torch.from_numpy(hand_points)
        wrist_pose_in_camera = torch.from_numpy(wrist_pose_in_camera)
    else:
        is_numpy = False

    D = hand_points.shape[-1]
    # Get left and right hand points
    hand_points_left = hand_points[..., :D//2]
    hand_points_right = hand_points[..., D//2:]
    
    # Get wrist pose homogeneous matrix in camera frame, i.e. T_wrist2camera
    homo_wrist_pose_left, homo_wrist_pose_right = homo_matrix_from_wrist_pose(wrist_pose_in_camera)
    
    # Transform hand points from wrist frame to camera frame
    hand_points_left = transform_hand_points_to_target_frame(hand_points_left, homo_wrist_pose_left)
    hand_points_right = transform_hand_points_to_target_frame(hand_points_right, homo_wrist_pose_right)
    hand_points = torch.cat([hand_points_left, hand_points_right], dim=-1)

    if is_numpy:
        hand_points = hand_points.numpy()
    return hand_points
