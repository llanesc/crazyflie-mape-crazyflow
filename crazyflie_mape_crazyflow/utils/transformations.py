"""Array API compatible rotation transformations.

This module provides rotation utilities that work with numpy, JAX, and PyTorch
using the Python Array API standard. No scipy dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

if TYPE_CHECKING:
    from typing import Any
    Array = Any


def quat_to_matrix(quat: Array) -> Array:
    """Convert quaternion to rotation matrix.

    Args:
        quat: Quaternion in xyzw format with shape (..., 4).

    Returns:
        Rotation matrix with shape (..., 3, 3).
    """
    xp = array_namespace(quat)

    # Normalize quaternion
    norm = xp.linalg.vector_norm(quat, axis=-1, keepdims=True)
    quat = quat / norm

    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]

    # Precompute products
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    xz = x * z
    xw = x * w
    yz = y * z
    yw = y * w
    zw = z * w

    # Build rotation matrix
    r00 = x2 - y2 - z2 + w2
    r01 = 2.0 * (xy - zw)
    r02 = 2.0 * (xz + yw)

    r10 = 2.0 * (xy + zw)
    r11 = -x2 + y2 - z2 + w2
    r12 = 2.0 * (yz - xw)

    r20 = 2.0 * (xz - yw)
    r21 = 2.0 * (yz + xw)
    r22 = -x2 - y2 + z2 + w2

    # Stack into matrix
    row0 = xp.stack([r00, r01, r02], axis=-1)
    row1 = xp.stack([r10, r11, r12], axis=-1)
    row2 = xp.stack([r20, r21, r22], axis=-1)

    return xp.stack([row0, row1, row2], axis=-2)


def euler_to_matrix(rpy: Array) -> Array:
    """Convert euler angles (roll, pitch, yaw) to rotation matrix.

    Uses XYZ (extrinsic) convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Args:
        rpy: Roll, pitch, yaw angles in radians with shape (..., 3).

    Returns:
        Rotation matrix with shape (..., 3, 3).
    """
    xp = array_namespace(rpy)

    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]

    cr = xp.cos(roll)
    sr = xp.sin(roll)
    cp = xp.cos(pitch)
    sp = xp.sin(pitch)
    cy = xp.cos(yaw)
    sy = xp.sin(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    row0 = xp.stack([r00, r01, r02], axis=-1)
    row1 = xp.stack([r10, r11, r12], axis=-1)
    row2 = xp.stack([r20, r21, r22], axis=-1)

    return xp.stack([row0, row1, row2], axis=-2)


def matrix_to_euler(matrix: Array) -> Array:
    """Convert rotation matrix to euler angles (roll, pitch, yaw).

    Uses XYZ (extrinsic) convention.

    Args:
        matrix: Rotation matrix with shape (..., 3, 3).

    Returns:
        Roll, pitch, yaw angles in radians with shape (..., 3).
    """
    xp = array_namespace(matrix)

    # Extract elements
    r20 = matrix[..., 2, 0]
    r21 = matrix[..., 2, 1]
    r22 = matrix[..., 2, 2]
    r10 = matrix[..., 1, 0]
    r00 = matrix[..., 0, 0]

    # Compute pitch (handle gimbal lock)
    pitch = xp.asin(xp.clip(-r20, -1.0, 1.0))

    # Compute roll and yaw
    roll = xp.atan2(r21, r22)
    yaw = xp.atan2(r10, r00)

    return xp.stack([roll, pitch, yaw], axis=-1)


def quat_to_euler(quat: Array) -> Array:
    """Convert quaternion to euler angles (roll, pitch, yaw).

    Args:
        quat: Quaternion in xyzw format with shape (..., 4).

    Returns:
        Roll, pitch, yaw angles in radians with shape (..., 3).
    """
    matrix = quat_to_matrix(quat)
    return matrix_to_euler(matrix)


def euler_to_quat(rpy: Array) -> Array:
    """Convert euler angles (roll, pitch, yaw) to quaternion.

    Args:
        rpy: Roll, pitch, yaw angles in radians with shape (..., 3).

    Returns:
        Quaternion in xyzw format with shape (..., 4).
    """
    xp = array_namespace(rpy)

    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]

    # Half angles
    cr = xp.cos(roll * 0.5)
    sr = xp.sin(roll * 0.5)
    cp = xp.cos(pitch * 0.5)
    sp = xp.sin(pitch * 0.5)
    cy = xp.cos(yaw * 0.5)
    sy = xp.sin(yaw * 0.5)

    # Quaternion components (xyzw format)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return xp.stack([x, y, z, w], axis=-1)


def quat_multiply(q1: Array, q2: Array) -> Array:
    """Multiply two quaternions.

    Args:
        q1: First quaternion in xyzw format with shape (..., 4).
        q2: Second quaternion in xyzw format with shape (..., 4).

    Returns:
        Product quaternion in xyzw format with shape (..., 4).
    """
    xp = array_namespace(q1)

    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return xp.stack([x, y, z, w], axis=-1)


def quat_inverse(quat: Array) -> Array:
    """Compute quaternion inverse (conjugate for unit quaternions).

    Args:
        quat: Quaternion in xyzw format with shape (..., 4).

    Returns:
        Inverse quaternion in xyzw format with shape (..., 4).
    """
    xp = array_namespace(quat)

    # For unit quaternions, inverse is conjugate
    x = -quat[..., 0]
    y = -quat[..., 1]
    z = -quat[..., 2]
    w = quat[..., 3]

    return xp.stack([x, y, z, w], axis=-1)


def rotate_vector(quat: Array, vec: Array) -> Array:
    """Rotate a vector by a quaternion.

    Args:
        quat: Quaternion in xyzw format with shape (..., 4).
        vec: Vector with shape (..., 3).

    Returns:
        Rotated vector with shape (..., 3).
    """
    matrix = quat_to_matrix(quat)
    xp = array_namespace(vec)
    # matrix @ vec
    return xp.squeeze(xp.matmul(matrix, xp.expand_dims(vec, axis=-1)), axis=-1)
