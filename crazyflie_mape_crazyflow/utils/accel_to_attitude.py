"""Convert desired acceleration to attitude commands.

This module implements a Mellinger-style conversion from desired acceleration
vectors to roll/pitch/yaw/thrust attitude commands, following the same approach
as drone-controllers state2attitude.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from crazyflie_mape_crazyflow.utils.transformations import (
    quat_to_matrix,
    matrix_to_euler,
)

if TYPE_CHECKING:
    from typing import Any
    Array = Any


def accel_to_attitude(
    accel: Array,
    current_quat: Array,
    mass: float,
    desired_yaw: float = 0.0,
) -> tuple[Array, Array]:
    """Convert desired acceleration to attitude commands.

    Uses the same geometric approach as the Mellinger controller to compute
    the attitude that would produce the desired force direction.

    Args:
        accel: Desired acceleration in world frame, shape (..., 3).
               Should include gravity compensation.
        current_quat: Current orientation as xyzw quaternion, shape (..., 4).
                      Used for thrust projection onto current z-axis.
        mass: Vehicle mass [kg].
        desired_yaw: Desired yaw angle [rad]. Defaults to 0.

    Returns:
        rpy_des: Desired roll, pitch, yaw angles, shape (..., 3).
        thrust_des: Desired thrust [N], shape (...).
    """
    xp = array_namespace(accel)

    # Desired force (target_thrust in Mellinger)
    target_thrust = accel * mass

    # Current body z-axis from quaternion
    rot = quat_to_matrix(current_quat)
    z_axis = rot[..., :, 2]  # 3rd column of rotation matrix is z axis

    # Current thrust: projection of target force onto current z-axis
    current_thrust = xp.sum(target_thrust * z_axis, axis=-1)

    # Desired body z-axis (normalized force direction)
    force_norm = xp.linalg.vector_norm(target_thrust, axis=-1, keepdims=True)
    z_axis_desired = target_thrust / force_norm

    # Desired x_c from yaw angle
    x_c_des_x = xp.cos(xp.asarray(desired_yaw))
    x_c_des_y = xp.sin(xp.asarray(desired_yaw))
    x_c_des_z = xp.zeros_like(x_c_des_x)
    x_c_des = xp.stack((x_c_des_x, x_c_des_y, x_c_des_z), axis=-1)

    # Broadcast x_c_des to match z_axis_desired shape if needed
    if z_axis_desired.ndim > 1:
        x_c_des = xp.broadcast_to(x_c_des, z_axis_desired.shape)

    # yB_des = zB_des x xC_des (normalized)
    y_axis_desired = xp.linalg.cross(z_axis_desired, x_c_des)
    y_axis_desired = y_axis_desired / xp.linalg.vector_norm(y_axis_desired, axis=-1, keepdims=True)

    # xB_des = yB_des x zB_des
    x_axis_desired = xp.linalg.cross(y_axis_desired, z_axis_desired)

    # Build rotation matrix and convert to RPY
    matrix = xp.stack((x_axis_desired, y_axis_desired, z_axis_desired), axis=-1)
    command_rpy = matrix_to_euler(matrix)

    return command_rpy, current_thrust
