"""Pure Pursuit guidance law for red pursuers.

This module implements a simple proportional-derivative guidance law
that steers the pursuer toward the evader's current position.
"""

import jax.numpy as jnp
from jax import Array

# k_pxy: float = 6.1624,
# k_vxy: float = 3.39,
# k_pz: float = 20.0,
# k_vz: float = 10.0,

def pure_pursuit(
    pos_rb: Array,
    vel_rb: Array,
    k_pxy: float = 2.1624,
    k_vxy: float = 0.59,
    k_pz: float = 10.0,
    k_vz: float = 5.0,
    gravity: float = 9.81,
) -> Array:
    """Compute pure pursuit acceleration command.

    Implements a PD controller in position/velocity space that generates
    acceleration commands to pursue the target.

    Args:
        pos_rb: Position of blue (target) relative to red (pursuer).
                Shape: (..., 3), computed as (target_pos - pursuer_pos).
        vel_rb: Velocity of blue relative to red.
                Shape: (..., 3), computed as (target_vel - pursuer_vel).
        k_pxy: Position gain for x/y axes.
        k_vxy: Velocity gain for x/y axes.
        k_pz: Position gain for z axis.
        k_vz: Velocity gain for z axis.
        gravity: Gravity magnitude [m/s^2].

    Returns:
        Desired acceleration in world frame, shape (..., 3).
        Includes gravity compensation in z-axis.
    """
    fx = k_pxy * pos_rb[..., 0] + k_vxy * vel_rb[..., 0]
    fy = k_pxy * pos_rb[..., 1] + k_vxy * vel_rb[..., 1]
    fz = k_pz * pos_rb[..., 2] + k_vz * vel_rb[..., 2] + gravity

    return jnp.stack([fx, fy, fz], axis=-1)
