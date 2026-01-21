"""Proportional Navigation guidance law for red pursuers.

This module implements Augmented Proportional Navigation (APN) which
accounts for target maneuvers through feedforward acceleration terms.
Falls back to pure pursuit when velocity closure is insufficient.
"""

import jax.numpy as jnp
from jax import Array

from .pure_pursuit import pure_pursuit


def proportional_nav(
    pos_rb: Array,
    vel_rb: Array,
    accel_target: Array,
    N_fb: float = 5.0,
    N_ff: float = 1.0,
    velocity_closure_threshold: float = 0.5,
    gravity: float = 9.81,
) -> Array:
    """Compute augmented proportional navigation acceleration.

    ProNav steers the pursuer to intercept the target by nulling the
    line-of-sight rate. The augmented version adds feedforward terms
    to account for target acceleration.

    Falls back to pure pursuit when velocity closure is below threshold.

    Args:
        pos_rb: Position of blue (target) relative to red (pursuer).
                Shape: (..., 3), computed as (target_pos - pursuer_pos).
        vel_rb: Velocity of blue relative to red.
                Shape: (..., 3), computed as (target_vel - pursuer_vel).
        accel_target: Target (blue) acceleration estimate.
                      Shape: (..., 3).
        N_fb: Feedback navigation constant (typically 3-5).
        N_ff: Feedforward navigation constant for target acceleration.
        velocity_closure_threshold: Minimum velocity closure to use ProNav.
                                   Falls back to pure pursuit below this.
        gravity: Gravity magnitude [m/s^2].

    Returns:
        Desired acceleration in world frame, shape (..., 3).
        Includes gravity compensation in z-axis.
    """
    # Range and direction to target
    range_rb = jnp.linalg.norm(pos_rb, axis=-1, keepdims=True) + 1e-6
    direction_rb = pos_rb / range_rb

    # Line-of-sight rate (omega_LOS = r x v / |r|^2)
    omega_los = jnp.cross(pos_rb, vel_rb, axis=-1) / (range_rb ** 2 + 1e-6)

    # Orthogonal component of target acceleration
    # Project target accel onto LOS, subtract to get orthogonal
    accel_proj = jnp.sum(accel_target * direction_rb, axis=-1, keepdims=True) * direction_rb
    accel_orthogonal = accel_target - accel_proj

    # ProNav acceleration: N * V x omega_LOS + Nff * a_target_orthogonal
    accel_pronav = N_fb * jnp.cross(vel_rb, omega_los, axis=-1) + N_ff * accel_orthogonal

    # Check velocity closure (positive when closing)
    # velocity_closure = -v_rb . direction_rb (closing velocity)
    velocity_closure = -jnp.sum(vel_rb * direction_rb, axis=-1)

    # Fall back to pure pursuit when closure is insufficient
    use_pure_pursuit = velocity_closure < velocity_closure_threshold

    accel_pp = pure_pursuit(pos_rb, vel_rb, gravity=gravity)

    # Select between ProNav and pure pursuit
    # Need to add gravity to pronav (pure_pursuit already includes it)
    accel_pronav = accel_pronav.at[..., 2].add(gravity)

    accel = jnp.where(use_pure_pursuit[..., None], accel_pp, accel_pronav)

    return accel
