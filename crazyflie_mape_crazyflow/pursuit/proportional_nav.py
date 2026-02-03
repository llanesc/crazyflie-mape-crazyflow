"""Proportional Navigation guidance law for red pursuers.

This module implements Augmented Proportional Navigation (APN) which
accounts for target maneuvers through feedforward acceleration terms.
Falls back to pure pursuit when velocity closure is insufficient.
"""

import jax.numpy as jnp
from jax import Array

from .pure_pursuit import pure_pursuit

def pronav_with_axial(
    pos_rb: Array,
    vel_rb: Array,
    vel_pursuer: Array,
    N_gain: float = 5.0,
    V_min: float = 0.5,
    K_v: float = 2.5,
    gravity: float = 9.81,
) -> Array:
    """Compute augmented proportional navigation acceleration with minimum speed floor.

    This implementation uses APN with lateral steering and axial speed floor:
        a_lat = N * Vc * (omega x u_r) + (N * a_target_orthogonal / 2)
        a_axial = max(0, K_v * (V_min - |vel_pursuer|)) * vel_pursuer_unit
        a_cmd = a_lat + a_axial + gravity

    The axial component only accelerates when speed drops below V_min (no upper bound).
    Falls back to pure pursuit when velocity closure is below threshold.

    Args:
        pos_rb: Position of blue (target) relative to red (pursuer).
                Shape: (..., 3), computed as (target_pos - pursuer_pos).
        vel_rb: Velocity of blue relative to red.
                Shape: (..., 3), computed as (target_vel - pursuer_vel).
        vel_pursuer: Velocity of the pursuer in world frame.
                     Shape: (..., 3).
        N_gain: Navigation constant (typically 3-5).
        V_min: Minimum speed threshold [m/s]. Only accelerates if below this.
        K_v: Speed floor gain.
        gravity: Gravity magnitude [m/s^2].

    Returns:
        Desired acceleration in world frame, shape (..., 3).
        Includes gravity compensation in z-axis.
    """
    # 1. Standard 3D APN Math
    dist = jnp.linalg.norm(pos_rb, axis=-1, keepdims=True) + 1e-6
    u_r = pos_rb / dist  # LOS unit vector
    omega = jnp.cross(pos_rb, vel_rb, axis=-1) / (dist ** 2)

    # Lateral (Steering)
    a_lat = N_gain * jnp.linalg.norm(vel_rb) * jnp.cross(omega, u_r, axis=-1)

    # Axial (Speed Floor)
    # Only accelerate if we drop below the minimum threshold (no upper bound)
    pursuer_speed = jnp.linalg.norm(vel_pursuer, axis=-1, keepdims=True) + 1e-6
    vel_pursuer_unit = vel_pursuer / pursuer_speed
    a_axial_mag = jnp.maximum(0.0, K_v * (V_min - pursuer_speed))
    a_axial = a_axial_mag * u_r

    # 4. AugProNav acceleration with gravity compensation
    accel_pronav = a_lat + a_axial
    accel_pronav = accel_pronav.at[..., 2].add(gravity)

    return accel_pronav

def augmented_pronav(
    pos_rb: Array,
    vel_rb: Array,
    vel_pursuer: Array,
    accel_target: Array,
    N_gain: float = 5.0,
    V_min: float = 0.5,
    K_v: float = 2.5,
    velocity_closure_threshold: float = 0.5,
    gravity: float = 9.81,
) -> Array:
    """Compute augmented proportional navigation acceleration with minimum speed floor.

    This implementation uses APN with lateral steering and axial speed floor:
        a_lat = N * Vc * (omega x u_r) + (N * a_target_orthogonal / 2)
        a_axial = max(0, K_v * (V_min - |vel_pursuer|)) * vel_pursuer_unit
        a_cmd = a_lat + a_axial + gravity

    The axial component only accelerates when speed drops below V_min (no upper bound).
    Falls back to pure pursuit when velocity closure is below threshold.

    Args:
        pos_rb: Position of blue (target) relative to red (pursuer).
                Shape: (..., 3), computed as (target_pos - pursuer_pos).
        vel_rb: Velocity of blue relative to red.
                Shape: (..., 3), computed as (target_vel - pursuer_vel).
        vel_pursuer: Velocity of the pursuer in world frame.
                     Shape: (..., 3).
        accel_target: Target (blue) acceleration estimate.
                      Shape: (..., 3).
        N_gain: Navigation constant (typically 3-5).
        V_min: Minimum speed threshold [m/s]. Only accelerates if below this.
        K_v: Speed floor gain.
        velocity_closure_threshold: Minimum velocity closure to use AugProNav.
                                   Falls back to pure pursuit below this.
        gravity: Gravity magnitude [m/s^2].

    Returns:
        Desired acceleration in world frame, shape (..., 3).
        Includes gravity compensation in z-axis.
    """
    # 1. Standard 3D APN Math
    dist = jnp.linalg.norm(pos_rb, axis=-1, keepdims=True) + 1e-6
    u_r = pos_rb / dist  # LOS unit vector
    omega = jnp.cross(pos_rb, vel_rb, axis=-1) / (dist ** 2)

    # 2. Compute Vc for the APN gain
    Vc = -jnp.sum(u_r * vel_rb, axis=-1, keepdims=True)

    # 3. Calculate the two components of acceleration
    # Orthogonal component of target acceleration
    # Project target accel onto LOS, subtract to get orthogonal
    accel_proj = jnp.sum(accel_target * u_r, axis=-1, keepdims=True) * u_r
    accel_orthogonal = accel_target - accel_proj

    # Lateral (Steering)
    a_lat = N_gain * Vc * jnp.cross(omega, u_r, axis=-1) + (N_gain * accel_orthogonal / 2.0)

    # Axial (Speed Floor)
    # Only accelerate if we drop below the minimum threshold (no upper bound)
    pursuer_speed = jnp.linalg.norm(vel_pursuer, axis=-1, keepdims=True) + 1e-6
    vel_pursuer_unit = vel_pursuer / pursuer_speed
    a_axial_mag = jnp.maximum(0.0, K_v * (V_min - pursuer_speed))
    a_axial = a_axial_mag * vel_pursuer_unit

    # 4. AugProNav acceleration with gravity compensation
    accel_pronav = a_lat + a_axial
    accel_pronav = accel_pronav.at[..., 2].add(gravity)

    # Fall back to pure pursuit when closure is insufficient
    velocity_closure = Vc.squeeze(-1)  # Remove keepdims for comparison
    use_pure_pursuit = velocity_closure < velocity_closure_threshold

    accel_pp = pure_pursuit(pos_rb, vel_rb, gravity=gravity)

    # Select between AugProNav and pure pursuit
    accel_cmd = jnp.where(use_pure_pursuit[..., None], accel_pp, accel_pronav)

    return accel_cmd


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
