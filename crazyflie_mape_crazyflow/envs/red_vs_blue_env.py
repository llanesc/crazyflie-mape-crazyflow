"""Red vs Blue multi-agent pursuit-evasion environment using Crazyflow.

Blue agents (evaders) are controlled by learned policies.
Red agents (pursuers) use scripted control (ProNav or Pure Pursuit).
"""

import time
from functools import partial
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax.scipy.spatial.transform import Rotation as JaxRotation

from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.physics import Physics
from crazyflow.control.control import Control
from crazyflow.randomize import randomize_mass, randomize_inertia
from crazyflow.utils import leaf_replace
from drone_models.core import load_params

from crazyflie_mape_crazyflow.envs.red_vs_blue_config import RedVsBlueEnvConfig
from crazyflie_mape_crazyflow.envs.spawn import SpawnFn, create_deterministic_spawn_fn


# ============================================================================
# JIT-compiled helper functions for performance
# ============================================================================

@partial(jax.jit, static_argnames=["N", "B", "R"])
def _jit_random_target_assignment(
    key: jax.Array,
    N: int,
    B: int,
    R: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate random target assignments for red agents.

    Each red agent is assigned a random blue target. When R == B,
    this creates a random permutation (1-to-1 mapping). When R != B,
    targets are sampled uniformly.

    Args:
        key: JAX random key.
        N: Number of worlds.
        B: Number of blue agents.
        R: Number of red agents.

    Returns:
        Tuple of (red_target, red_target_one_hot):
        - red_target: (N, R) array of target indices
        - red_target_one_hot: (N, R, B) one-hot encoding
    """
    if R == B:
        # Generate random permutations for each world
        keys = jax.random.split(key, N)
        red_target = jax.vmap(lambda k: jax.random.permutation(k, B))(keys)
    else:
        # Random assignment with possible duplicates
        red_target = jax.random.randint(key, (N, R), 0, B)

    # Create one-hot encoding
    red_target_one_hot = jax.nn.one_hot(red_target, B)

    return red_target, red_target_one_hot


@partial(jax.jit, static_argnames=["N", "B", "R"])
def _jit_deterministic_target_assignment(
    N: int,
    B: int,
    R: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate deterministic target assignments for red agents.

    Red agent i targets blue agent i % B.

    Args:
        N: Number of worlds.
        B: Number of blue agents.
        R: Number of red agents.

    Returns:
        Tuple of (red_target, red_target_one_hot):
        - red_target: (N, R) array of target indices
        - red_target_one_hot: (N, R, B) one-hot encoding
    """
    # Red i targets blue i % B
    red_target = jnp.arange(R).reshape(1, R).repeat(N, axis=0) % B
    red_target_one_hot = jax.nn.one_hot(red_target, B)
    return red_target, red_target_one_hot


@jax.jit
def _jit_check_collisions(
    blue_pos: jnp.ndarray,
    red_pos: jnp.ndarray,
    blue_alive: jnp.ndarray,
    red_alive: jnp.ndarray,
    bb_tol: float,
    rr_tol: float,
    br_tol: float,
    min_alt: float,
    max_alt: float,
    boundary: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled collision detection."""
    B = blue_pos.shape[1]
    R = red_pos.shape[1]

    # Blue-blue collisions (vectorized)
    blue_pos_i = blue_pos[:, :, None, :]
    blue_pos_j = blue_pos[:, None, :, :]
    bb_dist = jnp.linalg.norm(blue_pos_i - blue_pos_j, axis=-1)
    alive_mask_bb = blue_alive[:, :, None] & blue_alive[:, None, :]
    bb_collision_matrix = (bb_dist < bb_tol) & alive_mask_bb
    bb_collision_matrix = bb_collision_matrix & ~jnp.eye(B, dtype=jnp.bool_)[None, :, :]
    bb_collision = bb_collision_matrix.any(axis=-1)

    # Red-red collisions (vectorized)
    red_pos_i = red_pos[:, :, None, :]
    red_pos_j = red_pos[:, None, :, :]
    rr_dist = jnp.linalg.norm(red_pos_i - red_pos_j, axis=-1)
    alive_mask_rr = red_alive[:, :, None] & red_alive[:, None, :]
    rr_collision_matrix = (rr_dist < rr_tol) & alive_mask_rr
    rr_collision_matrix = rr_collision_matrix & ~jnp.eye(R, dtype=jnp.bool_)[None, :, :]
    rr_collision = rr_collision_matrix.any(axis=-1)

    # Blue-red collisions / captures (vectorized)
    br_dist = jnp.linalg.norm(blue_pos[:, :, None, :] - red_pos[:, None, :, :], axis=-1)
    alive_mask_br = blue_alive[:, :, None] & red_alive[:, None, :]
    br_collision_matrix = (br_dist < br_tol) & alive_mask_br
    rb_collision = br_collision_matrix.any(axis=-1)

    # Boundary violations (blue only)
    out_of_bounds = (
        (blue_pos[:, :, 2] < min_alt) |
        (blue_pos[:, :, 2] > max_alt) |
        (jnp.abs(blue_pos[:, :, 0]) > boundary) |
        (jnp.abs(blue_pos[:, :, 1]) > boundary)
    ) & blue_alive

    return bb_collision, rr_collision, rb_collision, out_of_bounds


@jax.jit
def _jit_update_alive_and_reassign(
    blue_alive: jnp.ndarray,
    red_alive: jnp.ndarray,
    red_target: jnp.ndarray,
    bb_collision: jnp.ndarray,
    rr_collision: jnp.ndarray,
    rb_collision: jnp.ndarray,
    out_of_bounds: jnp.ndarray,
    blue_pos: jnp.ndarray,
    red_pos: jnp.ndarray,
    br_tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled alive status update and target reassignment."""
    B = blue_pos.shape[1]

    # Mark dead agents
    blue_dead = bb_collision | rb_collision | out_of_bounds
    red_dead = rr_collision

    # Also mark red as dead if they captured a blue (vectorized)
    rb_dist = jnp.linalg.norm(red_pos[:, :, None, :] - blue_pos[:, None, :, :], axis=-1)
    alive_mask_rb = red_alive[:, :, None] & blue_alive[:, None, :]
    rb_collision_matrix = (rb_dist < br_tol) & alive_mask_rb
    red_captured = rb_collision_matrix.any(axis=-1)
    red_dead = red_dead | red_captured

    # Update alive status
    new_blue_alive = blue_alive & ~blue_dead
    new_red_alive = red_alive & ~red_dead

    # Reassign red targets (vectorized)
    current_targets = red_target.astype(jnp.int32)
    target_died = jnp.take_along_axis(blue_dead, current_targets, axis=1)
    needs_reassignment = target_died & new_red_alive

    # Compute distances from each red to each blue
    rb_dist_for_reassign = jnp.linalg.norm(red_pos[:, :, None, :] - blue_pos[:, None, :, :], axis=-1)
    rb_dist_masked = jnp.where(new_blue_alive[:, None, :], rb_dist_for_reassign, jnp.inf)
    new_targets = jnp.argmin(rb_dist_masked, axis=-1)

    # Only update targets that need reassignment
    new_red_target = jnp.where(needs_reassignment, new_targets, red_target)
    new_one_hot = jax.nn.one_hot(new_red_target.astype(jnp.int32), B)

    return new_blue_alive, new_red_alive, new_red_target, new_one_hot


@jax.jit
def _jit_compute_rewards(
    bb_collision: jnp.ndarray,
    rr_collision: jnp.ndarray,
    rb_collision: jnp.ndarray,
    out_of_bounds: jnp.ndarray,
    blue_alive: jnp.ndarray,
    red_pos: jnp.ndarray,
    blue_rpy: jnp.ndarray,
    blue_vel: jnp.ndarray,
    blue_cmd: jnp.ndarray,
    last_blue_cmd: jnp.ndarray,
    reward_blue_crash: float,
    reward_red_crash: float,
    reward_capture: float,
    reward_boundary: float,
    reward_alive: float,
    reward_pursuer_proximity: float,
    reward_pursuer_proximity_decay: float,
    reward_angle_coef: float,
    reward_velocity_coef: float,
    reward_action_coef: float,
    reward_action_smoothness_thrust: float,
    reward_action_smoothness_rpy: float,
    n_pairs: int,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """JIT-compiled reward computation.

    Returns:
        Tuple of (total_reward, components) where components is a dict of individual
        reward contributions (already normalized by n_pairs).
    """
    reward_bb = (bb_collision.astype(jnp.float32) * blue_alive.astype(jnp.float32)).sum(axis=1) * reward_blue_crash / 2
    reward_rr = rr_collision.astype(jnp.float32).sum(axis=1) * reward_red_crash / 2
    reward_br = rb_collision.astype(jnp.float32).sum(axis=1) * reward_capture
    reward_fence = (out_of_bounds.astype(jnp.float32) * blue_alive.astype(jnp.float32)).sum(axis=1) * reward_boundary
    reward_alive_total = blue_alive.astype(jnp.float32).sum(axis=1) * reward_alive
    pursuer_dist = jnp.linalg.norm(red_pos[:, 0] - red_pos[:, 1], axis=-1)
    reward_proximity = reward_pursuer_proximity * jnp.exp(-reward_pursuer_proximity_decay * pursuer_dist)

    # Angle penalty: penalizes orientation deviation from level flight
    # Sum ||rpy|| across all blue agents, weighted by alive status
    rpy_norm = jnp.linalg.norm(blue_rpy, axis=-1)  # (N, B)
    angle_penalty = reward_angle_coef * (rpy_norm * blue_alive.astype(jnp.float32)).sum(axis=1)

    # Velocity penalty: penalizes high speeds
    # Sum ||vel||² across all blue agents, weighted by alive status
    vel_sq = (blue_vel ** 2).sum(axis=-1)  # (N, B) - squared velocity magnitude
    velocity_penalty = reward_velocity_coef * (vel_sq * blue_alive.astype(jnp.float32)).sum(axis=1)

    # Action penalties
    # Energy penalty: act_coef * thrust² (thrust is the 4th action component)
    thrust = blue_cmd[..., 3]  # (N, B)
    energy_penalty = reward_action_coef * (thrust ** 2 * blue_alive.astype(jnp.float32)).sum(axis=1)

    # Action smoothness penalties
    action_diff = blue_cmd - last_blue_cmd  # (N, B, 4)
    # Thrust smoothness: d_act_th_coef * (Δthrust)²
    thrust_diff = action_diff[..., 3]  # (N, B)
    thrust_smoothness_penalty = reward_action_smoothness_thrust * (thrust_diff ** 2 * blue_alive.astype(jnp.float32)).sum(axis=1)
    # RPY smoothness: d_act_xy_coef * Σ(Δroll² + Δpitch² + Δyaw²)
    rpy_diff = action_diff[..., :3]  # (N, B, 3)
    rpy_smoothness_penalty = reward_action_smoothness_rpy * ((rpy_diff ** 2).sum(axis=-1) * blue_alive.astype(jnp.float32)).sum(axis=1)

    total_reward = (reward_bb + reward_rr + reward_br + reward_fence + reward_alive_total + reward_proximity
                    - angle_penalty - velocity_penalty - energy_penalty - thrust_smoothness_penalty - rpy_smoothness_penalty) / n_pairs

    # Return individual components (normalized by n_pairs to match total)
    components = {
        "bb_collision": reward_bb / n_pairs,
        "rr_collision": reward_rr / n_pairs,
        "rb_collision": reward_br / n_pairs,
        "boundary": reward_fence / n_pairs,
        "alive": reward_alive_total / n_pairs,
        "proximity": reward_proximity / n_pairs,
        "angle_penalty": -angle_penalty / n_pairs,
        "velocity_penalty": -velocity_penalty / n_pairs,
        "energy_penalty": -energy_penalty / n_pairs,
        "thrust_smoothness": -thrust_smoothness_penalty / n_pairs,
        "rpy_smoothness": -rpy_smoothness_penalty / n_pairs,
    }

    return total_reward, components


@jax.jit
def _jit_quat_to_rpy(quat: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled quaternion to RPY conversion using JAX scipy Rotation.

    Uses jax.scipy.spatial.transform.Rotation for consistency with drone-models library.

    Args:
        quat: Quaternion array in xyzw order, shape (..., 4).

    Returns:
        RPY angles in radians, shape (..., 3).
    """
    return JaxRotation.from_quat(quat).as_euler('xyz')


@jax.jit
def _jit_ang_vel_to_rpy_rates(quat: jnp.ndarray, ang_vel: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled body angular velocity to Euler rates conversion.

    Converts body-frame angular velocity [p, q, r] to Euler angle rates [droll, dpitch, dyaw].
    Based on drone_models.utils.rotation.ang_vel2rpy_rates, using JAX scipy Rotation.

    The transformation matrix W satisfies: rpy_rates = W @ ang_vel
    where W depends on the current roll (phi) and pitch (theta) angles.

    Args:
        quat: Quaternion array in xyzw order, shape (..., 4).
        ang_vel: Body angular velocity [p, q, r], shape (..., 3).

    Returns:
        Euler angle rates [droll, dpitch, dyaw], shape (..., 3).
    """
    rpy = JaxRotation.from_quat(quat).as_euler('xyz')
    phi = rpy[..., 0]    # roll
    theta = rpy[..., 1]  # pitch

    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    cos_theta = jnp.cos(theta)
    tan_theta = jnp.tan(theta)
    inv_cos_theta = 1.0 / (cos_theta + 1e-8)  # Add small epsilon to avoid division by zero

    # Build transformation matrix W
    # W = [[1, sin(phi)*tan(theta), cos(phi)*tan(theta)],
    #      [0, cos(phi),            -sin(phi)],
    #      [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]]
    p = ang_vel[..., 0]
    q = ang_vel[..., 1]
    r = ang_vel[..., 2]

    droll = p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
    dpitch = cos_phi * q - sin_phi * r
    dyaw = sin_phi * inv_cos_theta * q + cos_phi * inv_cos_theta * r

    return jnp.stack([droll, dpitch, dyaw], axis=-1)


@jax.jit
def _jit_quat_to_matrix(quat: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled quaternion to rotation matrix using JAX scipy Rotation.

    Args:
        quat: Quaternion in xyzw format with shape (..., 4).

    Returns:
        Rotation matrix with shape (..., 3, 3).
    """
    return JaxRotation.from_quat(quat).as_matrix()


@jax.jit
def _jit_matrix_to_euler(matrix: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled rotation matrix to euler angles using JAX scipy Rotation.

    Args:
        matrix: Rotation matrix with shape (..., 3, 3).

    Returns:
        RPY angles in radians with shape (..., 3).
    """
    return JaxRotation.from_matrix(matrix).as_euler('xyz')


def _jit_accel_to_attitude(
    accel: jnp.ndarray,
    current_quat: jnp.ndarray,
    mass: float,
    desired_yaw: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-friendly accel to attitude conversion."""
    target_thrust = accel * mass

    rot = _jit_quat_to_matrix(current_quat)
    z_axis = rot[..., :, 2]

    current_thrust = jnp.sum(target_thrust * z_axis, axis=-1)

    force_norm = jnp.linalg.norm(target_thrust, axis=-1, keepdims=True) + 1e-6
    z_axis_desired = target_thrust / force_norm

    x_c_des = jnp.array([jnp.cos(desired_yaw), jnp.sin(desired_yaw), 0.0])
    x_c_des = jnp.broadcast_to(x_c_des, z_axis_desired.shape)

    y_axis_desired = jnp.cross(z_axis_desired, x_c_des)
    y_norm = jnp.linalg.norm(y_axis_desired, axis=-1, keepdims=True) + 1e-6
    y_axis_desired = y_axis_desired / y_norm

    x_axis_desired = jnp.cross(y_axis_desired, z_axis_desired)

    matrix = jnp.stack([x_axis_desired, y_axis_desired, z_axis_desired], axis=-1)
    command_rpy = _jit_matrix_to_euler(matrix)

    return command_rpy, current_thrust


def _jit_pure_pursuit(
    pos_rb: jnp.ndarray,
    vel_rb: jnp.ndarray,
    k_pxy: float = 1.0,
    k_vxy: float = 1.0,
    k_pz: float = 1.0,
    k_vz: float = 1.0,
    gravity: float = 9.81,
) -> jnp.ndarray:
    """JIT-friendly pure pursuit guidance."""
    accel_xy = k_pxy * pos_rb[..., :2] + k_vxy * vel_rb[..., :2]
    accel_z = k_pz * pos_rb[..., 2:3] + k_vz * vel_rb[..., 2:3] + gravity
    return jnp.concatenate([accel_xy, accel_z], axis=-1)


def _jit_proportional_nav(
    pos_rb: jnp.ndarray,
    vel_rb: jnp.ndarray,
    accel_target: jnp.ndarray,
    N_fb: float = 5.0,
    N_ff: float = 1.0,
    velocity_closure_threshold: float = 0.5,
    gravity: float = 9.81,
    k_pxy: float = 1.0,
    k_vxy: float = 1.0,
    k_pz: float = 1.0,
    k_vz: float = 1.0,
) -> jnp.ndarray:
    """JIT-friendly proportional navigation guidance."""
    range_rb = jnp.linalg.norm(pos_rb, axis=-1, keepdims=True) + 1e-6
    direction_rb = pos_rb / range_rb

    omega_los = jnp.cross(pos_rb, vel_rb) / (range_rb ** 2 + 1e-6)

    accel_proj = jnp.sum(accel_target * direction_rb, axis=-1, keepdims=True) * direction_rb
    accel_orthogonal = accel_target - accel_proj

    accel_pronav = N_fb * jnp.cross(vel_rb, omega_los) + N_ff * accel_orthogonal
    accel_pronav = accel_pronav.at[..., 2].add(gravity)

    velocity_closure = -jnp.sum(vel_rb * direction_rb, axis=-1)
    use_pure_pursuit = velocity_closure < velocity_closure_threshold

    accel_pp = _jit_pure_pursuit(pos_rb, vel_rb, k_pxy, k_vxy, k_pz, k_vz, gravity)

    accel = jnp.where(use_pure_pursuit[..., None], accel_pp, accel_pronav)
    return accel


def _jit_augmented_pronav(
    pos_rb: jnp.ndarray,
    vel_rb: jnp.ndarray,
    vel_pursuer: jnp.ndarray,
    accel_target: jnp.ndarray,
    N_gain: float = 3.0,
    V_min: float = 0.5,
    K_v: float = 2.5,
    velocity_closure_threshold: float = 0.5,
    gravity: float = 9.81,
    k_pxy: float = 1.0,
    k_vxy: float = 1.0,
    k_pz: float = 1.0,
    k_vz: float = 1.0,
) -> jnp.ndarray:
    """JIT-friendly augmented proportional navigation with speed floor.

    Uses APN with lateral steering and axial speed floor:
        a_lat = N * Vc * (omega x u_r) + (N * a_target_orthogonal / 2)
        a_axial = max(0, K_v * (V_min - |vel_pursuer|)) * vel_pursuer_unit
    """
    # 1. Standard 3D APN Math
    dist = jnp.linalg.norm(pos_rb, axis=-1, keepdims=True) + 1e-6
    u_r = pos_rb / dist  # LOS unit vector
    omega = jnp.cross(pos_rb, vel_rb) / (dist ** 2)

    # 2. Compute Vc for the APN gain
    Vc = -jnp.sum(u_r * vel_rb, axis=-1, keepdims=True)

    # 3. Orthogonal component of target acceleration
    accel_proj = jnp.sum(accel_target * u_r, axis=-1, keepdims=True) * u_r
    accel_orthogonal = accel_target - accel_proj

    # 4. Lateral (Steering)
    a_lat = N_gain * Vc * jnp.cross(omega, u_r) + (N_gain * accel_orthogonal / 2.0)

    # 5. Axial (Speed Floor) - only accelerate if below V_min
    pursuer_speed = jnp.linalg.norm(vel_pursuer, axis=-1, keepdims=True) + 1e-6
    vel_pursuer_unit = vel_pursuer / pursuer_speed
    a_axial_mag = jnp.maximum(0.0, K_v * (V_min - pursuer_speed))
    a_axial = a_axial_mag * vel_pursuer_unit

    # 6. AugProNav acceleration with gravity compensation
    accel_pronav = a_lat + a_axial
    accel_pronav = accel_pronav.at[..., 2].add(gravity)

    # Fall back to pure pursuit when closure is insufficient
    velocity_closure = Vc.squeeze(-1)
    use_pure_pursuit = velocity_closure < velocity_closure_threshold

    accel_pp = _jit_pure_pursuit(pos_rb, vel_rb, k_pxy, k_vxy, k_pz, k_vz, gravity)

    accel = jnp.where(use_pure_pursuit[..., None], accel_pp, accel_pronav)
    return accel


def _jit_tpn_with_axial(
    pos_rb: jnp.ndarray,
    vel_rb: jnp.ndarray,
    vel_pursuer: jnp.ndarray,
    N_gain: float = 3.0,
    V_min: float = 0.5,
    K_v: float = 2.5,
    gravity: float = 9.81,
) -> jnp.ndarray:
    """JIT-friendly True Proportional Navigation with axial speed floor.

    Uses TPN (||vel_rb|| instead of Vc for stability) with axial speed floor:
        a_lat = N * ||vel_rb|| * (omega x u_r)
        a_axial = max(0, K_v * (V_min - |vel_pursuer|)) * u_r

    Unlike AugProNav:
    - Does NOT fall back to pure pursuit (always uses TPN)
    - Does NOT use target acceleration feedforward
    - Uses ||vel_rb|| instead of Vc for more stable behavior
    """
    # 1. Standard 3D ProNav Math
    dist = jnp.linalg.norm(pos_rb, axis=-1, keepdims=True) + 1e-6
    u_r = pos_rb / dist  # LOS unit vector
    omega = jnp.cross(pos_rb, vel_rb) / (dist ** 2)

    # 2. Lateral (Steering) - uses ||vel_rb|| instead of Vc for TPN
    vel_rb_norm = jnp.linalg.norm(vel_rb, axis=-1, keepdims=True)
    a_lat = N_gain * vel_rb_norm * jnp.cross(omega, u_r)

    # 3. Axial (Speed Floor) - only accelerate if below V_min
    pursuer_speed = jnp.linalg.norm(vel_pursuer, axis=-1, keepdims=True) + 1e-6
    a_axial_mag = jnp.maximum(0.0, K_v * (V_min - pursuer_speed))
    a_axial = a_axial_mag * u_r

    # 4. TPN acceleration with gravity compensation
    accel = a_lat + a_axial
    accel = accel.at[..., 2].add(gravity)

    return accel


@partial(jax.jit, static_argnames=["pursuer_strategy", "mass", "roll_pitch_max", "yaw_max", "min_thrust", "max_thrust",
                                   "pp_k_pxy", "pp_k_vxy", "pp_k_pz", "pp_k_vz", "N_pronav_fb", "N_pronav_ff",
                                   "velocity_closure_threshold", "gravity", "N_gain", "V_min", "K_v"])
def _jit_compute_red_control(
    blue_pos: jnp.ndarray,
    blue_vel: jnp.ndarray,
    red_pos: jnp.ndarray,
    red_vel: jnp.ndarray,
    red_quat: jnp.ndarray,
    red_target: jnp.ndarray,
    red_alive: jnp.ndarray,
    prev_blue_accel: jnp.ndarray,
    pursuer_strategy: str,
    mass: float,
    roll_pitch_max: float,
    yaw_max: float,
    min_thrust: float,
    max_thrust: float,
    pp_k_pxy: float,
    pp_k_vxy: float,
    pp_k_pz: float,
    pp_k_vz: float,
    N_pronav_fb: float,
    N_pronav_ff: float,
    velocity_closure_threshold: float,
    gravity: float,
    N_gain: float,
    V_min: float,
    K_v: float,
) -> jnp.ndarray:
    """JIT-compiled red pursuit control computation."""
    target_idx = red_target

    target_pos = jnp.take_along_axis(blue_pos, target_idx[:, :, None].astype(jnp.int32), axis=1)
    target_vel = jnp.take_along_axis(blue_vel, target_idx[:, :, None].astype(jnp.int32), axis=1)

    pos_rb = target_pos - red_pos
    vel_rb = target_vel - red_vel

    if pursuer_strategy == "PP":
        accel = _jit_pure_pursuit(pos_rb, vel_rb, pp_k_pxy, pp_k_vxy, pp_k_pz, pp_k_vz, gravity)
    elif pursuer_strategy == "AugProNav":
        accel_target = jnp.take_along_axis(prev_blue_accel, target_idx[:, :, None].astype(jnp.int32), axis=1)
        accel = _jit_augmented_pronav(
            pos_rb, vel_rb, red_vel, accel_target,
            N_gain, V_min, K_v, velocity_closure_threshold, gravity,
            pp_k_pxy, pp_k_vxy, pp_k_pz, pp_k_vz
        )
    elif pursuer_strategy == "TPN":
        accel = _jit_tpn_with_axial(
            pos_rb, vel_rb, red_vel,
            N_gain, V_min, K_v, gravity
        )
    else:  # ProNav (default)
        accel_target = jnp.take_along_axis(prev_blue_accel, target_idx[:, :, None].astype(jnp.int32), axis=1)
        accel = _jit_proportional_nav(
            pos_rb, vel_rb, accel_target,
            N_pronav_fb, N_pronav_ff, velocity_closure_threshold, gravity,
            pp_k_pxy, pp_k_vxy, pp_k_pz, pp_k_vz
        )

    rpy_des, thrust_des = _jit_accel_to_attitude(accel, red_quat, mass)

    rpy_des = jnp.clip(
        rpy_des,
        jnp.array([-roll_pitch_max, -roll_pitch_max, -yaw_max]),
        jnp.array([roll_pitch_max, roll_pitch_max, yaw_max])
    )
    thrust_des = jnp.clip(thrust_des, min_thrust, max_thrust)

    red_cmd = jnp.concatenate([rpy_des, thrust_des[..., None]], axis=-1)
    red_cmd = red_cmd * red_alive[:, :, None]

    return red_cmd


class RedVsBlueEnv(gym.Env):
    """Multi-agent Red vs Blue pursuit-evasion environment.

    Implements a multi-agent interface with:
    - Blue agents (evaders): Learned policy, outputs attitude commands
    - Red agents (pursuers): Scripted pursuit (ProNav or Pure Pursuit)

    Architecture:
    - Single Crazyflow Sim with n_drones = n_red + n_blue
    - Physics: first_principles at 500Hz
    - Control: attitude mode (Mellinger internally at 500Hz)
    - MPC/Pursuer control: 100Hz
    - Drone indices: [0:n_blue] = blue, [n_blue:n_drones] = red

    The environment step frequency is determined by control_freq (default 100Hz).
    Each step executes sim_steps_per_control physics steps.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        cfg: RedVsBlueEnvConfig | None = None,
        render_mode: str | None = None,
        spawn_fn: SpawnFn | None = None,
    ):
        """Initialize the environment.

        Args:
            cfg: Environment configuration. Uses defaults if None.
            render_mode: Rendering mode ("human" or "rgb_array").
            spawn_fn: Spawn function with signature (key, N, B, R) -> (blue_pos, red_pos).
                If None, uses deterministic spawn with default parameters.
                See crazyflie_mape_crazyflow.envs.spawn for factory functions.
        """
        super().__init__()
        self.cfg = cfg or RedVsBlueEnvConfig()
        self.render_mode = render_mode
        self.n_pairs = self.cfg.n_pairs

        # Store spawn function (use default deterministic if not provided)
        self._spawn_fn = spawn_fn or create_deterministic_spawn_fn()

        # Initialize simulator
        self._init_simulator()

        # Agent names (only blue agents are learning agents)
        self.possible_agents = [f"blue_{i}" for i in range(self.cfg.n_blue)]
        self.agents = self.possible_agents.copy()
        self.num_agents = len(self.possible_agents)

        # Define spaces
        self._define_spaces()

        # Initialize state tracking
        self._init_state_tensors()

        # Episode tracking (per-world for vectorized env)
        self.episode_steps = np.zeros(self.cfg.n_worlds, dtype=np.int32)
        self._max_episode_steps = self.cfg.max_episode_steps

    def _init_simulator(self):
        """Initialize Crazyflow simulator."""
        self.sim = Sim(
            n_worlds=self.cfg.n_worlds,
            n_drones=self.cfg.n_drones,
            drone_model=self.cfg.drone_model,
            physics=Physics.first_principles,
            control=Control.attitude,
            freq=self.cfg.sim_freq,
            attitude_freq=self.cfg.mellinger_freq,
            device=self.cfg.device,
        )

        # Override mass in drone params
        params = self.sim.data.params
        params = params.replace(mass=params.mass.at[:, :].set(self.cfg.mass))
        self.sim.data = self.sim.data.replace(params=params)

        # Apply domain randomization if configured
        self._apply_domain_randomization()

        # Store base pipeline (without disturbance) for dynamic enable/disable
        self._base_step_pipeline = self.sim.step_pipeline
        self._disturbance_enabled = False

        # Inject disturbance function into simulation pipeline if enabled
        if self.cfg.enable_disturbance:
            self._enable_disturbance()

        # Calculate hover RPM from first_principles model parameters
        fp_params = load_params("first_principles", self.cfg.drone_model)
        rpm2thrust = fp_params["rpm2thrust"]  # [c0, c1, c2] where thrust = c0 + c1*rpm + c2*rpm^2
        thrust_per_motor_hover = self.cfg.mass * self.cfg.gravity / 4
        # Solve quadratic: c2*rpm^2 + c1*rpm + (c0 - thrust) = 0
        a = rpm2thrust[2]
        b = rpm2thrust[1]
        c = rpm2thrust[0] - thrust_per_motor_hover
        self.hover_rpm = float((-b + np.sqrt(b**2 - 4*a*c)) / (2*a))

    def _apply_domain_randomization(self, mask: jnp.ndarray | None = None):
        """Apply domain randomization to mass and/or inertia.

        Args:
            mask: Boolean array of shape (n_worlds,) indicating which worlds to randomize.
                  If None, all worlds are randomized.
        """
        if not self.cfg.randomize_mass and not self.cfg.randomize_inertia:
            return

        if self.cfg.randomize_mass:
            key = jax.random.key(np.random.randint(0, 2**31))
            mass_noise = jax.random.normal(
                key, (self.cfg.n_worlds, self.cfg.n_drones, 1)
            ) * self.cfg.mass_randomization_std
            randomized_masses = self.cfg.mass + mass_noise
            randomize_mass(self.sim, randomized_masses, mask)

        if self.cfg.randomize_inertia:
            key = jax.random.key(np.random.randint(0, 2**31))
            J_noise = jax.random.normal(
                key, (self.cfg.n_worlds, self.cfg.n_drones, 3, 3)
            ) * self.cfg.inertia_randomization_std
            J_randomized = self.sim.data.params.J + J_noise
            randomize_inertia(self.sim, J_randomized, mask)

    def _create_disturbance_fn(self):
        """Create a disturbance function for the simulation pipeline.

        Returns:
            A function that applies random force and torque disturbances to the simulation.
        """
        force_std = self.cfg.disturbance_force_std
        torque_std = self.cfg.disturbance_torque_std

        def disturbance_fn(data: SimData) -> SimData:
            key = data.core.rng_key
            key, force_key, torque_key = jax.random.split(key, 3)

            states = data.states
            disturbance_force = jax.random.normal(force_key, states.force.shape) * force_std
            disturbance_torque = jax.random.normal(torque_key, states.torque.shape) * torque_std

            states = states.replace(
                force=disturbance_force,
                torque=disturbance_torque,
            )
            core = data.core.replace(rng_key=key)
            return data.replace(states=states, core=core)

        return disturbance_fn

    def _enable_disturbance(self):
        """Enable disturbance injection in simulation pipeline."""
        if self._disturbance_enabled:
            # Already enabled, just update the function with new std values
            self._disable_disturbance()

        disturbance_fn = self._create_disturbance_fn()
        self.sim.step_pipeline = (
            self._base_step_pipeline[:2] + (disturbance_fn,) + self._base_step_pipeline[2:]
        )
        self.sim.build_step_fn()
        self._disturbance_enabled = True

    def _disable_disturbance(self):
        """Disable disturbance injection in simulation pipeline."""
        if not self._disturbance_enabled:
            return

        self.sim.step_pipeline = self._base_step_pipeline
        self.sim.build_step_fn()
        self._disturbance_enabled = False

    def _define_spaces(self):
        """Define observation and action spaces."""
        n = self.cfg.n_pairs

        # Per-agent observation dimension:
        # - Own state: pos(3) + vel(3) + rpy(3) + ang_vel(3) = 12
        # - Own one-hot: n_blue
        # - All blue states + alive: n_blue * 7
        # - All red states + alive: n_red * 7
        # - Red target assignments: n_red * n_blue
        self.obs_dim = 12 + n + n * 7 + n * 7 + n * n

        # MPC state dimension for internal use (Euler dynamics)
        self.mpc_state_dim = 12  # [pos(3), rpy(3), vel(3), drpy(3)]

        # Observation space (dict for each agent)
        self.observation_space = spaces.Dict({
            agent: spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        })

        # Action space: physical attitude commands [roll, pitch, yaw, thrust]
        self.action_space = spaces.Dict({
            agent: spaces.Box(
                low=np.array([
                    -self.cfg.roll_pitch_max,
                    -self.cfg.roll_pitch_max,
                    -self.cfg.yaw_max,
                    self.cfg.min_thrust
                ], dtype=np.float32),
                high=np.array([
                    self.cfg.roll_pitch_max,
                    self.cfg.roll_pitch_max,
                    self.cfg.yaw_max,
                    self.cfg.max_thrust
                ], dtype=np.float32),
            )
            for agent in self.possible_agents
        })

        # PettingZoo-style aliases (plural names for spaces)
        self.observation_spaces = self.observation_space
        self.action_spaces = self.action_space

        # Shared state for centralized critic
        # All blue states + all red states + target assignments
        shared_dim = n * 10 + n * 10 + n * n
        self.shared_observation_space = spaces.Box(
            -np.inf, np.inf, (shared_dim,), dtype=np.float32
        )

        # State space for PettingZoo wrapper (same shared state for all agents)
        self.state_spaces = {
            agent: self.shared_observation_space
            for agent in self.possible_agents
        }

    def _init_state_tensors(self):
        """Initialize state tracking tensors."""
        N, B, R = self.cfg.n_worlds, self.cfg.n_blue, self.cfg.n_red

        # Alive status
        self.blue_alive = jnp.ones((N, B), dtype=jnp.bool_)
        self.red_alive = jnp.ones((N, R), dtype=jnp.bool_)

        # Target assignments: which blue each red is pursuing
        if self.cfg.random_target_assignment:
            key = self.sim.data.core.rng_key
            key, target_key = jax.random.split(key)
            self.sim.data = self.sim.data.replace(
                core=self.sim.data.core.replace(rng_key=key)
            )
            self.red_target, self.red_target_one_hot = _jit_random_target_assignment(
                target_key, N, B, R
            )
        else:
            self.red_target, self.red_target_one_hot = _jit_deterministic_target_assignment(
                N, B, R
            )

        # Ally one-hot for observations
        self.ally_one_hot = jnp.eye(B)[None, :, :].repeat(N, axis=0)

        # Pending blue commands
        self.blue_cmd = jnp.zeros((N, B, 4))

        # Previous blue commands (for action smoothness penalty)
        self.last_blue_cmd = jnp.zeros((N, B, 4))

        # Previous blue velocity and acceleration (for ProNav feedforward)
        self.prev_blue_vel = jnp.zeros((N, B, 3))
        self.prev_blue_accel = jnp.zeros((N, B, 3))

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self.cfg.n_worlds

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional options (unused).

        Returns:
            Tuple of (observations, info).
        """
        super().reset(seed=seed)

        if seed is not None:
            self.sim.seed(seed)

        # Reset simulator
        self.sim.reset()

        # Apply domain randomization (samples new mass/inertia each episode)
        self._apply_domain_randomization()

        # Reset episode counters (per-world)
        self.episode_steps = np.zeros(self.cfg.n_worlds, dtype=np.int32)

        # Reset termination event tracking (rates, normalized by n_worlds)
        self.last_termination_events = {
            "bb_collision": 0.0,
            "rr_collision": 0.0,
            "rb_collision": 0.0,
            "out_of_bounds": 0.0,
            "red_win": 0.0,
            "blue_win": 0.0,
            "max_steps": 0.0,
        }

        # Reset alive status
        self.blue_alive = jnp.ones((self.cfg.n_worlds, self.cfg.n_blue), dtype=jnp.bool_)
        self.red_alive = jnp.ones((self.cfg.n_worlds, self.cfg.n_red), dtype=jnp.bool_)

        # Reset blue velocity/acceleration tracking (for ProNav feedforward)
        self.prev_blue_vel = jnp.zeros((self.cfg.n_worlds, self.cfg.n_blue, 3))
        self.prev_blue_accel = jnp.zeros((self.cfg.n_worlds, self.cfg.n_blue, 3))

        # Reset last blue commands (for action smoothness penalty)
        self.last_blue_cmd = jnp.zeros((self.cfg.n_worlds, self.cfg.n_blue, 4))

        # Reset target assignments
        N, B, R = self.cfg.n_worlds, self.cfg.n_blue, self.cfg.n_red
        if self.cfg.random_target_assignment:
            key = self.sim.data.core.rng_key
            key, target_key = jax.random.split(key)
            self.sim.data = self.sim.data.replace(
                core=self.sim.data.core.replace(rng_key=key)
            )
            self.red_target, self.red_target_one_hot = _jit_random_target_assignment(
                target_key, N, B, R
            )
        else:
            self.red_target, self.red_target_one_hot = _jit_deterministic_target_assignment(
                N, B, R
            )

        # Spawn agents
        self._spawn_agents()

        # Get observations
        obs = self._get_observations()
        info = self._get_info()

        return obs, info

    def _spawn_agents(self):
        """Spawn agents using the configured spawn function."""
        N, B, R = self.cfg.n_worlds, self.cfg.n_blue, self.cfg.n_red

        # Get random key from simulator and split it
        key = self.sim.data.core.rng_key
        key, spawn_key = jax.random.split(key)

        # Update simulator's key so next call gets a different key
        self.sim.data = self.sim.data.replace(
            core=self.sim.data.core.replace(rng_key=key)
        )

        # Call spawn function to get positions
        blue_pos, red_pos = self._spawn_fn(spawn_key, N, B, R)

        # Combine positions for all drones
        all_pos = jnp.concatenate([blue_pos, red_pos], axis=1)

        # Initialize rotor velocities at hover RPM for all drones
        all_rotor_vel = jnp.full((N, B + R, 4), self.hover_rpm)

        # Reset velocities and orientation
        all_vel = jnp.zeros((N, B + R, 3))
        all_ang_vel = jnp.zeros((N, B + R, 3))
        identity_quat = jnp.array([0.0, 0.0, 0.0, 1.0])
        all_quat = jnp.broadcast_to(identity_quat, (N, B + R, 4))

        # Update sim state
        states = self.sim.data.states.replace(
            pos=all_pos,
            vel=all_vel,
            quat=all_quat,
            ang_vel=all_ang_vel,
            rotor_vel=all_rotor_vel
        )
        self.sim.data = self.sim.data.replace(states=states)

    def _reset_done_worlds(self, done_mask: np.ndarray):
        """Reset specific worlds that have terminated or truncated.

        Uses vectorized jnp.where operations instead of Python for loops
        for efficient batch reset.

        Args:
            done_mask: Boolean array of shape (N,) indicating which worlds to reset.
        """
        N, B, R = self.cfg.n_worlds, self.cfg.n_blue, self.cfg.n_red

        if not done_mask.any():
            return

        # Convert mask to JAX for vectorized operations
        mask_jnp = jnp.array(done_mask)

        # Reset episode steps for done worlds
        self.episode_steps[done_mask] = 0

        # Reset alive status using vectorized where
        self.blue_alive = jnp.where(mask_jnp[:, None], True, self.blue_alive)
        self.red_alive = jnp.where(mask_jnp[:, None], True, self.red_alive)

        # Reset blue velocity/acceleration tracking
        self.prev_blue_vel = jnp.where(mask_jnp[:, None, None], 0.0, self.prev_blue_vel)
        self.prev_blue_accel = jnp.where(mask_jnp[:, None, None], 0.0, self.prev_blue_accel)

        # Reset last blue commands (for action smoothness penalty)
        self.last_blue_cmd = jnp.where(mask_jnp[:, None, None], 0.0, self.last_blue_cmd)

        # Respawn agents in done worlds using spawn function
        key = self.sim.data.core.rng_key
        key, spawn_key, target_key = jax.random.split(key, 3)

        # Reset target assignments for done worlds
        # Always compute for all N worlds to avoid JIT recompilation
        if self.cfg.random_target_assignment:
            all_targets, all_one_hot = _jit_random_target_assignment(
                target_key, N, B, R
            )
        else:
            all_targets, all_one_hot = _jit_deterministic_target_assignment(
                N, B, R
            )
        # Vectorized target update using where
        self.red_target = jnp.where(mask_jnp[:, None], all_targets, self.red_target)
        self.red_target_one_hot = jnp.where(mask_jnp[:, None, None], all_one_hot, self.red_target_one_hot)

        # Update simulator's key so next call gets a different key
        self.sim.data = self.sim.data.replace(
            core=self.sim.data.core.replace(rng_key=key)
        )

        # Reset simulator state for done worlds (velocities, quaternions, etc.)
        self.sim.reset(mask=mask_jnp)

        # Always spawn for all N worlds to avoid JIT recompilation
        all_blue_pos, all_red_pos = self._spawn_fn(spawn_key, N, B, R)
        new_pos = jnp.concatenate([all_blue_pos, all_red_pos], axis=1)  # (N, B+R, 3)

        # Set rotor velocities to hover RPM
        hover_rpm = jnp.broadcast_to(
            jnp.array(self.hover_rpm), self.sim.data.states.rotor_vel.shape
        )

        # Update sim state with spawn positions and hover RPM using leaf_replace
        states = leaf_replace(
            self.sim.data.states, mask=mask_jnp, pos=new_pos, rotor_vel=hover_rpm
        )
        self.sim.data = self.sim.data.replace(states=states)

        # Apply domain randomization to reset worlds
        self._apply_domain_randomization(mask=mask_jnp)

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        """Execute one environment step.

        Args:
            actions: Dict mapping agent names to attitude commands.
                     Each action is [roll, pitch, yaw, thrust].

        Returns:
            observations, rewards, terminated, truncated, info
        """
        debug_timing = self.cfg.debug_timing
        if debug_timing:
            t_start = time.perf_counter()
            timings = {}

        # Save current blue cmd before updating (for action smoothness penalty)
        self.last_blue_cmd = self.blue_cmd

        # Process blue agent actions
        if debug_timing:
            t0 = time.perf_counter()
        self._process_blue_actions(actions)
        if debug_timing:
            timings["process_actions"] = time.perf_counter() - t0

        # Apply controls
        if debug_timing:
            t0 = time.perf_counter()
        self._apply_controls()
        if debug_timing:
            timings["apply_controls"] = time.perf_counter() - t0

        # Step simulation
        if debug_timing:
            t0 = time.perf_counter()
        self.sim.step(n_steps=self.cfg.sim_steps_per_control)
        if debug_timing:
            timings["sim_step"] = time.perf_counter() - t0

        # Check collisions and update alive status
        if debug_timing:
            t0 = time.perf_counter()
        bb_collision, rr_collision, rb_collision, out_of_bounds = self._check_collisions()
        if debug_timing:
            timings["check_collisions"] = time.perf_counter() - t0

        # Save blue_alive BEFORE updating for reward computation
        # (agents that just died should still receive their death penalty)
        pre_collision_blue_alive = self.blue_alive

        if debug_timing:
            t0 = time.perf_counter()
        self._update_alive_status(bb_collision, rr_collision, rb_collision, out_of_bounds)
        if debug_timing:
            timings["update_alive"] = time.perf_counter() - t0

        # Batch termination event counts into single JAX sync
        # Count per-agent collisions (not per-world) for accurate rate calculation
        if debug_timing:
            t0 = time.perf_counter()
        n_worlds = self.cfg.n_worlds
        term_counts = jnp.stack([
            bb_collision.sum(),  # Total blue agents with BB collision
            rr_collision.sum(),  # Total red agents with RR collision
            rb_collision.sum() * 2,  # Total agents in RB collision (1 blue + 1 red each)
            out_of_bounds.sum(),  # Total blue agents out of bounds
        ])
        term_counts_np = np.asarray(term_counts)
        if debug_timing:
            timings["term_counts_sync"] = time.perf_counter() - t0
        self.last_termination_events = {
            "bb_collision": float(term_counts_np[0]) / n_worlds,
            "rr_collision": float(term_counts_np[1]) / n_worlds,
            "rb_collision": float(term_counts_np[2]) / n_worlds,
            "out_of_bounds": float(term_counts_np[3]) / n_worlds,
        }

        # Compute rewards using pre-collision alive status
        # (so agents that just died receive their death penalty)
        if debug_timing:
            t0 = time.perf_counter()
        rewards = self._compute_rewards(
            bb_collision, rr_collision, rb_collision, out_of_bounds, pre_collision_blue_alive
        )
        if debug_timing:
            timings["compute_rewards"] = time.perf_counter() - t0

        # Update episode steps (per-world)
        self.episode_steps += 1

        # Check termination and truncation (compute alive once, reuse)
        if debug_timing:
            t0 = time.perf_counter()
        all_blue_dead = ~self.blue_alive.any(axis=1)
        all_red_dead = ~self.red_alive.any(axis=1)
        episode_terminated = np.asarray(all_blue_dead | all_red_dead)
        if debug_timing:
            timings["termination_sync"] = time.perf_counter() - t0

        # Per-agent termination: agent is terminated if dead OR episode ended
        # This is important for correct GAE computation in SKRL - dead agents
        # should have terminated=True so value bootstrap is 0
        blue_alive_np = np.asarray(self.blue_alive)
        terminated = {}
        for i, agent in enumerate(self.possible_agents):
            agent_dead = ~blue_alive_np[:, i]
            terminated[agent] = agent_dead | episode_terminated

        truncated = self._check_truncated()

        # Add termination/truncation rates (reuse already-computed arrays)
        sample_agent = self.possible_agents[0]
        all_blue_dead_np = np.asarray(all_blue_dead)
        all_red_dead_np = np.asarray(all_red_dead)
        # Red wins: all blues dead (includes draws where both teams die)
        self.last_termination_events["red_win"] = float(all_blue_dead_np.sum()) / n_worlds
        # Blue wins: all reds dead AND at least one blue survives (excludes mutual destruction)
        blue_win = all_red_dead_np & ~all_blue_dead_np
        self.last_termination_events["blue_win"] = float(blue_win.sum()) / n_worlds
        # Only count max_steps for worlds that truncated but did NOT terminate
        # (avoids double counting if termination happens at max_steps)
        truncated_only = truncated[sample_agent] & ~episode_terminated
        self.last_termination_events["max_steps"] = float(truncated_only.sum()) / n_worlds

        # Save alive status BEFORE auto-reset for info dict
        pre_reset_blue_alive = np.array(self.blue_alive)
        pre_reset_red_alive = np.array(self.red_alive)

        # Auto-reset worlds that are done (episode terminated or truncated)
        # Use episode_terminated (not per-agent terminated) for reset decision
        if debug_timing:
            t0 = time.perf_counter()
        done_mask = episode_terminated | truncated[sample_agent]
        if done_mask.any():
            self._reset_done_worlds(done_mask)
        if debug_timing:
            timings["reset_done_worlds"] = time.perf_counter() - t0

        # Get observations (after auto-reset so we return initial obs for reset worlds)
        if debug_timing:
            t0 = time.perf_counter()
        obs = self._get_observations()
        if debug_timing:
            timings["get_observations"] = time.perf_counter() - t0

        if debug_timing:
            t0 = time.perf_counter()
        info = self._get_info(
            bb_collision, rr_collision, rb_collision, out_of_bounds, rewards,
            pre_reset_blue_alive, pre_reset_red_alive, episode_terminated
        )
        if debug_timing:
            timings["get_info"] = time.perf_counter() - t0
            timings["total"] = time.perf_counter() - t_start
            # Print timing breakdown (every 10 steps to reduce noise but still be useful)
            if not hasattr(self, "_debug_step_count"):
                self._debug_step_count = 0
            self._debug_step_count += 1
            if self._debug_step_count % 10 == 0:
                timing_str = " | ".join(f"{k}={v*1000:.2f}ms" for k, v in timings.items())
                print(f"[Step {self._debug_step_count}] {timing_str}")

        return obs, rewards, terminated, truncated, info

    def _process_blue_actions(self, actions: dict[str, np.ndarray]):
        """Process blue agent actions into attitude commands.

        Actions are expected in physical units [roll, pitch, yaw, thrust].
        Use RescaleActionWrapper to convert normalized [-1, 1] actions from policies.
        """
        B = self.cfg.n_blue

        # Stack actions for all blue agents
        action_array = np.zeros((self.cfg.n_worlds, B, 4), dtype=np.float32)
        for i, agent_name in enumerate(self.possible_agents):
            if agent_name in actions:
                action_array[:, i] = actions[agent_name]

        # Clip to ensure within bounds
        action_array[..., 0] = np.clip(action_array[..., 0], -self.cfg.roll_pitch_max, self.cfg.roll_pitch_max)
        action_array[..., 1] = np.clip(action_array[..., 1], -self.cfg.roll_pitch_max, self.cfg.roll_pitch_max)
        action_array[..., 2] = np.clip(action_array[..., 2], -self.cfg.yaw_max, self.cfg.yaw_max)
        action_array[..., 3] = np.clip(action_array[..., 3], self.cfg.min_thrust, self.cfg.max_thrust)

        self.blue_cmd = jnp.array(action_array)

    def _apply_controls(self):
        """Apply control commands to the simulator."""
        B = self.cfg.n_blue
        R = self.cfg.n_red

        # Get current state
        states = self.sim.data.states
        pos = states.pos
        vel = states.vel
        quat = states.quat
        ang_vel = states.ang_vel

        # Extract blue and red states
        blue_pos = pos[:, :B]
        blue_vel = vel[:, :B]
        blue_quat = quat[:, :B]

        red_pos = pos[:, B:]
        red_vel = vel[:, B:]
        red_quat = quat[:, B:]

        # Update blue acceleration estimate from velocity change (for ProNav feedforward)
        dt = self.cfg.dt
        self.prev_blue_accel = (blue_vel - self.prev_blue_vel) / dt
        self.prev_blue_vel = blue_vel

        # Compute red pursuit control
        red_cmd = self._compute_red_control(
            blue_pos, blue_vel, red_pos, red_vel, red_quat
        )

        # Combine blue and red commands
        # Blue commands are already set in self.blue_cmd
        all_cmd = jnp.concatenate([self.blue_cmd, red_cmd], axis=1)

        # Apply attitude control
        self.sim.attitude_control(all_cmd)

    def _compute_red_control(
        self,
        blue_pos: jnp.ndarray,
        blue_vel: jnp.ndarray,
        red_pos: jnp.ndarray,
        red_vel: jnp.ndarray,
        red_quat: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute red pursuit control commands (JIT-compiled).

        Args:
            blue_pos: Blue positions, shape (N, B, 3).
            blue_vel: Blue velocities, shape (N, B, 3).
            red_pos: Red positions, shape (N, R, 3).
            red_vel: Red velocities, shape (N, R, 3).
            red_quat: Red quaternions (xyzw), shape (N, R, 4).

        Returns:
            Red attitude commands [roll, pitch, yaw, thrust], shape (N, R, 4).
        """
        return _jit_compute_red_control(
            blue_pos, blue_vel, red_pos, red_vel, red_quat,
            self.red_target, self.red_alive, self.prev_blue_accel,
            self.cfg.pursuer_strategy,
            self.cfg.mass,
            self.cfg.roll_pitch_max,
            self.cfg.yaw_max,
            self.cfg.min_thrust,
            self.cfg.max_thrust,
            self.cfg.pp_k_pxy,
            self.cfg.pp_k_vxy,
            self.cfg.pp_k_pz,
            self.cfg.pp_k_vz,
            self.cfg.N_pronav_fb,
            self.cfg.N_pronav_ff,
            self.cfg.velocity_closure_threshold,
            self.cfg.gravity,
            self.cfg.N_gain,
            self.cfg.V_min,
            self.cfg.K_v,
        )

    def _quat_to_rpy(self, quat: jnp.ndarray) -> jnp.ndarray:
        """Convert quaternion (xyzw) to roll-pitch-yaw using JIT-compiled JAX.

        Args:
            quat: Quaternion array, shape (..., 4) in xyzw order.

        Returns:
            RPY angles, shape (..., 3).
        """
        return _jit_quat_to_rpy(quat)

    def _check_collisions(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Check for collisions and boundary violations (JIT-compiled).

        Returns:
            bb_collision: Blue-blue collisions, shape (N, B).
            rr_collision: Red-red collisions, shape (N, R).
            rb_collision: Blue-red captures, shape (N, B).
            out_of_bounds: Blue boundary violations, shape (N, B).
        """
        B = self.cfg.n_blue

        states = self.sim.data.states
        blue_pos = states.pos[:, :B]
        red_pos = states.pos[:, B:]

        return _jit_check_collisions(
            blue_pos, red_pos,
            self.blue_alive, self.red_alive,
            self.cfg.bb_collision_tolerance,
            self.cfg.rr_collision_tolerance,
            self.cfg.rb_collision_tolerance,
            self.cfg.min_altitude,
            self.cfg.max_altitude,
            self.cfg.boundary_size,
        )

    def _update_alive_status(
        self,
        bb_collision: jnp.ndarray,
        rr_collision: jnp.ndarray,
        rb_collision: jnp.ndarray,
        out_of_bounds: jnp.ndarray,
    ):
        """Update alive status based on collisions (JIT-compiled)."""
        B = self.cfg.n_blue
        states = self.sim.data.states
        blue_pos = states.pos[:, :B]
        red_pos = states.pos[:, B:]

        # Use JIT-compiled function for alive update and target reassignment
        self.blue_alive, self.red_alive, self.red_target, self.red_target_one_hot = (
            _jit_update_alive_and_reassign(
                self.blue_alive, self.red_alive, self.red_target,
                bb_collision, rr_collision, rb_collision, out_of_bounds,
                blue_pos, red_pos,
                self.cfg.rb_collision_tolerance,
            )
        )

    def _compute_rewards(
        self,
        bb_collision: jnp.ndarray,
        rr_collision: jnp.ndarray,
        rb_collision: jnp.ndarray,
        out_of_bounds: jnp.ndarray,
        blue_alive: jnp.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute rewards for each agent (JIT-compiled).

        Uses team reward - all blue agents get the same reward.
        Also stores individual reward components in self.last_reward_components
        for logging purposes.

        Args:
            blue_alive: Pre-collision alive status to use for reward computation.
                       Agents that just died should still receive their death penalty.
        """
        B = self.cfg.n_blue
        states = self.sim.data.states
        red_pos = states.pos[:, B:]

        # Compute blue RPY for angle penalty
        blue_quat = states.quat[:, :B]
        blue_rpy = self._quat_to_rpy(blue_quat)

        # Get blue velocity for velocity penalty
        blue_vel = states.vel[:, :B]

        total_reward, components = _jit_compute_rewards(
            bb_collision, rr_collision, rb_collision, out_of_bounds,
            blue_alive, red_pos,
            blue_rpy, blue_vel, self.blue_cmd, self.last_blue_cmd,
            self.cfg.reward_blue_crash,
            self.cfg.reward_red_crash,
            self.cfg.reward_capture,
            self.cfg.reward_boundary,
            self.cfg.reward_alive,
            self.cfg.reward_pursuer_proximity,
            self.cfg.reward_pursuer_proximity_decay,
            self.cfg.reward_angle_coef,
            self.cfg.reward_velocity_coef,
            self.cfg.reward_action_coef,
            self.cfg.reward_action_smoothness_thrust,
            self.cfg.reward_action_smoothness_rpy,
            self.n_pairs,
        )

        # Store individual reward components per world (for episode return tracking)
        self.last_reward_components = {
            key: np.array(val) for key, val in components.items()
        }

        total_reward_np = np.array(total_reward)
        return {agent: total_reward_np for agent in self.possible_agents}

    def _check_terminated(self) -> dict[str, np.ndarray]:
        """Check if episode is terminated (all blue dead or all red dead)."""
        # Keep in JAX until final conversion (np.asarray is zero-copy)
        all_blue_dead = ~self.blue_alive.any(axis=1)
        all_red_dead = ~self.red_alive.any(axis=1)
        # Episode ends if either team is eliminated
        terminated_np = np.asarray(all_blue_dead | all_red_dead)
        return {agent: terminated_np for agent in self.possible_agents}

    def _check_truncated(self) -> dict[str, np.ndarray]:
        """Check if episode is truncated (max steps reached)."""
        truncated_np = self.episode_steps >= self._max_episode_steps
        return {agent: truncated_np for agent in self.possible_agents}

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Get observations for each blue agent (vectorized)."""
        N, B, R = self.cfg.n_worlds, self.cfg.n_blue, self.cfg.n_red

        # Single JAX->NumPy conversion for all state data
        states = self.sim.data.states
        pos_np = np.asarray(states.pos)
        vel_np = np.asarray(states.vel)

        blue_pos = pos_np[:, :B]
        blue_vel = vel_np[:, :B]
        red_pos = pos_np[:, B:]
        red_vel = vel_np[:, B:]

        # Convert alive status once
        blue_alive_np = np.asarray(self.blue_alive).astype(np.float32)
        red_alive_np = np.asarray(self.red_alive).astype(np.float32)
        ally_one_hot_np = np.asarray(self.ally_one_hot)
        red_target_one_hot_np = np.asarray(self.red_target_one_hot)

        # Convert quaternion to RPY (JIT-compiled)
        blue_quat = states.quat[:, :B]
        blue_rpy = np.asarray(self._quat_to_rpy(blue_quat))

        # Cache RPY for _get_shared_state (avoids recomputation in env.state())
        self._cached_blue_rpy = blue_rpy

        # Convert body angular velocity to Euler rates (JIT-compiled)
        blue_ang_vel = states.ang_vel[:, :B]
        blue_rpy_rates = np.asarray(_jit_ang_vel_to_rpy_rates(blue_quat, blue_ang_vel))

        # Build all agents' own states at once: (N, B, 12)
        all_own_states = np.concatenate([
            blue_pos, blue_vel, blue_rpy, blue_rpy_rates
        ], axis=-1)

        # All ally one-hots masked by alive: (N, B, B)
        all_ally_one_hot = ally_one_hot_np * blue_alive_np[:, None, :]

        # Build shared masked states (same for all agents)
        blue_states = np.concatenate([
            blue_pos, blue_vel, blue_alive_np[:, :, None]
        ], axis=-1)
        red_states = np.concatenate([
            red_pos, red_vel, red_alive_np[:, :, None]
        ], axis=-1)
        blue_states_masked = (blue_states * blue_alive_np[:, :, None]).reshape(N, -1)
        red_states_masked = (red_states * red_alive_np[:, :, None]).reshape(N, -1)
        target_one_hot_masked = (red_target_one_hot_np * red_alive_np[:, :, None]).reshape(N, -1)

        # Tile shared parts for all agents: (N, B, shared_dim)
        shared = np.concatenate([blue_states_masked, red_states_masked, target_one_hot_masked], axis=-1)
        shared_tiled = np.broadcast_to(shared[:, None, :], (N, B, shared.shape[-1]))

        # Concatenate all parts: (N, B, obs_dim)
        all_obs = np.concatenate([all_own_states, all_ally_one_hot, shared_tiled], axis=-1).astype(np.float32)

        # Split into per-agent dict
        return {agent: all_obs[:, i] for i, agent in enumerate(self.possible_agents)}

    def _get_shared_state(self) -> np.ndarray:
        """Get shared state for centralized critic.

        Reuses cached blue_rpy from _get_observations when available (called in same step).
        """
        N, B, R = self.cfg.n_worlds, self.cfg.n_blue, self.cfg.n_red

        states = self.sim.data.states
        pos_np = np.asarray(states.pos)
        vel_np = np.asarray(states.vel)

        blue_pos = pos_np[:, :B]
        blue_vel = vel_np[:, :B]

        # Reuse cached blue_rpy if available (set by _get_observations in same step)
        if hasattr(self, '_cached_blue_rpy') and self._cached_blue_rpy is not None:
            blue_rpy = self._cached_blue_rpy
        else:
            blue_rpy = np.asarray(self._quat_to_rpy(states.quat[:, :B]))

        red_pos = pos_np[:, B:]
        red_vel = vel_np[:, B:]
        red_rpy = np.asarray(self._quat_to_rpy(states.quat[:, B:]))

        blue_alive_np = np.asarray(self.blue_alive).astype(np.float32)
        red_alive_np = np.asarray(self.red_alive).astype(np.float32)

        blue_states = np.concatenate([
            blue_pos, blue_vel, blue_rpy, blue_alive_np[:, :, None]
        ], axis=-1).reshape(N, -1)

        red_states = np.concatenate([
            red_pos, red_vel, red_rpy, red_alive_np[:, :, None]
        ], axis=-1).reshape(N, -1)

        target_one_hot = np.asarray(self.red_target_one_hot).reshape(N, -1)

        return np.concatenate([blue_states, red_states, target_one_hot], axis=-1).astype(np.float32)

    def state(self) -> np.ndarray:
        """Get global/shared state for centralized critic (SKRL interface)."""
        return self._get_shared_state()

    def _get_info(
        self,
        bb_collision: jnp.ndarray | None = None,
        rr_collision: jnp.ndarray | None = None,
        rb_collision: jnp.ndarray | None = None,
        out_of_bounds: jnp.ndarray | None = None,
        rewards: dict[str, np.ndarray] | None = None,
        pre_reset_blue_alive: np.ndarray | None = None,
        pre_reset_red_alive: np.ndarray | None = None,
        episode_terminated: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Get info dict.

        Note: Shared state for the centralized critic is NOT included here.
        SKRL's trainer calls env.state() separately after each step.

        Args:
            bb_collision: Blue-blue collisions this step, shape (N, B).
            rr_collision: Red-red collisions this step, shape (N, R).
            rb_collision: Blue-red captures this step, shape (N, B).
            out_of_bounds: Boundary violations this step, shape (N, B).
            rewards: Reward dict per agent, each shape (N,).
            pre_reset_blue_alive: Blue alive status before auto-reset, shape (N, B).
            pre_reset_red_alive: Red alive status before auto-reset, shape (N, R).
            episode_terminated: Episode-level termination (all blue or all red dead), shape (N,).

        Returns:
            Info dictionary with alive status and termination events.
        """
        # Use pre-reset alive status if provided (for accurate termination tracking)
        blue_alive = pre_reset_blue_alive if pre_reset_blue_alive is not None else np.array(self.blue_alive)
        red_alive = pre_reset_red_alive if pre_reset_red_alive is not None else np.array(self.red_alive)

        info = {
            "blue_alive": blue_alive,
            "red_alive": red_alive,
            # Episode-level termination for eval scripts (distinct from per-agent terminated)
            "episode_terminated": episode_terminated if episode_terminated is not None else np.zeros(self.cfg.n_worlds, dtype=bool),
        }

        # Add termination event rates if provided (normalized by n_worlds)
        if bb_collision is not None:
            n_worlds = self.cfg.n_worlds
            info["termination/bb_collision"] = float(bb_collision.any(axis=1).sum()) / n_worlds
            info["termination/rr_collision"] = float(rr_collision.any(axis=1).sum()) / n_worlds
            info["termination/rb_collision"] = float(rb_collision.any(axis=1).sum()) / n_worlds
            info["termination/out_of_bounds"] = float(out_of_bounds.any(axis=1).sum()) / n_worlds

        # Add mean reward (normalized across worlds)
        if rewards is not None:
            sample_agent = self.possible_agents[0]
            info["reward/mean"] = float(rewards[sample_agent].mean())

        return info

    def render(self, world: int = 0) -> np.ndarray | None:
        """Render the environment with LED colors for teams and dead drones hidden.

        Blue team drones glow blue, red team drones glow red.
        Dead drones are moved underground and have LEDs turned off.

        Args:
            world: Which world to render (default 0).
        """
        if self.render_mode is None:
            return None

        import mujoco
        from crazyflow.sim.visualize import change_material

        # Initialize viewer if needed (copied from Crazyflow)
        if self.sim.viewer is None:
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
            self.sim.mj_model.vis.global_.offwidth = 1920
            self.sim.mj_model.vis.global_.offheight = 1080
            # Camera config: distance=6m for wider view of the arena
            cam_config = {
                "distance": 6.0,
                "azimuth": 90.0,
                "elevation": -25.0,  # Consistent with eval script
                "lookat": [0.0, 0.0, 1.0],  # Look at center of arena at 1m height
            }
            self.sim.viewer = MujocoRenderer(
                self.sim.mj_model,
                self.sim.mj_data,
                max_geom=self.sim.max_visual_geom,
                default_cam_config=cam_config,
                height=1080,
                width=1920,
                camera_id=-1,
            )
            # Initialize LED colors on first render
            self._leds_initialized = False

        # Set LED colors for teams (only need to set once, then just update emission)
        if not getattr(self, '_leds_initialized', False):
            # Blue team: indices 0 to n_blue-1
            blue_ids = np.arange(self.cfg.n_blue)
            blue_rgba = np.array([0.0, 0.0, 1.0, 1.0])  # Pure blue
            change_material(self.sim, "led_top", blue_ids, rgba=blue_rgba, emission=1.0)
            change_material(self.sim, "led_bot", blue_ids, rgba=blue_rgba, emission=1.0)

            # Red team: indices n_blue to n_blue+n_red-1
            red_ids = np.arange(self.cfg.n_blue, self.cfg.n_blue + self.cfg.n_red)
            red_rgba = np.array([1.0, 0.0, 0.0, 1.0])  # Pure red
            change_material(self.sim, "led_top", red_ids, rgba=red_rgba, emission=1.0)
            change_material(self.sim, "led_bot", red_ids, rgba=red_rgba, emission=1.0)

            self._leds_initialized = True

        # Build qpos from current state (pos + quat for each drone)
        # Format: [x, y, z, qw, qx, qy, qz] per drone
        states = self.sim.data.states
        pos = np.asarray(states.pos[world])  # (n_drones, 3)
        quat = np.asarray(states.quat[world])  # (n_drones, 4)

        # Get alive status
        blue_alive = np.asarray(self.blue_alive[world])
        red_alive = np.asarray(self.red_alive[world])
        all_alive = np.concatenate([blue_alive, red_alive])

        # Update LED emission based on alive status
        # Alive drones: full emission, dead drones: no emission
        blue_ids = np.arange(self.cfg.n_blue)
        red_ids = np.arange(self.cfg.n_blue, self.cfg.n_blue + self.cfg.n_red)

        # Update blue team LEDs
        alive_blue_ids = blue_ids[blue_alive]
        dead_blue_ids = blue_ids[~blue_alive]
        if len(alive_blue_ids) > 0:
            change_material(self.sim, "led_top", alive_blue_ids, emission=1.0)
            change_material(self.sim, "led_bot", alive_blue_ids, emission=1.0)
        if len(dead_blue_ids) > 0:
            change_material(self.sim, "led_top", dead_blue_ids, emission=0.0)
            change_material(self.sim, "led_bot", dead_blue_ids, emission=0.0)

        # Update red team LEDs
        alive_red_ids = red_ids[red_alive]
        dead_red_ids = red_ids[~red_alive]
        if len(alive_red_ids) > 0:
            change_material(self.sim, "led_top", alive_red_ids, emission=1.0)
            change_material(self.sim, "led_bot", alive_red_ids, emission=1.0)
        if len(dead_red_ids) > 0:
            change_material(self.sim, "led_top", dead_red_ids, emission=0.0)
            change_material(self.sim, "led_bot", dead_red_ids, emission=0.0)

        # Build qpos with dead drones moved underground
        qpos = np.zeros(self.cfg.n_drones * 7)
        for drone_idx in range(self.cfg.n_drones):
            qpos_start = drone_idx * 7
            if all_alive[drone_idx]:
                qpos[qpos_start:qpos_start + 3] = pos[drone_idx]
                qpos[qpos_start + 3:qpos_start + 7] = quat[drone_idx]
            else:
                # Dead drone - move underground
                qpos[qpos_start:qpos_start + 3] = [0, 0, -100]
                qpos[qpos_start + 3:qpos_start + 7] = [1, 0, 0, 0]  # Identity quaternion

        self.sim.mj_data.qpos[:] = qpos
        self.sim.mj_data.mocap_pos[:] = np.asarray(self.sim.mjx_data.mocap_pos[world, :])
        self.sim.mj_data.mocap_quat[:] = np.asarray(self.sim.mjx_data.mocap_quat[world, :])

        # Forward dynamics to update rendering state
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)

        # Render
        return self.sim.viewer.render(self.render_mode)

    def close(self):
        """Close the environment."""
        self.sim.close()

    def update_curriculum_params(
        self,
        spawn_fn: SpawnFn | None = None,
        **params,
    ):
        """Update environment parameters for curriculum learning.

        This method allows dynamic adjustment of any config parameters
        during training without recreating the environment.

        Args:
            spawn_fn: New spawn function for agent positioning (optional).
            **params: Any parameters from RedVsBlueEnvConfig to update.
                Examples: rb_collision_tolerance, bb_collision_tolerance, boundary_size,
                reward_capture, N_pronav_fb, etc.

        Raises:
            AttributeError: If a parameter name doesn't exist in the config.
        """
        # Update spawn function if provided
        if spawn_fn is not None:
            self._spawn_fn = spawn_fn

        # Track if disturbance settings changed
        disturbance_changed = False
        old_enable_disturbance = self.cfg.enable_disturbance

        # Update any config parameters
        for param_name, param_value in params.items():
            if not hasattr(self.cfg, param_name):
                raise AttributeError(
                    f"Unknown curriculum parameter: '{param_name}'. "
                    f"Must be a valid RedVsBlueEnvConfig field."
                )
            setattr(self.cfg, param_name, param_value)

            # Track disturbance-related changes
            if param_name in ("enable_disturbance", "disturbance_force_std", "disturbance_torque_std"):
                disturbance_changed = True

        # Handle disturbance enable/disable or std changes
        if disturbance_changed:
            if self.cfg.enable_disturbance:
                # Enable or update disturbance (recreates function with new std values)
                self._enable_disturbance()
            elif old_enable_disturbance and not self.cfg.enable_disturbance:
                # Disable disturbance
                self._disable_disturbance()
