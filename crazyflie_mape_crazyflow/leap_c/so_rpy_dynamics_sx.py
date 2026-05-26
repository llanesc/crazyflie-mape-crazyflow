"""SX-based so_rpy dynamics for faster MPC solve times.

This module provides CasADi SX symbolic dynamics equivalent to drone-models'
so_rpy models, but using SX instead of MX for better performance in small
optimization problems.

Two dynamics functions are provided:

1. symbolic_dynamics_euler_sx: Euler angle state representation (12D)
   State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
   Equivalent to drone_models.so_rpy.symbolic_dynamics_euler

2. symbolic_dynamics_sx: Quaternion + body rates state representation (13D)
   State: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
   Equivalent to drone_models.so_rpy.symbolic_dynamics

Control: [roll_cmd, pitch_cmd, yaw_cmd, thrust] (4D) for both.
"""

from typing import TYPE_CHECKING

import casadi as cs
import numpy as np

if TYPE_CHECKING:
    from drone_models._typing import Array


# =============================================================================
# SX Rotation Utilities
# =============================================================================


def sx_rpy2matrix(rpy: cs.SX) -> cs.SX:
    """Create rotation matrix from roll, pitch, yaw using SX (XYZ extrinsic convention).

    Equivalent to scipy.spatial.transform.Rotation.from_euler('xyz', rpy).as_matrix()

    Args:
        rpy: Roll, pitch, yaw angles [rad] as SX vector.

    Returns:
        3x3 rotation matrix as SX.
    """
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

    cr = cs.cos(roll)
    sr = cs.sin(roll)
    cp = cs.cos(pitch)
    sp = cs.sin(pitch)
    cy = cs.cos(yaw)
    sy = cs.sin(yaw)

    # Rotation matrix for R = Rz(yaw) * Ry(pitch) * Rx(roll)
    matrix = cs.vertcat(
        cs.horzcat(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        cs.horzcat(sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        cs.horzcat(-sp, cp * sr, cp * cr),
    )

    return matrix


def sx_quat2matrix(quat: cs.SX) -> cs.SX:
    """Create rotation matrix from quaternion (xyzw) using SX.

    Equivalent to scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()

    Args:
        quat: Quaternion as SX vector [qx, qy, qz, qw].

    Returns:
        3x3 rotation matrix as SX.
    """
    n = cs.norm_2(quat)
    x, y, z, w = quat[0] / n, quat[1] / n, quat[2] / n, quat[3] / n

    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, xz, xw = x * y, x * z, x * w
    yz, yw, zw = y * z, y * w, z * w

    matrix = cs.vertcat(
        cs.horzcat(x2 - y2 - z2 + w2, 2.0 * (xy - zw), 2.0 * (xz + yw)),
        cs.horzcat(2.0 * (xy + zw), -x2 + y2 - z2 + w2, 2.0 * (yz - xw)),
        cs.horzcat(2.0 * (xz - yw), 2.0 * (yz + xw), -x2 - y2 + z2 + w2),
    )

    return matrix


def sx_quat2euler_xyz(quat: cs.SX) -> cs.SX:
    """Convert quaternion (xyzw) to XYZ extrinsic Euler angles using SX.

    Equivalent to scipy.spatial.transform.Rotation.from_quat(quat).as_euler('xyz').
    Follows the same algorithm as drone_models.utils.rotation.cs_quat2euler for
    the 'xyz' sequence.

    Args:
        quat: Quaternion as SX vector [qx, qy, qz, qw].

    Returns:
        Euler angles [roll, pitch, yaw] as SX vector.
    """
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

    # xyz extrinsic, non-symmetric case: i=0, j=1, k=2, sign=1
    a = qw - qy
    b = qx + qz
    c = qy + qw
    d = qz - qx

    eps = 1e-7

    angles1 = 2.0 * cs.atan2(cs.sqrt(c**2 + d**2), cs.sqrt(a**2 + b**2))

    case = cs.if_else(
        cs.fabs(angles1) <= eps, 1, cs.if_else(cs.fabs(angles1 - np.pi) <= eps, 2, 0)
    )

    half_sum = cs.atan2(b, a)
    half_diff = cs.atan2(d, c)

    # Normal case (no gimbal lock)
    roll_normal = half_sum - half_diff
    yaw_normal = half_sum + half_diff

    # Edge cases (gimbal lock: pitch ≈ ±π/2)
    roll_edge = cs.if_else(case == 1, 2.0 * half_sum, -2.0 * half_diff)
    yaw_edge = 0.0

    roll = cs.if_else(case == 0, roll_normal, roll_edge)
    pitch = angles1 - np.pi / 2  # Non-symmetric adjustment
    yaw = cs.if_else(case == 0, yaw_normal, yaw_edge)

    # Wrap to [-π, π]
    roll = roll + cs.if_else(roll < -np.pi, 2 * np.pi, cs.if_else(roll > np.pi, -2 * np.pi, 0))
    pitch = pitch + cs.if_else(
        pitch < -np.pi, 2 * np.pi, cs.if_else(pitch > np.pi, -2 * np.pi, 0)
    )
    yaw = yaw + cs.if_else(yaw < -np.pi, 2 * np.pi, cs.if_else(yaw > np.pi, -2 * np.pi, 0))

    return cs.vertcat(roll, pitch, yaw)


def sx_ang_vel2rpy_rates(rpy: cs.SX, ang_vel: cs.SX) -> cs.SX:
    """Convert body angular velocity to Euler angle rates using SX.

    Computes drpy = W @ ang_vel where W is the angular velocity to Euler rates
    Jacobian matrix.

    Args:
        rpy: Roll, pitch, yaw angles [rad] as SX vector.
        ang_vel: Body angular velocity [rad/s] as SX vector [wx, wy, wz].

    Returns:
        Euler angle rates [droll, dpitch, dyaw] as SX vector.
    """
    phi, theta = rpy[0], rpy[1]

    sp = cs.sin(phi)
    cp = cs.cos(phi)
    tt = cs.tan(theta)
    ict = 1.0 / cs.cos(theta)

    W = cs.vertcat(
        cs.horzcat(1, sp * tt, cp * tt),
        cs.horzcat(0, cp, -sp),
        cs.horzcat(0, sp * ict, cp * ict),
    )

    return W @ ang_vel


def sx_rpy_rates2ang_vel(rpy: cs.SX, rpy_rates: cs.SX) -> cs.SX:
    """Convert Euler angle rates to body angular velocity using SX.

    Computes ang_vel = W_inv @ drpy where W_inv is the inverse of the Euler
    rates Jacobian.

    Args:
        rpy: Roll, pitch, yaw angles [rad] as SX vector.
        rpy_rates: Euler angle rates [rad/s] as SX vector [droll, dpitch, dyaw].

    Returns:
        Body angular velocity [rad/s] as SX vector [wx, wy, wz].
    """
    phi, theta = rpy[0], rpy[1]

    sp = cs.sin(phi)
    cp = cs.cos(phi)
    ct = cs.cos(theta)
    st = cs.sin(theta)

    W_inv = cs.vertcat(
        cs.horzcat(1, 0, -st),
        cs.horzcat(0, cp, sp * ct),
        cs.horzcat(0, -sp, cp * ct),
    )

    return W_inv @ rpy_rates


def sx_rpy_rates_deriv2ang_vel_deriv(
    rpy: cs.SX, rpy_rates: cs.SX, rpy_rates_deriv: cs.SX
) -> cs.SX:
    """Convert Euler rate derivatives to body angular velocity derivatives using SX.

    Computes ang_vel_dot = W_inv_dot @ drpy + W_inv @ ddrpy using the chain rule.

    Equivalent to drone_models.utils.rotation.cs_rpy_rates_deriv2ang_vel_deriv.

    Args:
        rpy: Roll, pitch, yaw angles [rad] as SX vector.
        rpy_rates: Euler angle rates [rad/s] as SX vector [droll, dpitch, dyaw].
        rpy_rates_deriv: Euler angle accelerations [rad/s^2] as SX vector.

    Returns:
        Body angular acceleration [rad/s^2] as SX vector [dwx, dwy, dwz].
    """
    phi, theta = rpy[0], rpy[1]
    phi_dot, theta_dot = rpy_rates[0], rpy_rates[1]

    sp = cs.sin(phi)
    cp = cs.cos(phi)
    ct = cs.cos(theta)
    st = cs.sin(theta)

    # Time derivative of W_inv
    W_inv_dot = cs.vertcat(
        cs.horzcat(0, 0, -ct * theta_dot),
        cs.horzcat(0, -sp * phi_dot, cp * phi_dot * ct - sp * st * theta_dot),
        cs.horzcat(0, -cp * phi_dot, -sp * phi_dot * ct - cp * st * theta_dot),
    )

    return W_inv_dot @ rpy_rates + sx_rpy_rates2ang_vel(rpy, rpy_rates_deriv)


# =============================================================================
# SX Integrators
# =============================================================================


def integrate_rk4_sx(X_dot: cs.SX, X: cs.SX, U: cs.SX, dt: float) -> cs.SX:
    """RK4 integration for SX dynamics.

    Args:
        X_dot: State derivative expression (function of X and U).
        X: State vector (SX symbol).
        U: Control vector (SX symbol).
        dt: Integration timestep [s].

    Returns:
        Next state as SX expression.
    """
    f = cs.Function("f_rk4", [X, U], [X_dot])
    k1 = f(X, U)
    k2 = f(X + dt / 2 * k1, U)
    k3 = f(X + dt / 2 * k2, U)
    k4 = f(X + dt * k3, U)
    return X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_euler_sx(X_dot: cs.SX, X: cs.SX, U: cs.SX, dt: float) -> cs.SX:
    """Forward Euler integration for SX dynamics.

    Args:
        X_dot: State derivative expression (function of X and U).
        X: State vector (SX symbol).
        U: Control vector (SX symbol).
        dt: Integration timestep [s].

    Returns:
        Next state as SX expression.
    """
    return X + dt * X_dot


# =============================================================================
# Euler Angle State Dynamics (12D)
# =============================================================================


def symbolic_dynamics_euler_sx(
    model_rotor_vel: bool = False,
    *,
    mass: float,
    gravity_vec: "Array",
    J: "Array",
    J_inv: "Array",
    acc_coef: float,
    cmd_f_coef: float,
    rpy_coef: "Array",
    rpy_rates_coef: "Array",
    cmd_rpy_coef: "Array",
) -> tuple[cs.SX, cs.SX, cs.SX, cs.SX]:
    """Fitted linear, second order RPY dynamics using CasADi SX.

    This is equivalent to drone_models.so_rpy.symbolic_dynamics_euler but uses
    SX instead of MX for better performance in optimization.

    State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw] (12D)

    Args:
        model_rotor_vel: Whether to model rotor velocity (not supported, kept for API compatibility).
        mass: Mass of the drone [kg].
        gravity_vec: Gravity vector [m/s^2], e.g. [0, 0, -9.81].
        J: Inertia matrix [kg m^2] (unused in this model but kept for API compatibility).
        J_inv: Inverse inertia matrix (unused in this model but kept for API compatibility).
        acc_coef: Acceleration coefficient [m/s^2].
        cmd_f_coef: Thrust command coefficient [m/s^2/N].
        rpy_coef: RPY dynamics coefficient [1/s^2].
        rpy_rates_coef: RPY rates dynamics coefficient [1/s].
        cmd_rpy_coef: RPY command coefficient [1/s^2].

    Returns:
        Tuple of (X_dot, X, U, Y):
            X_dot: State derivative as SX expression.
            X: State vector as SX symbol [pos, rpy, vel, drpy].
            U: Control vector as SX symbol [roll_cmd, pitch_cmd, yaw_cmd, thrust].
            Y: Output vector as SX expression [pos, rpy].
    """
    # Convert numpy arrays to flat lists for CasADi
    gravity_vec = np.asarray(gravity_vec).flatten()
    rpy_coef = np.asarray(rpy_coef).flatten()
    rpy_rates_coef = np.asarray(rpy_rates_coef).flatten()
    cmd_rpy_coef = np.asarray(cmd_rpy_coef).flatten()

    # Define SX symbols for states
    px = cs.SX.sym("px")
    py = cs.SX.sym("py")
    pz = cs.SX.sym("pz")
    pos = cs.vertcat(px, py, pz)

    roll = cs.SX.sym("roll")
    pitch = cs.SX.sym("pitch")
    yaw = cs.SX.sym("yaw")
    rpy = cs.vertcat(roll, pitch, yaw)

    vx = cs.SX.sym("vx")
    vy = cs.SX.sym("vy")
    vz = cs.SX.sym("vz")
    vel = cs.vertcat(vx, vy, vz)

    droll = cs.SX.sym("droll")
    dpitch = cs.SX.sym("dpitch")
    dyaw = cs.SX.sym("dyaw")
    drpy = cs.vertcat(droll, dpitch, dyaw)

    # State vector: [pos, rpy, vel, drpy]
    X = cs.vertcat(pos, rpy, vel, drpy)

    # Define SX symbols for controls
    cmd_roll = cs.SX.sym("cmd_roll")
    cmd_pitch = cs.SX.sym("cmd_pitch")
    cmd_yaw = cs.SX.sym("cmd_yaw")
    cmd_thrust = cs.SX.sym("cmd_thrust")

    # Control vector: [roll_cmd, pitch_cmd, yaw_cmd, thrust]
    U = cs.vertcat(cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust)
    cmd_rpy_vec = cs.vertcat(cmd_roll, cmd_pitch, cmd_yaw)

    # Rotation matrix from body to world frame
    rot = sx_rpy2matrix(rpy)

    # Thrust dynamics
    # forces_motor = acc_coef + cmd_f_coef * cmd_thrust
    forces_motor_vec = cs.vertcat(0, 0, acc_coef + cmd_f_coef * cmd_thrust)

    # Linear equation of motion
    pos_dot = vel
    vel_dot = rot @ forces_motor_vec / mass + cs.vertcat(gravity_vec[0], gravity_vec[1], gravity_vec[2])

    # Rotational equation of motion (second-order linear RPY dynamics)
    # ddrpy = rpy_coef * rpy + rpy_rates_coef * drpy + cmd_rpy_coef * cmd_rpy
    rpy_coef_diag = cs.diag(cs.vertcat(rpy_coef[0], rpy_coef[1], rpy_coef[2]))
    rpy_rates_coef_diag = cs.diag(cs.vertcat(rpy_rates_coef[0], rpy_rates_coef[1], rpy_rates_coef[2]))
    cmd_rpy_coef_diag = cs.diag(cs.vertcat(cmd_rpy_coef[0], cmd_rpy_coef[1], cmd_rpy_coef[2]))

    ddrpy = rpy_coef_diag @ rpy + rpy_rates_coef_diag @ drpy + cmd_rpy_coef_diag @ cmd_rpy_vec

    # State derivative
    X_dot = cs.vertcat(pos_dot, drpy, vel_dot, ddrpy)

    # Output
    Y = cs.vertcat(pos, rpy)

    return X_dot, X, U, Y


# =============================================================================
# Quaternion + Body Rates Dynamics (13D)
# =============================================================================


def symbolic_dynamics_sx(
    model_rotor_vel: bool = False,
    *,
    mass: float,
    gravity_vec: "Array",
    J: "Array",
    J_inv: "Array",
    acc_coef: float,
    cmd_f_coef: float,
    rpy_coef: "Array",
    rpy_rates_coef: "Array",
    cmd_rpy_coef: "Array",
) -> tuple[cs.SX, cs.SX, cs.SX, cs.SX]:
    """Fitted linear, second order RPY dynamics in quaternion + body rates form using CasADi SX.

    This is equivalent to drone_models.so_rpy.symbolic_dynamics but uses SX instead
    of MX for better performance in optimization. Internally computes the RPY dynamics
    and converts to quaternion kinematics + body angular velocity derivatives.

    State: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz] (13D)

    Args:
        model_rotor_vel: Whether to model rotor velocity (not supported, kept for API compatibility).
        mass: Mass of the drone [kg].
        gravity_vec: Gravity vector [m/s^2], e.g. [0, 0, -9.81].
        J: Inertia matrix [kg m^2] (unused in this model but kept for API compatibility).
        J_inv: Inverse inertia matrix (unused in this model but kept for API compatibility).
        acc_coef: Acceleration coefficient [m/s^2].
        cmd_f_coef: Thrust command coefficient [m/s^2/N].
        rpy_coef: RPY dynamics coefficient [1/s^2].
        rpy_rates_coef: RPY rates dynamics coefficient [1/s].
        cmd_rpy_coef: RPY command coefficient [1/s^2].

    Returns:
        Tuple of (X_dot, X, U, Y):
            X_dot: State derivative as SX expression (13D).
            X: State vector as SX symbol [pos, quat, vel, ang_vel] (13D).
            U: Control vector as SX symbol [roll_cmd, pitch_cmd, yaw_cmd, thrust] (4D).
            Y: Output vector as SX expression [pos, quat] (7D).
    """
    # Convert numpy arrays to flat lists for CasADi
    gravity_vec = np.asarray(gravity_vec).flatten()
    rpy_coef_arr = np.asarray(rpy_coef).flatten()
    rpy_rates_coef_arr = np.asarray(rpy_rates_coef).flatten()
    cmd_rpy_coef_arr = np.asarray(cmd_rpy_coef).flatten()

    # Define SX symbols for states
    px = cs.SX.sym("px")
    py = cs.SX.sym("py")
    pz = cs.SX.sym("pz")
    pos = cs.vertcat(px, py, pz)

    qx = cs.SX.sym("qx")
    qy = cs.SX.sym("qy")
    qz = cs.SX.sym("qz")
    qw = cs.SX.sym("qw")
    quat = cs.vertcat(qx, qy, qz, qw)

    vx = cs.SX.sym("vx")
    vy = cs.SX.sym("vy")
    vz = cs.SX.sym("vz")
    vel = cs.vertcat(vx, vy, vz)

    wx = cs.SX.sym("wx")
    wy = cs.SX.sym("wy")
    wz = cs.SX.sym("wz")
    ang_vel = cs.vertcat(wx, wy, wz)

    # State vector: [pos, quat, vel, ang_vel]
    X = cs.vertcat(pos, quat, vel, ang_vel)

    # Define SX symbols for controls
    cmd_roll = cs.SX.sym("cmd_roll")
    cmd_pitch = cs.SX.sym("cmd_pitch")
    cmd_yaw = cs.SX.sym("cmd_yaw")
    cmd_thrust = cs.SX.sym("cmd_thrust")

    # Control vector: [roll_cmd, pitch_cmd, yaw_cmd, thrust]
    U = cs.vertcat(cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust)
    cmd_rpy_vec = cs.vertcat(cmd_roll, cmd_pitch, cmd_yaw)

    # === Convert quaternion state to RPY for internal dynamics computation ===
    rpy = sx_quat2euler_xyz(quat)
    drpy = sx_ang_vel2rpy_rates(rpy, ang_vel)

    # Rotation matrix from body to world frame (computed from quaternion directly)
    rot = sx_quat2matrix(quat)

    # === Linear equation of motion ===
    forces_motor_vec = cs.vertcat(0, 0, acc_coef + cmd_f_coef * cmd_thrust)
    pos_dot = vel
    vel_dot = rot @ forces_motor_vec / mass + cs.vertcat(
        gravity_vec[0], gravity_vec[1], gravity_vec[2]
    )

    # === Rotational equation of motion ===
    # Step 1: Compute ddrpy in Euler angle space (fitted linear dynamics)
    rpy_coef_diag = cs.diag(cs.vertcat(rpy_coef_arr[0], rpy_coef_arr[1], rpy_coef_arr[2]))
    rpy_rates_coef_diag = cs.diag(
        cs.vertcat(rpy_rates_coef_arr[0], rpy_rates_coef_arr[1], rpy_rates_coef_arr[2])
    )
    cmd_rpy_coef_diag = cs.diag(
        cs.vertcat(cmd_rpy_coef_arr[0], cmd_rpy_coef_arr[1], cmd_rpy_coef_arr[2])
    )

    ddrpy = rpy_coef_diag @ rpy + rpy_rates_coef_diag @ drpy + cmd_rpy_coef_diag @ cmd_rpy_vec

    # Step 2: Convert ddrpy to ang_vel_dot using the chain rule
    ang_vel_dot = sx_rpy_rates_deriv2ang_vel_deriv(rpy, drpy, ddrpy)

    # Step 3: Quaternion kinematics: quat_dot = 0.5 * Xi @ quat
    p, q, r = ang_vel[0], ang_vel[1], ang_vel[2]
    xi = cs.vertcat(
        cs.horzcat(0, -p, -q, -r),
        cs.horzcat(p, 0, r, -q),
        cs.horzcat(q, -r, 0, p),
        cs.horzcat(r, q, -p, 0),
    )
    quat_dot = 0.5 * (xi @ quat)

    # State derivative: [pos_dot, quat_dot, vel_dot, ang_vel_dot]
    X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)

    # Output
    Y = cs.vertcat(pos, quat)

    return X_dot, X, U, Y
