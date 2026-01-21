"""SX-based so_rpy Euler dynamics for faster MPC solve times.

This module provides CasADi SX symbolic dynamics equivalent to drone-models'
so_rpy symbolic_dynamics_euler, but using SX instead of MX for better performance
in small optimization problems.

State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw] (12D)
Control: [roll_cmd, pitch_cmd, yaw_cmd, thrust] (4D)
"""

from typing import TYPE_CHECKING

import casadi as cs
import numpy as np

if TYPE_CHECKING:
    from drone_models._typing import Array


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
