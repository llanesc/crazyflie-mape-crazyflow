"""Acados OCP definition for quadrotor attitude control with so_rpy Euler dynamics.

This module defines the optimal control problem for a quadrotor using attitude
commands (roll, pitch, yaw, thrust) with the fitted so_rpy Euler dynamics model
from drone-models. The cost function follows the standard quadratic form:

    J = 0.5 * x^T Q x + p_x^T x + 0.5 * u^T R u + p_u^T u

State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw] (12D)
Control: [roll, pitch, yaw, thrust] (4D)

For stagewise parameters, state-related costs (Q_state, p_x) apply to N+1 stages
(including terminal), while control-related costs (Q_ctrl, p_u) apply to N stages.
"""

from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import cs_rpy2matrix
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

# State dimensions
NX = 12  # [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
NU = 4   # [roll, pitch, yaw, thrust]

# Parameter sizes per stage (all diagonal costs for Euler state)
# Position (3) + RPY (3) + Velocity (3) + RPY rates (3) = 12
Q_STATE_SIZE = 12
Q_CTRL_SIZE = 4     # Control diagonal
P_X_SIZE = 12       # State linear
P_U_SIZE = 4        # Control linear

# Combined sizes
Q_SIZE = Q_STATE_SIZE + Q_CTRL_SIZE  # 16
P_SIZE = P_X_SIZE + P_U_SIZE         # 16

QuadrotorAcadosParamInterface = Literal["global", "stagewise"]

def integrate_euler(f_expl: ca.SX, x: ca.SX, dt: float) -> ca.SX:
    """Integrate dynamics using the forward Euler method.

    Args:
        f_expl: The explicit dynamics (x_dot).
        x: The state vector.
        dt: The time step for integration.

    Returns:
        The updated state vector after integration.
    """
    return x + dt * f_expl


def integrate_erk4(f_expl: ca.SX, x: ca.SX, u: ca.SX, p: ca.SX, dt: float) -> ca.SX:
    """Integrate dynamics using the explicit RK4 method.

    Args:
        f_expl: The explicit dynamics function.
        x: The state vector.
        u: The control input vector.
        p: The parameter vector.
        dt: The time step for integration.

    Returns:
        The updated state vector after integration.
    """
    ode = ca.Function("ode", [x, u, p], [f_expl])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)
    k3 = ode(x + dt / 2 * k2, u, p)
    k4 = ode(x + dt * k3, u, p)

    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def create_quadrotor_params(
    N_horizon: int = 2,
    param_interface: QuadrotorAcadosParamInterface = "global",
    drone_model: str = "cf2x_L250",
) -> list[AcadosParameter]:
    """Create learnable parameters for quadrotor MPC with so_rpy Euler dynamics.

    Parameters are split into state-related and control-related for proper
    stagewise handling. All costs are diagonal for the Euler state representation.

    State-related (N+1 stages including terminal):
    - q_state: Position(3) + RPY(3) + Velocity(3) + RPY rates(3) = 12 (diagonal)
    - p_x: State linear term = 12

    Control-related (N stages, no terminal):
    - q_control: Control diagonal = 4
    - p_u: Control linear term = 4

    Args:
        N_horizon: Number of MPC horizon steps.
        param_interface: "global" for same params all stages, "stagewise" for varying.
        drone_model: Drone model identifier for loading physical parameters.

    Returns:
        List of AcadosParameter objects.
    """
    # Load physical parameters from drone-models
    drone_params = load_params("so_rpy", drone_model)
    mass = float(drone_params["mass"])
    gravity = float(np.abs(drone_params["gravity_vec"][2]))

    state_end_stages = list(range(N_horizon + 1)) if param_interface == "stagewise" else []
    ctrl_end_stages = list(range(N_horizon)) if param_interface == "stagewise" else []

    # Cost penalties for each state/control component
    # State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
    # Position (3) + RPY (3) + Velocity (3) + RPY rates (3) = 12
    state_penalty = np.array([50., 50., 100., 1., 1., 1., 10., 10., 10., 5., 5., 5.])
    # Control: [roll_cmd, pitch_cmd, yaw_cmd, thrust]
    control_penalty = np.array([1., 1., 1., 50.])

    # State scale (bounds for linear cost calculation)
    # Position scale, RPY scale (rad), Velocity scale, RPY rates scale
    state_scale = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    # Action mean and scale
    action_mean = np.array([0.0, 0.0, 0.0, mass * gravity])
    action_scale = np.array([0.5, 0.5, 0.3, mass * gravity])

    # Scaling constants - must match quadrotor_policy.py for proper parameter bounds
    q_nom = np.concatenate((state_penalty, control_penalty))
    px = np.sqrt(state_penalty)
    pu = np.sqrt(control_penalty)
    p_nom = np.concatenate((px, pu)) * np.concatenate((state_scale, action_scale))
    range_Q = 2. * q_nom
    range_p = 2. * p_nom
    range_p_t = 2 * range_Q[-1] / 2 * mass * gravity
    epsilon = 0.1

    q_state_low = np.full((Q_STATE_SIZE,), epsilon)
    q_state_high = range_Q[:Q_STATE_SIZE] + epsilon
    q_state_default = (range_Q[:Q_STATE_SIZE] / 2) + epsilon

    q_ctrl_low = np.full((Q_CTRL_SIZE,), epsilon)
    q_ctrl_high = range_Q[Q_STATE_SIZE:] + epsilon
    q_ctrl_default = (range_Q[Q_STATE_SIZE:] / 2) + epsilon

    p_x_low = -range_p[:P_X_SIZE] / 2
    p_x_high = range_p[:P_X_SIZE] / 2
    p_x_default = np.full((P_X_SIZE,), 0.0)

    p_u_low = -range_p[P_X_SIZE:] / 2
    p_u_high = range_p[P_X_SIZE:] / 2
    # Thrust linear term should be positive (bias toward hover)
    p_u_low[-1] = epsilon
    p_u_high[-1] = range_p_t + epsilon
    p_u_default = np.full((P_U_SIZE,), 0.0)
    p_u_default[-1] = (range_p_t / 2) + epsilon

    return [
        AcadosParameter(
            name="q_state",
            default=q_state_default,
            space=gym.spaces.Box(low=q_state_low, high=q_state_high, dtype=np.float64),
            interface="learnable",
            end_stages=state_end_stages,
        ),
        AcadosParameter(
            name="q_control",
            default=q_ctrl_default,
            space=gym.spaces.Box(low=q_ctrl_low, high=q_ctrl_high, dtype=np.float64),
            interface="learnable",
            end_stages=ctrl_end_stages,
        ),
        AcadosParameter(
            name="p_x",
            default=p_x_default,
            space=gym.spaces.Box(low=p_x_low, high=p_x_high, dtype=np.float64),
            interface="learnable",
            end_stages=state_end_stages,
        ),
        AcadosParameter(
            name="p_u",
            default=p_u_default,
            space=gym.spaces.Box(low=p_u_low, high=p_u_high, dtype=np.float64),
            interface="learnable",
            end_stages=ctrl_end_stages,
        ),
    ]


def get_learnable_param_dim(N_horizon: int, param_interface: QuadrotorAcadosParamInterface) -> int:
    """Get total dimension of learnable parameters.

    Args:
        N_horizon: Number of MPC horizon steps.
        param_interface: "global" or "stagewise".

    Returns:
        Total number of learnable parameters.
    """
    if param_interface == "global":
        return Q_STATE_SIZE + Q_CTRL_SIZE + P_X_SIZE + P_U_SIZE  # 43
    else:
        n_state_stages = N_horizon + 1
        n_ctrl_stages = N_horizon
        return (Q_STATE_SIZE + P_X_SIZE) * n_state_stages + (Q_CTRL_SIZE + P_U_SIZE) * n_ctrl_stages


def define_so_rpy_euler_dynamics(dt: float, drone_model: str = "cf2x_L250") -> tuple[ca.SX, ca.SX, ca.SX]:
    """Define discrete quadrotor dynamics using so_rpy Euler model with SX symbols.

    Uses the fitted so_rpy Euler model which has linear second-order attitude dynamics
    with Euler angle representation (no quaternion normalization needed).
    Integration is performed using explicit RK4 for improved accuracy.

    State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
    Control: [roll_cmd, pitch_cmd, yaw_cmd, thrust]

    Args:
        dt: Integration timestep [s].
        drone_model: Drone model identifier for parameter loading.

    Returns:
        Tuple of (x_next, x, u) CasADi SX expressions.
    """
    # Load drone parameters
    params = load_params("so_rpy", drone_model)
    mass = float(params["mass"])
    gravity = float(np.abs(params["gravity_vec"][2]))
    acc_coef = float(params["acc_coef"])
    cmd_f_coef = float(params["cmd_f_coef"])
    rpy_coef = np.array(params["rpy_coef"])
    rpy_rates_coef = np.array(params["rpy_rates_coef"])
    cmd_rpy_coef = np.array(params["cmd_rpy_coef"])

    # Create SX symbols for state and control
    X = ca.SX.sym("x", NX)
    U = ca.SX.sym("u", NU)

    # Extract states
    pos = X[0:3]       # [x, y, z]
    rpy = X[3:6]       # [roll, pitch, yaw]
    vel = X[6:9]       # [vx, vy, vz]
    drpy = X[9:12]     # [droll, dpitch, dyaw]

    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

    # Extract controls
    roll_cmd = U[0]
    pitch_cmd = U[1]
    yaw_cmd = U[2]
    thrust = U[3]

    # Rotation matrix (body to inertial) using drone-models utility
    R = cs_rpy2matrix(rpy)

    # Position dynamics: pos_dot = vel
    pos_dot = vel

    # RPY dynamics: rpy_dot = drpy
    rpy_dot = drpy

    # Velocity dynamics: vel_dot = R @ [0, 0, thrust_z/mass]^T + gravity_vec
    # From drone-models so_rpy: thrust_z = acc_coef + cmd_f_coef * thrust
    thrust_z = acc_coef + cmd_f_coef * thrust
    thrust_body = ca.vertcat(0, 0, thrust_z / mass)
    vel_dot = R @ thrust_body + ca.vertcat(0, 0, -gravity)

    # RPY rates dynamics (second-order linear):
    # drpy_dot = rpy_coef * rpy + rpy_rates_coef * drpy + cmd_rpy_coef * cmd_rpy
    cmd_rpy = ca.vertcat(roll_cmd, pitch_cmd, yaw_cmd)
    drpy_dot = ca.vertcat(
        rpy_coef[0] * roll + rpy_rates_coef[0] * drpy[0] + cmd_rpy_coef[0] * roll_cmd,
        rpy_coef[1] * pitch + rpy_rates_coef[1] * drpy[1] + cmd_rpy_coef[1] * pitch_cmd,
        rpy_coef[2] * yaw + rpy_rates_coef[2] * drpy[2] + cmd_rpy_coef[2] * yaw_cmd,
    )

    # Continuous dynamics
    X_dot = ca.vertcat(pos_dot, rpy_dot, vel_dot, drpy_dot)

    # RK4 integration for discretization
    p = ca.SX.sym("p_empty", 0)
    X_next = integrate_erk4(X_dot, X, U, p, dt)

    return X_next, X, U


def build_Q_matrix(q_state: ca.SX) -> ca.SX:
    """Build the Q matrix for state cost from parameter vector.

    Args:
        q_state: Parameter vector of size 12 (all diagonal).
            - Position diagonal: 3 (x, y, z)
            - RPY diagonal: 3 (roll, pitch, yaw)
            - Velocity diagonal: 3 (vx, vy, vz)
            - RPY rates diagonal: 3 (droll, dpitch, dyaw)

    Returns:
        12x12 state cost matrix Q (diagonal).
    """
    Q = ca.SX.zeros(NX, NX)

    # All diagonal costs for Euler state representation
    for i in range(NX):
        Q[i, i] = q_state[i]

    return Q


def build_R_matrix(q_control: ca.SX) -> ca.SX:
    """Build the R matrix for control cost from parameter vector.

    Args:
        q_control: Parameter vector of size 4.

    Returns:
        4x4 control cost matrix R (diagonal).
    """
    R = ca.SX.zeros(NU, NU)
    R[0, 0] = q_control[0]  # roll
    R[1, 1] = q_control[1]  # pitch
    R[2, 2] = q_control[2]  # yaw
    R[3, 3] = q_control[3]  # thrust
    return R


def define_cost_expression(
    ocp: AcadosOcp,
    param_manager: AcadosParameterManager,
) -> tuple[ca.SX, ca.SX]:
    """Define external cost expressions.

    Cost structure: J = 0.5 * x^T Q x + p_x^T x + 0.5 * u^T R u + p_u^T u

    Args:
        ocp: AcadosOcp object with model defined.
        param_manager: Manager containing learnable parameters.

    Returns:
        Tuple of (stage_cost, terminal_cost) CasADi expressions.
    """
    y = ca.vertcat(ocp.model.x, ocp.model.u)
    y_e = ocp.model.x

    # Get parameters
    q_state = param_manager.get("q_state")
    q_control = param_manager.get("q_control")
    p_x = param_manager.get("p_x")
    p_u = param_manager.get("p_u")
    p = ca.vertcat(p_x, p_u)
    p_e = p_x

    # Build Q and R matrices
    Q = build_Q_matrix(q_state)
    R = build_R_matrix(q_control)

    W = ca.diagcat(Q, R)
    W_e = Q

    # Stage cost
    cost_stage = 0.5 * (y.T @ W @ y) + p.T @ y

    # Terminal cost
    cost_terminal = 0.5 * (y_e.T @ W_e @ y_e) + p_e.T @ y_e

    return cost_stage, cost_terminal


def export_parametric_ocp(
    param_manager: AcadosParameterManager,
    name: str = "quadrotor_so_rpy_euler",
    N_horizon: int = 2,
    T_horizon: float = 0.02,
    dt: float = 0.01,
    drone_model: str = "cf2x_L250",
    velocity_max: float | None = None,
    roll_pitch_max: float = 0.5,
    yaw_max: float = 0.5,
) -> AcadosOcp:
    """Export the quadrotor OCP for leap-c using so_rpy Euler dynamics.

    Args:
        param_manager: Manager containing learnable and non-learnable parameters.
        name: Model name for acados code generation.
        N_horizon: MPC horizon steps.
        T_horizon: Total horizon time [s].
        dt: Integration timestep [s].
        drone_model: Drone model identifier.
        velocity_max: Maximum velocity per axis [m/s]. None to disable.
        roll_pitch_max: Maximum roll/pitch command [rad].
        yaw_max: Maximum yaw command [rad].

    Returns:
        Configured AcadosOcp object.
    """
    ocp = AcadosOcp()

    # Solver options
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    # Assign parameters from manager
    param_manager.assign_to_ocp(ocp)

    # Model setup
    ocp.model.name = name
    ocp.dims.nx = NX
    ocp.dims.nu = NU

    # Get symbolic dynamics from so_rpy Euler model
    x_next, x, u = define_so_rpy_euler_dynamics(dt, drone_model)

    # Assign state and control symbols
    ocp.model.x = x
    ocp.model.u = u

    # Discrete dynamics
    ocp.model.disc_dyn_expr = x_next

    # Cost function
    cost_stage, cost_terminal = define_cost_expression(ocp, param_manager)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = cost_stage
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = cost_terminal

    # Initial state constraint (all zeros for Euler state)
    ocp.constraints.x0 = np.zeros(NX)

    # Load physical parameters for thrust constraints
    drone_params = load_params("so_rpy", drone_model)
    min_thrust = float(drone_params["thrust_min"]) * 4  # Per motor -> collective
    max_thrust = float(drone_params["thrust_max"]) * 4

    # Control box constraints
    ocp.constraints.lbu = np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, min_thrust])
    ocp.constraints.ubu = np.array([roll_pitch_max, roll_pitch_max, yaw_max, max_thrust])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # State box constraints (velocity)
    if velocity_max is not None:
        # Constrain velocity components: vx, vy, vz (indices 6, 7, 8)
        ocp.constraints.lbx = np.array([-velocity_max, -velocity_max, -velocity_max])
        ocp.constraints.ubx = np.array([velocity_max, velocity_max, velocity_max])
        ocp.constraints.idxbx = np.array([6, 7, 8])  # vx, vy, vz indices
        # Apply at all intermediate stages (1 to N-1)
        ocp.constraints.lbx_e = np.array([-velocity_max, -velocity_max, -velocity_max])
        ocp.constraints.ubx_e = np.array([velocity_max, velocity_max, velocity_max])
        ocp.constraints.idxbx_e = np.array([6, 7, 8])

    # Solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.print_level = 0

    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.qp_solver_cond_N = N_horizon
    ocp.solver_options.qp_solver_warm_start = 1
    
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_tol = 1e-6
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    return ocp
