"""Acados OCP definition for quadrotor attitude control with so_rpy dynamics.

This module defines the optimal control problem for a quadrotor using attitude
commands (roll, pitch, yaw, thrust) with the fitted so_rpy dynamics model
from drone-models. The cost function follows the standard quadratic form:

    J = 0.5 * x^T Q x + p_x^T x + 0.5 * u^T R u + p_u^T u

State: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz] (13D)
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
from drone_models.so_rpy import symbolic_dynamics
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

# State dimensions
NX = 13  # [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
NU = 4   # [roll, pitch, yaw, thrust]

# Parameter sizes per stage
# Position (3) + Quaternion (4) + Velocity (3) + Angular velocity (3) = 13
Q_STATE_SIZE = 13
Q_CTRL_SIZE = 4     # Control diagonal
P_X_SIZE = 13       # State linear
P_U_SIZE = 4        # Control linear

# Combined sizes
Q_SIZE = Q_STATE_SIZE + Q_CTRL_SIZE  # 17
P_SIZE = P_X_SIZE + P_U_SIZE         # 17

QuadrotorAcadosParamInterface = Literal["global", "stagewise"]

def integrate_euler(f_expl: ca.MX, x: ca.MX, u: ca.MX, dt: float) -> ca.MX:
    """Integrate continuous dynamics using forward Euler."""
    return x + dt * f_expl


def create_quadrotor_params_qp(
    N_horizon: int = 2,
    param_interface: QuadrotorAcadosParamInterface = "global",
    drone_model: str = "cf2x_L250",
    roll_pitch_max: float = 0.5,
    yaw_max: float = 0.5,
    thrust_min: float | None = None,
    thrust_max: float | None = None,
    mass: float | None = None,
    gravity: float | None = None,
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
        roll_pitch_max: Maximum roll/pitch command [rad].
        yaw_max: Maximum yaw command [rad].
        thrust_min: Minimum collective thrust [N]. None to load from drone_model.
        thrust_max: Maximum collective thrust [N]. None to load from drone_model.
        mass: Drone mass [kg]. None to load from drone_model.
        gravity: Gravitational acceleration [m/s^2]. None to load from drone_model.

    Returns:
        List of AcadosParameter objects.
    """
    # Load physical parameters from drone-models (for fallback values)
    drone_params = load_params("so_rpy", drone_model)
    # Use provided values or fall back to drone_model values
    if mass is None:
        mass = float(drone_params["mass"])
    if gravity is None:
        gravity = float(np.abs(drone_params["gravity_vec"][2]))
    if thrust_min is None:
        thrust_min = float(drone_params["thrust_min"]) * 4  # Per motor -> collective
    if thrust_max is None:
        thrust_max = float(drone_params["thrust_max"]) * 4
    cmd_f_coef = float(drone_params["cmd_f_coef"])

    state_end_stages = list(range(N_horizon + 1)) if param_interface == "stagewise" else []
    ctrl_end_stages = list(range(N_horizon)) if param_interface == "stagewise" else []

    # Cost penalties for each state/control component
    # State: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    # Position (3) + Quaternion (4) + Velocity (3) + Angular velocity (3) = 13
    state_penalty = np.array([50., 50., 100., 1., 1., 1., 1., 10., 10., 10., 5., 5., 5.])
    # Control: [roll_cmd, pitch_cmd, yaw_cmd, thrust]
    control_penalty = np.array([1., 1., 1., 5.])

    # State scale (bounds for linear cost calculation)
    state_scale = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    # Action mean and scale (computed from control bounds)
    # Roll/pitch: [-roll_pitch_max, roll_pitch_max] -> mean=0, scale=roll_pitch_max
    # Yaw: [-yaw_max, yaw_max] -> mean=0, scale=yaw_max
    # Thrust: [thrust_min, thrust_max] -> mean=(min+max)/2, scale=(max-min)/2
    thrust_mean = (thrust_min + thrust_max) / 2.0
    thrust_scale = (thrust_max - thrust_min) / 2.0
    action_mean = np.array([0.0, 0.0, 0.0, thrust_mean])
    action_scale = np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_scale])

    # Scaling constants - must match quadrotor_policy.py for proper parameter bounds
    q_nom = np.concatenate((state_penalty, control_penalty))
    px = np.sqrt(state_penalty)
    pu = np.sqrt(control_penalty)
    p_nom = np.concatenate((px, pu)) * np.concatenate((state_scale, action_scale))
    range_Q = 2. * q_nom
    range_p = 2. * p_nom
    # Hover thrust command = (mass * gravity) / cmd_f_coef in MPC dynamics
    range_p_t = 2 * range_Q[-1] / 2 * (mass * gravity) / cmd_f_coef
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
    # Thrust linear term should be negative (bias toward hover thrust)
    # For cost J = 0.5*q*u² + p*u, optimal is u* = -p/q
    # With p = -q*m*g, we get u* = m*g (hover thrust)
    p_u_low[-1] = -(range_p_t + epsilon)
    p_u_high[-1] = -epsilon
    p_u_default = np.full((P_U_SIZE,), 0.0)
    p_u_default[-1] = -(range_p_t / 2 + epsilon)  # Negative, biasing toward hover

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


def get_learnable_param_dim_qp(N_horizon: int, param_interface: QuadrotorAcadosParamInterface) -> int:
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


def build_Q_matrix(q_state: ca.MX) -> ca.MX:
    """Build the Q matrix for state cost from parameter vector.

    Args:
        q_state: Parameter vector of size 13 (all diagonal).
            - Position diagonal: 3 (x, y, z)
            - Quaternion diagonal: 4 (qx, qy, qz, qw)
            - Velocity diagonal: 3 (vx, vy, vz)
            - Angular velocity diagonal: 3 (wx, wy, wz)

    Returns:
        13x13 state cost matrix Q (diagonal).
    """
    Q = ca.MX.zeros(NX, NX)
    for i in range(NX):
        Q[i, i] = q_state[i]
    return Q


def build_R_matrix(q_control: ca.MX) -> ca.MX:
    """Build the R matrix for control cost from parameter vector.

    Args:
        q_control: Parameter vector of size 4.

    Returns:
        4x4 control cost matrix R (diagonal).
    """
    R = ca.MX.zeros(NU, NU)
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


def export_parametric_ocp_qp(
    param_manager: AcadosParameterManager,
    name: str = "quadrotor_so_rpy",
    N_horizon: int = 2,
    T_horizon: float = 0.02,
    dt: float = 0.01,
    drone_model: str = "cf2x_L250",
    velocity_max: float | None = None,
    roll_pitch_max: float = 0.5,
    yaw_max: float = 0.5,
    thrust_min: float | None = None,
    thrust_max: float | None = None,
    mass: float | None = None,
    gravity: float | None = None,
) -> AcadosOcp:
    """Export the quadrotor OCP for leap-c using so_rpy dynamics.

    Args:
        param_manager: Manager containing learnable and non-learnable parameters.
        name: Model name for acados code generation.
        N_horizon: MPC horizon steps.
        T_horizon: Total horizon time [s].
        dt: Integration timestep [s] for RK4 discretization.
        drone_model: Drone model identifier.
        velocity_max: Maximum velocity per axis [m/s]. None to disable.
        roll_pitch_max: Maximum roll/pitch command [rad].
        yaw_max: Maximum yaw command [rad].
        thrust_min: Minimum collective thrust [N]. None to load from drone_model.
        thrust_max: Maximum collective thrust [N]. None to load from drone_model.
        mass: Drone mass [kg]. None to load from drone_model.
        gravity: Gravitational acceleration [m/s^2]. None to load from drone_model.

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

    # Load drone parameters and build symbolic dynamics via drone-models
    params = load_params("so_rpy", drone_model)
    X_dot, X, U, _ = symbolic_dynamics(
        model_rotor_vel=False,
        mass=float(params["mass"]),
        gravity_vec=params["gravity_vec"],
        J=params["J"],
        J_inv=params["J_inv"],
        acc_coef=params["acc_coef"],
        cmd_f_coef=params["cmd_f_coef"],
        rpy_coef=params["rpy_coef"],
        rpy_rates_coef=params["rpy_rates_coef"],
        cmd_rpy_coef=params["cmd_rpy_coef"],
    )

    # Discretize with forward Euler and assign to model
    X_next = integrate_euler(X_dot, X, U, dt)
    ocp.model.x = X
    ocp.model.u = U
    ocp.model.disc_dyn_expr = X_next

    # Cost function
    cost_stage, cost_terminal = define_cost_expression(ocp, param_manager)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = cost_stage
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = cost_terminal

    # Initial state constraint: identity quaternion at rest
    # State: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    ocp.constraints.x0 = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

    # Load physical parameters for thrust constraints if not provided
    if thrust_min is None:
        thrust_min = float(params["thrust_min"]) * 4  # Per motor -> collective
    if thrust_max is None:
        thrust_max = float(params["thrust_max"]) * 4

    # Control box constraints
    ocp.constraints.lbu = np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min])
    ocp.constraints.ubu = np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # State box constraints (velocity): vx, vy, vz at indices 7, 8, 9
    if velocity_max is not None:
        ocp.constraints.lbx = np.array([-velocity_max, -velocity_max, -velocity_max])
        ocp.constraints.ubx = np.array([velocity_max, velocity_max, velocity_max])
        ocp.constraints.idxbx = np.array([7, 8, 9])
        ocp.constraints.lbx_e = np.array([-velocity_max, -velocity_max, -velocity_max])
        ocp.constraints.ubx_e = np.array([velocity_max, velocity_max, velocity_max])
        ocp.constraints.idxbx_e = np.array([7, 8, 9])

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
