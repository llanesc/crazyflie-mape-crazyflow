"""Acados OCP definition for quadrotor attitude control with LINEAR_LS cost.

This module defines the optimal control problem using the LINEAR_LS cost type
which is numerically more stable for neural network integration.

The cost is: J = 0.5 * ||Vx*x + Vu*u - y_ref||_W^2

The NN outputs:
1. W (weights) - log scaled for orders of magnitude control
2. y_ref (references) - linearly scaled to physical bounds

Supports two state representations:
- Quaternion (13D): [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
- Euler (12D): [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]

Control: [roll_cmd, pitch_cmd, yaw_cmd, thrust] (4D)
"""

from typing import Literal

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from drone_models.core import load_params
from drone_models import so_rpy as dm_so_rpy, so_rpy_rotor_drag as dm_so_rpy_rotor_drag
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager

# State dimensions — quaternion representation (default)
NX = 13  # [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
NU = 4   # [roll, pitch, yaw, thrust]
NY = NX + NU  # Output dimension for LINEAR_LS

# State dimensions — Euler representation (for old runs)
NX_EULER = 12  # [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
NY_EULER = NX_EULER + NU

# Parameter sizes per stage for LINEAR_LS
# W (weights): diagonal of size NY
# y_ref (reference): vector of size NY
W_SIZE = NY  # 17 (weights for combined state + control)
YREF_SIZE = NY  # 17 (reference for combined state + control)

QuadrotorAcadosParamInterface = Literal["global", "stagewise"]


def integrate_euler(f_expl: ca.MX, x: ca.MX, u: ca.MX, dt: float) -> ca.MX:
    """Integrate continuous dynamics using forward Euler."""
    return x + dt * f_expl


def create_quadrotor_params_linear_ls(
    N_horizon: int = 2,
    param_interface: QuadrotorAcadosParamInterface = "global",
    drone_model: str = "cf2x_L250",
    mpc_model: str = "so_rpy",
    roll_pitch_max: float = 0.5,
    yaw_max: float = 0.5,
    thrust_min: float | None = None,
    thrust_max: float | None = None,
    mass: float | None = None,
    gravity: float | None = None,
) -> list[AcadosParameter]:
    """Create learnable parameters for quadrotor MPC with LINEAR_LS cost.

    Parameters are split into weights (W) and references (y_ref).
    The cost is: J = 0.5 * ||y - y_ref||_W^2 where y = [x; u]

    For LINEAR_LS, the NN outputs:
    - W: Diagonal weights (log-scaled for orders of magnitude control)
    - y_ref: Reference values (linearly scaled to physical bounds)

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
    # Load physical parameters from drone-models
    drone_params = load_params(mpc_model, drone_model)
    if mass is None:
        mass = float(drone_params["mass"])
    if gravity is None:
        gravity = float(np.abs(drone_params["gravity_vec"][2]))
    if thrust_min is None:
        thrust_min = float(drone_params["thrust_min"]) * 4
    if thrust_max is None:
        thrust_max = float(drone_params["thrust_max"]) * 4
    cmd_f_coef = float(drone_params["cmd_f_coef"])

    # Compute hover thrust
    hover_thrust = (mass * gravity) / cmd_f_coef

    state_end_stages = list(range(N_horizon + 1)) if param_interface == "stagewise" else []
    ctrl_end_stages = list(range(N_horizon)) if param_interface == "stagewise" else []

    # === Weight (W) parameter bounds ===
    # Using log scaling: W = 10^w, where w is in [min_log, max_log]
    # State: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    # Log bounds for each component
    w_state_min_log = np.array([-1., -1., -1., -2., -2., -2., -2., -1., -1., -1., -1., -1., -1.])
    w_state_max_log = np.array([2., 2., 2., 1., 1., 1., 1., 2., 2., 2., 1., 1., 1.])
    w_ctrl_min_log = np.array([-1., -1., -1., -1.])
    w_ctrl_max_log = np.array([1., 1., 1., 1.])

    # Default log values (middle of range)
    w_state_default_log = (w_state_min_log + w_state_max_log) / 2
    w_ctrl_default_log = (w_ctrl_min_log + w_ctrl_max_log) / 2

    # Convert to linear scale for bounds (since we store linear W, scale in policy)
    w_state_low = np.power(10., w_state_min_log)
    w_state_high = np.power(10., w_state_max_log)
    w_state_default = np.power(10., w_state_default_log)

    w_ctrl_low = np.power(10., w_ctrl_min_log)
    w_ctrl_high = np.power(10., w_ctrl_max_log)
    w_ctrl_default = np.power(10., w_ctrl_default_log)

    # === Reference (y_ref) parameter bounds ===
    # State refs: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    yref_state_low = np.array([-6., -6., 0., -1., -1., -1., -1., -5., -5., -5., -10., -10., -10.])
    yref_state_high = np.array([6., 6., 4.5, 1., 1., 1., 1., 5., 5., 5., 10., 10., 10.])
    yref_state_default = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])  # identity quat

    # Control refs: action bounds
    yref_ctrl_low = np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min])
    yref_ctrl_high = np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max])
    yref_ctrl_default = np.array([0., 0., 0., hover_thrust])  # Hover as default

    return [
        AcadosParameter(
            name="w_state",
            default=w_state_default,
            space=gym.spaces.Box(low=w_state_low, high=w_state_high, dtype=np.float64),
            interface="learnable",
            end_stages=state_end_stages,
        ),
        AcadosParameter(
            name="w_control",
            default=w_ctrl_default,
            space=gym.spaces.Box(low=w_ctrl_low, high=w_ctrl_high, dtype=np.float64),
            interface="learnable",
            end_stages=ctrl_end_stages,
        ),
        AcadosParameter(
            name="yref_state",
            default=yref_state_default,
            space=gym.spaces.Box(low=yref_state_low, high=yref_state_high, dtype=np.float64),
            interface="learnable",
            end_stages=state_end_stages,
        ),
        AcadosParameter(
            name="yref_control",
            default=yref_ctrl_default,
            space=gym.spaces.Box(low=yref_ctrl_low, high=yref_ctrl_high, dtype=np.float64),
            interface="learnable",
            end_stages=ctrl_end_stages,
        ),
    ]


def get_learnable_param_dim_linear_ls(N_horizon: int, param_interface: QuadrotorAcadosParamInterface) -> int:
    """Get total dimension of learnable parameters for LINEAR_LS.

    Args:
        N_horizon: Number of MPC horizon steps.
        param_interface: "global" or "stagewise".

    Returns:
        Total number of learnable parameters.
    """
    if param_interface == "global":
        # W_state + W_ctrl + yref_state + yref_ctrl
        return NX + NU + NX + NU  # 32
    else:
        n_state_stages = N_horizon + 1
        n_ctrl_stages = N_horizon
        # (W_state + yref_state) * state_stages + (W_ctrl + yref_ctrl) * ctrl_stages
        return (NX + NX) * n_state_stages + (NU + NU) * n_ctrl_stages


def export_parametric_ocp_linear_ls(
    param_manager: AcadosParameterManager,
    name: str = "quadrotor_so_rpy_linear_ls",
    N_horizon: int = 2,
    T_horizon: float = 0.02,
    dt: float = 0.01,
    drone_model: str = "cf2x_L250",
    mpc_model: str = "so_rpy",
    integrator: str = "euler",
    velocity_max: float | None = None,
    roll_pitch_max: float = 0.5,
    yaw_max: float = 0.5,
    thrust_min: float | None = None,
    thrust_max: float | None = None,
    mass: float | None = None,
    gravity: float | None = None,
) -> AcadosOcp:
    """Export the quadrotor OCP for leap-c using LINEAR_LS cost structure.

    Uses EXTERNAL cost type with LINEAR_LS structure to allow symbolic parameters:
    J = 0.5 * ||y - y_ref||_W^2 = 0.5 * sum(w_i * (y_i - yref_i)^2)

    This formulation is more numerically stable for NN integration:
    - W (weights) can be learned via log scaling
    - y_ref (references) can be learned independently

    Args:
        param_manager: Manager containing learnable and non-learnable parameters.
        name: Model name for acados code generation.
        N_horizon: MPC horizon steps.
        T_horizon: Total horizon time [s].
        dt: Integration timestep [s] for RK4 discretization.
        drone_model: Drone model identifier.
        mpc_model: Physics model ("so_rpy" or "so_rpy_rotor_drag").
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
    _VALID_MPC_MODELS = ("so_rpy", "so_rpy_rotor_drag")
    if mpc_model not in _VALID_MPC_MODELS:
        raise ValueError(f"mpc_model must be one of {_VALID_MPC_MODELS}, got '{mpc_model}'")

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
    params = load_params(mpc_model, drone_model)
    common_kwargs = dict(
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
    if mpc_model == "so_rpy":
        X_dot, X, U, _ = dm_so_rpy.symbolic_dynamics(**common_kwargs)
    else:  # so_rpy_rotor_drag
        X_dot, X, U, _ = dm_so_rpy_rotor_drag.symbolic_dynamics(
            **common_kwargs,
            thrust_time_coef=params["thrust_time_coef"],
            drag_matrix=params["drag_matrix"],
        )

    # Discretize dynamics
    if integrator == "rk4":
        f = ca.Function("f_rk4", [X, U], [X_dot])
        k1 = f(X, U)
        k2 = f(X + dt / 2 * k1, U)
        k3 = f(X + dt / 2 * k2, U)
        k4 = f(X + dt * k3, U)
        X_next = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        X_next = integrate_euler(X_dot, X, U, dt)
    ocp.model.x = X
    ocp.model.u = U
    ocp.model.disc_dyn_expr = X_next

    # === EXTERNAL Cost with LINEAR_LS structure ===
    # Cost: J = 0.5 * (y - y_ref)' W (y - y_ref) where y = [x; u]
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # Get parameters for W and y_ref
    w_state = param_manager.get("w_state")
    w_control = param_manager.get("w_control")
    yref_state = param_manager.get("yref_state")
    yref_control = param_manager.get("yref_control")

    # Build combined output and reference vectors
    y = ca.vertcat(X, U)
    y_ref = ca.vertcat(yref_state, yref_control)
    y_e = X
    y_ref_e = yref_state

    # Build diagonal weight matrices
    W = ca.diag(ca.vertcat(w_state, w_control))
    W_e = ca.diag(w_state)

    # Compute residuals
    y_res = y - y_ref
    y_res_e = y_e - y_ref_e

    # Stage cost: 0.5 * y_res' W y_res
    ocp.model.cost_expr_ext_cost = 0.5 * (y_res.T @ W @ y_res)

    # Terminal cost: 0.5 * y_res_e' W_e y_res_e
    ocp.model.cost_expr_ext_cost_e = 0.5 * (y_res_e.T @ W_e @ y_res_e)

    # Initial state constraint: identity quaternion at rest
    # State: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    ocp.constraints.x0 = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

    # Load physical parameters for thrust constraints if not provided
    if thrust_min is None:
        thrust_min = float(params["thrust_min"]) * 4
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


# =============================================================================
# Euler (12D RPY + drpy) variants — for backward compatibility with old runs
# =============================================================================


def create_quadrotor_params_linear_ls_euler(
    N_horizon: int = 2,
    param_interface: QuadrotorAcadosParamInterface = "global",
    drone_model: str = "cf2x_L250",
    mpc_model: str = "so_rpy",
    roll_pitch_max: float = 0.5,
    yaw_max: float = 0.5,
    thrust_min: float | None = None,
    thrust_max: float | None = None,
    mass: float | None = None,
    gravity: float | None = None,
) -> list[AcadosParameter]:
    """Create learnable parameters for quadrotor MPC with LINEAR_LS cost (Euler 12D state).

    Same structure as the quaternion version but with 12D state:
    [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]

    Args:
        N_horizon: Number of MPC horizon steps.
        param_interface: "global" for same params all stages, "stagewise" for varying.
        drone_model: Drone model identifier for loading physical parameters.
        mpc_model: Physics model ("so_rpy" or "so_rpy_rotor_drag").
        roll_pitch_max: Maximum roll/pitch command [rad].
        yaw_max: Maximum yaw command [rad].
        thrust_min: Minimum collective thrust [N]. None to load from drone_model.
        thrust_max: Maximum collective thrust [N]. None to load from drone_model.
        mass: Drone mass [kg]. None to load from drone_model.
        gravity: Gravitational acceleration [m/s^2]. None to load from drone_model.

    Returns:
        List of AcadosParameter objects.
    """
    drone_params = load_params(mpc_model, drone_model)
    if mass is None:
        mass = float(drone_params["mass"])
    if gravity is None:
        gravity = float(np.abs(drone_params["gravity_vec"][2]))
    if thrust_min is None:
        thrust_min = float(drone_params["thrust_min"]) * 4
    if thrust_max is None:
        thrust_max = float(drone_params["thrust_max"]) * 4
    cmd_f_coef = float(drone_params["cmd_f_coef"])

    hover_thrust = (mass * gravity) / cmd_f_coef

    state_end_stages = list(range(N_horizon + 1)) if param_interface == "stagewise" else []
    ctrl_end_stages = list(range(N_horizon)) if param_interface == "stagewise" else []

    nx = NX_EULER

    # State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
    w_state_min_log = np.array([-1., -1., -1., -2., -2., -2., -1., -1., -1., -1., -1., -1.])
    w_state_max_log = np.array([2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1.])
    w_ctrl_min_log = np.array([-1., -1., -1., -1.])
    w_ctrl_max_log = np.array([1., 1., 1., 1.])

    w_state_default_log = (w_state_min_log + w_state_max_log) / 2
    w_ctrl_default_log = (w_ctrl_min_log + w_ctrl_max_log) / 2

    w_state_low = np.power(10., w_state_min_log)
    w_state_high = np.power(10., w_state_max_log)
    w_state_default = np.power(10., w_state_default_log)

    w_ctrl_low = np.power(10., w_ctrl_min_log)
    w_ctrl_high = np.power(10., w_ctrl_max_log)
    w_ctrl_default = np.power(10., w_ctrl_default_log)

    # State refs: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
    yref_state_low = np.array([-6., -6., 0., -np.pi, -np.pi/2, -np.pi, -5., -5., -5., -10., -10., -10.])
    yref_state_high = np.array([6., 6., 4.5, np.pi, np.pi/2, np.pi, 5., 5., 5., 10., 10., 10.])
    yref_state_default = np.zeros(nx)

    # Control refs
    yref_ctrl_low = np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min])
    yref_ctrl_high = np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max])
    yref_ctrl_default = np.array([0., 0., 0., hover_thrust])

    return [
        AcadosParameter(
            name="w_state",
            default=w_state_default,
            space=gym.spaces.Box(low=w_state_low, high=w_state_high, dtype=np.float64),
            interface="learnable",
            end_stages=state_end_stages,
        ),
        AcadosParameter(
            name="w_control",
            default=w_ctrl_default,
            space=gym.spaces.Box(low=w_ctrl_low, high=w_ctrl_high, dtype=np.float64),
            interface="learnable",
            end_stages=ctrl_end_stages,
        ),
        AcadosParameter(
            name="yref_state",
            default=yref_state_default,
            space=gym.spaces.Box(low=yref_state_low, high=yref_state_high, dtype=np.float64),
            interface="learnable",
            end_stages=state_end_stages,
        ),
        AcadosParameter(
            name="yref_control",
            default=yref_ctrl_default,
            space=gym.spaces.Box(low=yref_ctrl_low, high=yref_ctrl_high, dtype=np.float64),
            interface="learnable",
            end_stages=ctrl_end_stages,
        ),
    ]


def get_learnable_param_dim_linear_ls_euler(
    N_horizon: int, param_interface: QuadrotorAcadosParamInterface
) -> int:
    """Get total dimension of learnable parameters for LINEAR_LS with Euler state.

    Args:
        N_horizon: Number of MPC horizon steps.
        param_interface: "global" or "stagewise".

    Returns:
        Total number of learnable parameters.
    """
    nx = NX_EULER
    if param_interface == "global":
        return nx + NU + nx + NU
    else:
        n_state_stages = N_horizon + 1
        n_ctrl_stages = N_horizon
        return (nx + nx) * n_state_stages + (NU + NU) * n_ctrl_stages


def export_parametric_ocp_linear_ls_euler(
    param_manager: AcadosParameterManager,
    name: str = "quadrotor_so_rpy_euler_linear_ls",
    N_horizon: int = 2,
    T_horizon: float = 0.02,
    dt: float = 0.01,
    drone_model: str = "cf2x_L250",
    mpc_model: str = "so_rpy",
    integrator: str = "rk4",
    velocity_max: float | None = None,
    roll_pitch_max: float = 0.5,
    yaw_max: float = 0.5,
    thrust_min: float | None = None,
    thrust_max: float | None = None,
    mass: float | None = None,
    gravity: float | None = None,
) -> AcadosOcp:
    """Export the quadrotor OCP using Euler (12D) state with LINEAR_LS cost.

    State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw] (12D)
    Uses SX dynamics with RK4 integration for backward compatibility with old runs.

    Args:
        param_manager: Manager containing learnable and non-learnable parameters.
        name: Model name for acados code generation.
        N_horizon: MPC horizon steps.
        T_horizon: Total horizon time [s].
        dt: Integration timestep [s] for RK4 discretization.
        drone_model: Drone model identifier.
        mpc_model: Physics model ("so_rpy" or "so_rpy_rotor_drag").
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
    from .so_rpy_dynamics_sx import (
        integrate_euler_sx,
        integrate_rk4_sx,
        symbolic_dynamics_euler_sx,
    )

    nx = NX_EULER

    _VALID_MPC_MODELS = ("so_rpy",)
    if mpc_model not in _VALID_MPC_MODELS:
        raise ValueError(
            f"Euler dynamics only supports mpc_model in {_VALID_MPC_MODELS}, got '{mpc_model}'"
        )

    ocp = AcadosOcp()

    # Solver options
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    # Assign parameters from manager
    param_manager.assign_to_ocp(ocp)

    # Model setup
    ocp.model.name = name
    ocp.dims.nx = nx
    ocp.dims.nu = NU

    # Load drone parameters and build SX dynamics
    params = load_params(mpc_model, drone_model)
    common_kwargs = dict(
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

    X_dot, X, U, _ = symbolic_dynamics_euler_sx(**common_kwargs)

    # Discretize dynamics
    if integrator == "rk4":
        X_next = integrate_rk4_sx(X_dot, X, U, dt)
    else:
        X_next = integrate_euler_sx(X_dot, X, U, dt)
    ocp.model.x = X
    ocp.model.u = U
    ocp.model.disc_dyn_expr = X_next

    # === EXTERNAL Cost with LINEAR_LS structure ===
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    w_state = param_manager.get("w_state")
    w_control = param_manager.get("w_control")
    yref_state = param_manager.get("yref_state")
    yref_control = param_manager.get("yref_control")

    y = ca.vertcat(X, U)
    y_ref = ca.vertcat(yref_state, yref_control)
    y_e = X
    y_ref_e = yref_state

    W = ca.diag(ca.vertcat(w_state, w_control))
    W_e = ca.diag(w_state)

    y_res = y - y_ref
    y_res_e = y_e - y_ref_e

    ocp.model.cost_expr_ext_cost = 0.5 * (y_res.T @ W @ y_res)
    ocp.model.cost_expr_ext_cost_e = 0.5 * (y_res_e.T @ W_e @ y_res_e)

    # Initial state constraint: level flight at origin
    ocp.constraints.x0 = np.zeros(nx)

    # Load physical parameters for thrust constraints if not provided
    if thrust_min is None:
        thrust_min = float(params["thrust_min"]) * 4
    if thrust_max is None:
        thrust_max = float(params["thrust_max"]) * 4

    # Control box constraints
    ocp.constraints.lbu = np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min])
    ocp.constraints.ubu = np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # State box constraints (velocity): vx, vy, vz at indices 6, 7, 8 for Euler state
    if velocity_max is not None:
        ocp.constraints.lbx = np.array([-velocity_max, -velocity_max, -velocity_max])
        ocp.constraints.ubx = np.array([velocity_max, velocity_max, velocity_max])
        ocp.constraints.idxbx = np.array([6, 7, 8])
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
