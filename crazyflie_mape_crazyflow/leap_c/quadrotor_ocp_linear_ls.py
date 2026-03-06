"""Acados OCP definition for quadrotor attitude control with LINEAR_LS cost.

This module defines the optimal control problem using the LINEAR_LS cost type
which is numerically more stable for neural network integration.

The cost is: J = 0.5 * ||Vx*x + Vu*u - y_ref||_W^2

The NN outputs:
1. W (weights) - log scaled for orders of magnitude control
2. y_ref (references) - linearly scaled to physical bounds

State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw] (12D)
Control: [roll, pitch, yaw, thrust] (4D)
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
NY = NX + NU  # Output dimension for LINEAR_LS

# Parameter sizes per stage for LINEAR_LS
# W (weights): diagonal of size NY
# y_ref (reference): vector of size NY
W_SIZE = NY  # 16 (weights for combined state + control)
YREF_SIZE = NY  # 16 (reference for combined state + control)

QuadrotorAcadosParamInterface = Literal["global", "stagewise"]


def integrate_euler(f_expl: ca.SX, x: ca.SX, dt: float) -> ca.SX:
    """Integrate dynamics using the forward Euler method."""
    return x + dt * f_expl


def integrate_erk4(f_expl: ca.SX, x: ca.SX, u: ca.SX, p: ca.SX, dt: float) -> ca.SX:
    """Integrate dynamics using the explicit RK4 method."""
    ode = ca.Function("ode", [x, u, p], [f_expl])
    k1 = ode(x, u, p)
    k2 = ode(x + dt / 2 * k1, u, p)
    k3 = ode(x + dt / 2 * k2, u, p)
    k4 = ode(x + dt * k3, u, p)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def create_quadrotor_params_linear_ls(
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
    drone_params = load_params("so_rpy", drone_model)
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
    # Log bounds for each component
    w_state_min_log = np.array([-1., -1., -1., -2., -2., -2., -1., -1., -1., -1., -1., -1.])
    w_state_max_log = np.array([2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1.])
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
    # State refs: physical bounds
    yref_state_low = np.array([-6., -6., 0., -roll_pitch_max, -roll_pitch_max, -yaw_max, -5., -5., -5., -10., -10., -10.])
    yref_state_high = np.array([6., 6., 4.5, roll_pitch_max, roll_pitch_max, yaw_max, 5., 5., 5., 10., 10., 10.])
    yref_state_default = np.zeros(NX)  # Default to origin

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


def define_so_rpy_euler_dynamics(
    dt: float,
    drone_model: str = "cf2x_L250",
    mass: float | None = None,
    gravity: float | None = None,
) -> tuple[ca.SX, ca.SX, ca.SX]:
    """Define discrete quadrotor dynamics using so_rpy Euler model with SX symbols.

    Uses the fitted so_rpy Euler model which has linear second-order attitude dynamics
    with Euler angle representation (no quaternion normalization needed).
    Integration is performed using explicit RK4 for improved accuracy.

    State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
    Control: [roll_cmd, pitch_cmd, yaw_cmd, thrust]

    Args:
        dt: Integration timestep [s].
        drone_model: Drone model identifier for parameter loading.
        mass: Drone mass [kg]. None to load from drone_model.
        gravity: Gravitational acceleration [m/s^2]. None to load from drone_model.

    Returns:
        Tuple of (x_next, x, u) CasADi SX expressions.
    """
    # Load drone parameters
    params = load_params("so_rpy", drone_model)
    if mass is None:
        mass = float(params["mass"])
    if gravity is None:
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
    pos = X[0:3]
    rpy = X[3:6]
    vel = X[6:9]
    drpy = X[9:12]

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
    thrust_z = acc_coef + cmd_f_coef * thrust
    thrust_body = ca.vertcat(0, 0, thrust_z / mass)
    vel_dot = R @ thrust_body + ca.vertcat(0, 0, -gravity)

    # RPY rates dynamics (second-order linear)
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


def export_parametric_ocp_linear_ls(
    param_manager: AcadosParameterManager,
    name: str = "quadrotor_so_rpy_euler_linear_ls",
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
        dt: Integration timestep [s].
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

    # Get symbolic dynamics
    x_next, x, u = define_so_rpy_euler_dynamics(dt, drone_model, mass=mass, gravity=gravity)

    # Assign state and control symbols
    ocp.model.x = x
    ocp.model.u = u

    # Discrete dynamics
    ocp.model.disc_dyn_expr = x_next

    # === EXTERNAL Cost with LINEAR_LS structure ===
    # Use EXTERNAL cost to allow symbolic parameters (W, y_ref)
    # Cost: J = 0.5 * (y - y_ref)' W (y - y_ref) where y = [x; u]
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # Get parameters for W and y_ref
    w_state = param_manager.get("w_state")
    w_control = param_manager.get("w_control")
    yref_state = param_manager.get("yref_state")
    yref_control = param_manager.get("yref_control")

    # Build combined output and reference vectors
    y = ca.vertcat(x, u)
    y_ref = ca.vertcat(yref_state, yref_control)
    y_e = x
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

    # Initial state constraint
    ocp.constraints.x0 = np.zeros(NX)

    # Load physical parameters for thrust constraints if not provided
    if thrust_min is None or thrust_max is None:
        drone_params = load_params("so_rpy", drone_model)
        if thrust_min is None:
            thrust_min = float(drone_params["thrust_min"]) * 4
        if thrust_max is None:
            thrust_max = float(drone_params["thrust_max"]) * 4

    # Control box constraints
    ocp.constraints.lbu = np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min])
    ocp.constraints.ubu = np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # State box constraints (velocity)
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
