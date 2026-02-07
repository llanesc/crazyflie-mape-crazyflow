"""Benchmark: Current OCP vs Variable-Scaled OCP for MINSTEP error reduction.

Tests whether variable scaling improves solver conditioning and reduces
ACADOS_MINSTEP (status 3) errors when cost parameters span a wide range.
"""

import time
from copy import deepcopy
from pathlib import Path

import casadi as ca
import numpy as np
import torch

from acados_template import AcadosOcp
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
)
from drone_models.core import load_params
from drone_models.utils.rotation import cs_rpy2matrix
from leap_c.ocp.acados.data import AcadosOcpSolverInput
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch

from crazyflie_mape_crazyflow.leap_c.quadrotor_ocp import (
    NX,
    NU,
    Q_STATE_SIZE,
    Q_CTRL_SIZE,
    P_X_SIZE,
    P_U_SIZE,
    QuadrotorAcadosParamInterface,
    integrate_erk4,
    create_quadrotor_params,
)


# --- Scaling constants ---
_STATE_PENALTY = np.array([50., 50., 100., 1., 1., 1., 10., 10., 10., 5., 5., 5.])
_CONTROL_PENALTY = np.array([1., 1., 1., 50.])

STATE_SCALE = 1.0 / np.sqrt(_STATE_PENALTY)
CTRL_SCALE = 1.0 / np.sqrt(_CONTROL_PENALTY)
STATE_SCALE_INV = np.sqrt(_STATE_PENALTY)
CTRL_SCALE_INV = np.sqrt(_CONTROL_PENALTY)


# --- Scaled dynamics ---
def _physical_xdot(x_phys, u_phys, params_dict):
    """Compute physical x_dot given physical state and control (as CasADi expressions)."""
    mass = params_dict["mass"]
    gravity = params_dict["gravity"]
    acc_coef = params_dict["acc_coef"]
    cmd_f_coef = params_dict["cmd_f_coef"]
    rpy_coef = params_dict["rpy_coef"]
    rpy_rates_coef = params_dict["rpy_rates_coef"]
    cmd_rpy_coef = params_dict["cmd_rpy_coef"]

    pos = x_phys[0:3]
    rpy = x_phys[3:6]
    vel = x_phys[6:9]
    drpy = x_phys[9:12]
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

    roll_cmd = u_phys[0]
    pitch_cmd = u_phys[1]
    yaw_cmd = u_phys[2]
    thrust = u_phys[3]

    R = cs_rpy2matrix(rpy)
    pos_dot = vel
    rpy_dot = drpy
    thrust_z = acc_coef + cmd_f_coef * thrust
    thrust_body = ca.vertcat(0, 0, thrust_z / mass)
    vel_dot = R @ thrust_body + ca.vertcat(0, 0, -gravity)
    drpy_dot = ca.vertcat(
        rpy_coef[0] * roll + rpy_rates_coef[0] * drpy[0] + cmd_rpy_coef[0] * roll_cmd,
        rpy_coef[1] * pitch + rpy_rates_coef[1] * drpy[1] + cmd_rpy_coef[1] * pitch_cmd,
        rpy_coef[2] * yaw + rpy_rates_coef[2] * drpy[2] + cmd_rpy_coef[2] * yaw_cmd,
    )
    return ca.vertcat(pos_dot, rpy_dot, vel_dot, drpy_dot)


def define_scaled_dynamics(dt: float, drone_model: str = "cf2x_L250"):
    """Define discrete dynamics in scaled variable space.

    Decision variables are x_bar and u_bar (scaled).
    Internally converts to physical, computes dynamics via RK4, converts back.
    """
    params = load_params("so_rpy", drone_model)
    params_dict = {
        "mass": float(params["mass"]),
        "gravity": float(np.abs(params["gravity_vec"][2])),
        "acc_coef": float(params["acc_coef"]),
        "cmd_f_coef": float(params["cmd_f_coef"]),
        "rpy_coef": np.array(params["rpy_coef"]),
        "rpy_rates_coef": np.array(params["rpy_rates_coef"]),
        "cmd_rpy_coef": np.array(params["cmd_rpy_coef"]),
    }

    # Scaled decision variables
    X_bar = ca.SX.sym("x", NX)
    U_bar = ca.SX.sym("u", NU)

    # Scaling as CasADi constants
    state_scale_sx = ca.SX(STATE_SCALE.reshape(-1, 1))
    ctrl_scale_sx = ca.SX(CTRL_SCALE.reshape(-1, 1))
    state_scale_inv_sx = ca.SX(STATE_SCALE_INV.reshape(-1, 1))

    # Convert to physical
    X_phys = X_bar * state_scale_sx
    U_phys = U_bar * ctrl_scale_sx

    # RK4 integration in physical space (inline, no CasADi Function)
    k1 = _physical_xdot(X_phys, U_phys, params_dict)
    k2 = _physical_xdot(X_phys + dt / 2 * k1, U_phys, params_dict)
    k3 = _physical_xdot(X_phys + dt / 2 * k2, U_phys, params_dict)
    k4 = _physical_xdot(X_phys + dt * k3, U_phys, params_dict)
    X_next_phys = X_phys + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Convert back to scaled space
    X_next_bar = X_next_phys * state_scale_inv_sx

    return X_next_bar, X_bar, U_bar


# --- Scaled cost expression ---
def define_scaled_cost_expression(ocp: AcadosOcp, param_manager: AcadosParameterManager):
    """Define cost in terms of scaled variables, applying cost on physical values."""
    x_bar = ocp.model.x
    u_bar = ocp.model.u

    # Convert to physical for cost computation
    state_scale_sx = ca.SX(STATE_SCALE.reshape(-1, 1))
    ctrl_scale_sx = ca.SX(CTRL_SCALE.reshape(-1, 1))

    x_phys = x_bar * state_scale_sx
    u_phys = u_bar * ctrl_scale_sx

    # Get parameters (cost weights operate on physical variables)
    q_state = param_manager.get("q_state")
    q_control = param_manager.get("q_control")
    p_x = param_manager.get("p_x")
    p_u = param_manager.get("p_u")

    # Build diagonal Q and R
    Q = ca.SX.zeros(NX, NX)
    for i in range(NX):
        Q[i, i] = q_state[i]
    R = ca.SX.zeros(NU, NU)
    for i in range(NU):
        R[i, i] = q_control[i]

    # Stage cost on physical variables
    y_phys = ca.vertcat(x_phys, u_phys)
    p_vec = ca.vertcat(p_x, p_u)
    W = ca.diagcat(Q, R)
    cost_stage = 0.5 * (y_phys.T @ W @ y_phys) + p_vec.T @ y_phys

    # Terminal cost
    cost_terminal = 0.5 * (x_phys.T @ Q @ x_phys) + p_x.T @ x_phys

    return cost_stage, cost_terminal


# --- Scaled OCP export ---
def export_scaled_ocp(
    param_manager: AcadosParameterManager,
    name: str = "quadrotor_scaled",
    N_horizon: int = 2,
    T_horizon: float = 0.02,
    dt: float = 0.01,
    drone_model: str = "cf2x_L250",
    velocity_max: float | None = None,
    roll_pitch_max: float = 0.5,
    yaw_max: float = 0.5,
) -> AcadosOcp:
    """Export scaled OCP where decision variables are in normalized space."""
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = T_horizon

    param_manager.assign_to_ocp(ocp)

    ocp.model.name = name
    ocp.dims.nx = NX
    ocp.dims.nu = NU

    # Scaled dynamics
    x_next, x, u = define_scaled_dynamics(dt, drone_model)
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.disc_dyn_expr = x_next

    # Cost
    cost_stage, cost_terminal = define_scaled_cost_expression(ocp, param_manager)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = cost_stage
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = cost_terminal

    # Initial state constraint (scaled: x0_bar = x0_phys * STATE_SCALE_INV)
    ocp.constraints.x0 = np.zeros(NX)  # zero scales to zero

    # Load physical parameters for thrust constraints
    drone_params = load_params("so_rpy", drone_model)
    thrust_min = float(drone_params["thrust_min"]) * 4
    thrust_max = float(drone_params["thrust_max"]) * 4

    # Control bounds in SCALED space
    lbu_phys = np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min])
    ubu_phys = np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max])
    ocp.constraints.lbu = lbu_phys * CTRL_SCALE_INV
    ocp.constraints.ubu = ubu_phys * CTRL_SCALE_INV
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Velocity constraints in scaled space
    if velocity_max is not None:
        vel_scale_inv = STATE_SCALE_INV[6:9]
        ocp.constraints.lbx = np.array([-velocity_max, -velocity_max, -velocity_max]) * vel_scale_inv
        ocp.constraints.ubx = np.array([velocity_max, velocity_max, velocity_max]) * vel_scale_inv
        ocp.constraints.idxbx = np.array([6, 7, 8])
        ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
        ocp.constraints.ubx_e = ocp.constraints.ubx.copy()
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

    # Regularization safety net (can be toggled for testing)
    ocp.solver_options.regularize_method = "PROJECT"

    return ocp


def export_scaled_ocp_no_reg(
    param_manager: AcadosParameterManager,
    name: str = "quadrotor_scaled_noreg",
    N_horizon: int = 2,
    T_horizon: float = 0.02,
    dt: float = 0.01,
    drone_model: str = "cf2x_L250",
    velocity_max: float | None = None,
    roll_pitch_max: float = 0.5,
    yaw_max: float = 0.5,
) -> AcadosOcp:
    """Scaled OCP WITHOUT PROJECT regularization."""
    ocp = export_scaled_ocp(
        param_manager, name, N_horizon, T_horizon, dt,
        drone_model, velocity_max, roll_pitch_max, yaw_max
    )
    ocp.solver_options.regularize_method = "NO_REGULARIZE"
    return ocp


# --- Initializers ---
class UnscaledHoverInitializer(AcadosDiffMpcInitializer):
    """Initializer for unscaled (current) OCP."""

    def __init__(self, ocp: AcadosOcp, mass: float, gravity: float):
        self.default_iterate = ocp.create_default_initial_iterate().flatten()
        self.N = ocp.solver_options.N_horizon
        self.nx = ocp.dims.nx
        self.nu = ocp.dims.nu
        self.hover_thrust = mass * gravity
        self.hover_u = np.zeros(self.nu)
        self.hover_u[-1] = self.hover_thrust
        self._hover_u_tiled = np.tile(self.hover_u, self.N)

    def single_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedIterate:
        iterate = deepcopy(self.default_iterate)
        x0 = solver_input.x0.flatten()
        iterate.x = np.tile(x0, self.N + 1)
        iterate.u = self._hover_u_tiled.copy()
        return iterate

    def batch_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedBatchIterate:
        B = solver_input.batch_size
        x_batch = np.tile(solver_input.x0, (1, self.N + 1))
        u_batch = np.tile(self._hover_u_tiled, (B, 1))
        z_size = self.default_iterate.z.size
        sl_size = self.default_iterate.sl.size
        su_size = self.default_iterate.su.size
        pi_size = self.default_iterate.pi.size
        lam_size = self.default_iterate.lam.size
        return AcadosOcpFlattenedBatchIterate(
            x=x_batch, u=u_batch,
            z=np.zeros((B, z_size)) if z_size > 0 else np.zeros((B, 0)),
            sl=np.zeros((B, sl_size)) if sl_size > 0 else np.zeros((B, 0)),
            su=np.zeros((B, su_size)) if su_size > 0 else np.zeros((B, 0)),
            pi=np.zeros((B, pi_size)) if pi_size > 0 else np.zeros((B, 0)),
            lam=np.zeros((B, lam_size)) if lam_size > 0 else np.zeros((B, 0)),
            N_batch=B,
        )


class ScaledHoverInitializer(AcadosDiffMpcInitializer):
    """Initializer for scaled OCP (provides iterates in scaled space)."""

    def __init__(self, ocp: AcadosOcp, mass: float, gravity: float):
        self.default_iterate = ocp.create_default_initial_iterate().flatten()
        self.N = ocp.solver_options.N_horizon
        self.nx = ocp.dims.nx
        self.nu = ocp.dims.nu
        # Hover control in scaled space
        hover_u_phys = np.zeros(self.nu)
        hover_u_phys[-1] = mass * gravity
        self.hover_u_scaled = hover_u_phys * CTRL_SCALE_INV
        self._hover_u_tiled = np.tile(self.hover_u_scaled, self.N)

    def single_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedIterate:
        iterate = deepcopy(self.default_iterate)
        # x0 from solver_input is already in scaled space (planner pre-scales)
        x0 = solver_input.x0.flatten()
        iterate.x = np.tile(x0, self.N + 1)
        iterate.u = self._hover_u_tiled.copy()
        return iterate

    def batch_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedBatchIterate:
        B = solver_input.batch_size
        x_batch = np.tile(solver_input.x0, (1, self.N + 1))
        u_batch = np.tile(self._hover_u_tiled, (B, 1))
        z_size = self.default_iterate.z.size
        sl_size = self.default_iterate.sl.size
        su_size = self.default_iterate.su.size
        pi_size = self.default_iterate.pi.size
        lam_size = self.default_iterate.lam.size
        return AcadosOcpFlattenedBatchIterate(
            x=x_batch, u=u_batch,
            z=np.zeros((B, z_size)) if z_size > 0 else np.zeros((B, 0)),
            sl=np.zeros((B, sl_size)) if sl_size > 0 else np.zeros((B, 0)),
            su=np.zeros((B, su_size)) if su_size > 0 else np.zeros((B, 0)),
            pi=np.zeros((B, pi_size)) if pi_size > 0 else np.zeros((B, 0)),
            lam=np.zeros((B, lam_size)) if lam_size > 0 else np.zeros((B, 0)),
            N_batch=B,
        )


# --- Test data generation ---
def generate_test_states(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random initial states (physical space)."""
    states = np.zeros((n_samples, NX))
    # Position: +/- 1m
    states[:, 0:3] = rng.uniform(-1.0, 1.0, (n_samples, 3))
    # RPY: +/- 0.3 rad
    states[:, 3:6] = rng.uniform(-0.3, 0.3, (n_samples, 3))
    # Velocity: +/- 1 m/s
    states[:, 6:9] = rng.uniform(-1.0, 1.0, (n_samples, 3))
    # RPY rates: +/- 1 rad/s
    states[:, 9:12] = rng.uniform(-1.0, 1.0, (n_samples, 3))
    return states


def generate_test_params(n_samples: int, N_horizon: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random cost parameters spanning the full learnable range.

    Uses the same scaling as the policy network to generate realistic parameters
    that can trigger ill-conditioning.
    """
    state_penalty = np.array([50., 50., 100., 1., 1., 1., 10., 10., 10., 5., 5., 5.])
    control_penalty = np.array([1., 1., 1., 50.])

    drone_params = load_params("so_rpy", "cf2x_L250")
    mass = float(drone_params["mass"])
    gravity = float(np.abs(drone_params["gravity_vec"][2]))

    n_state_stages = N_horizon + 1
    n_ctrl_stages = N_horizon
    epsilon = 0.01

    range_p_t = 2 * 2 * control_penalty[-1] / 2 * mass * gravity

    all_params = []
    for _ in range(n_samples):
        # Random sigmoid outputs in [0, 1]
        q_state_raw = rng.uniform(0, 1, (n_state_stages, Q_STATE_SIZE))
        q_ctrl_raw = rng.uniform(0, 1, (n_ctrl_stages, Q_CTRL_SIZE))
        p_x_raw = rng.uniform(0, 1, (n_state_stages, P_X_SIZE))
        p_u_raw = rng.uniform(0, 1, (n_ctrl_stages, P_U_SIZE))

        # Scale (same as policy._scale_parameters)
        q_state_scaled = q_state_raw * 2 * state_penalty + epsilon
        q_ctrl_scaled = q_ctrl_raw * 2 * control_penalty + epsilon

        px = np.sqrt(state_penalty)
        pu = np.sqrt(control_penalty)
        state_scale = np.ones(12)
        action_scale = np.array([0.5, 0.5, 0.1, (0.48 - 0.0513) / 2])

        p_x_scaled = (p_x_raw - 0.5) * 2 * px * state_scale
        p_att = (p_u_raw[:, :3] - 0.5) * 2 * pu[:3] * action_scale[:3]
        p_t = -(p_u_raw[:, 3:4] * range_p_t + epsilon)
        p_u_scaled = np.concatenate([p_att, p_t], axis=-1)

        # Flatten and concatenate
        param = np.concatenate([
            q_state_scaled.flatten(),
            q_ctrl_scaled.flatten(),
            p_x_scaled.flatten(),
            p_u_scaled.flatten(),
        ])
        all_params.append(param)

    return np.array(all_params)


def generate_max_condition_params(n_samples: int, N_horizon: int, rng: np.random.Generator) -> np.ndarray:
    """Generate params that maximize cost Hessian condition number.

    Sets position weights to maximum (200) and attitude weights to minimum (epsilon).
    This creates a condition number of ~2000x which is what the plan identifies as problematic.
    """
    state_penalty = np.array([50., 50., 100., 1., 1., 1., 10., 10., 10., 5., 5., 5.])
    control_penalty = np.array([1., 1., 1., 50.])

    drone_params = load_params("so_rpy", "cf2x_L250")
    mass = float(drone_params["mass"])
    gravity = float(np.abs(drone_params["gravity_vec"][2]))

    n_state_stages = N_horizon + 1
    n_ctrl_stages = N_horizon
    epsilon = 0.01

    range_p_t = 2 * 2 * control_penalty[-1] / 2 * mass * gravity

    all_params = []
    for _ in range(n_samples):
        # Max condition: position weights at max, attitude at min
        q_state_raw = np.zeros((n_state_stages, Q_STATE_SIZE))
        # Position (0:3) at max: raw=1.0 -> penalty = 2*50 + eps = 100
        q_state_raw[:, 0:3] = 0.99
        # Z position at max: raw=1.0 -> penalty = 2*100 + eps = 200
        q_state_raw[:, 2] = 0.99
        # Attitude (3:6) at min: raw=0.01 -> penalty = 0.02*1 + eps ≈ 0.03
        q_state_raw[:, 3:6] = 0.01

        # Velocity at max, rates at min
        q_state_raw[:, 6:9] = 0.99   # vel: 2*10 + eps = 20
        q_state_raw[:, 9:12] = 0.01  # rates: 0.02*5 + eps = 0.11

        # Control: attitude cmd at min, thrust at max
        q_ctrl_raw = np.zeros((n_ctrl_stages, Q_CTRL_SIZE))
        q_ctrl_raw[:, :3] = 0.01   # att cmd: 0.02*1 + eps ≈ 0.03
        q_ctrl_raw[:, 3] = 0.99    # thrust: 2*50 + eps = 100

        # Large linear costs to push solution away from origin
        p_x_raw = rng.uniform(0.0, 1.0, (n_state_stages, P_X_SIZE))
        p_u_raw = rng.uniform(0.0, 1.0, (n_ctrl_stages, P_U_SIZE))

        # Scale
        q_state_scaled = q_state_raw * 2 * state_penalty + epsilon
        q_ctrl_scaled = q_ctrl_raw * 2 * control_penalty + epsilon

        px = np.sqrt(state_penalty)
        pu = np.sqrt(control_penalty)
        state_scale = np.ones(12)
        action_scale = np.array([0.5, 0.5, 0.1, (0.48 - 0.0513) / 2])

        p_x_scaled = (p_x_raw - 0.5) * 2 * px * state_scale
        p_att = (p_u_raw[:, :3] - 0.5) * 2 * pu[:3] * action_scale[:3]
        p_t = -(p_u_raw[:, 3:4] * range_p_t + epsilon)
        p_u_scaled = np.concatenate([p_att, p_t], axis=-1)

        param = np.concatenate([
            q_state_scaled.flatten(),
            q_ctrl_scaled.flatten(),
            p_x_scaled.flatten(),
            p_u_scaled.flatten(),
        ])
        all_params.append(param)

    return np.array(all_params)


def generate_extreme_params(n_samples: int, N_horizon: int, rng: np.random.Generator) -> np.ndarray:
    """Generate extreme cost parameters that maximize ill-conditioning.

    These use values near the bounds to create maximum condition number ratio.
    """
    state_penalty = np.array([50., 50., 100., 1., 1., 1., 10., 10., 10., 5., 5., 5.])
    control_penalty = np.array([1., 1., 1., 50.])

    drone_params = load_params("so_rpy", "cf2x_L250")
    mass = float(drone_params["mass"])
    gravity = float(np.abs(drone_params["gravity_vec"][2]))

    n_state_stages = N_horizon + 1
    n_ctrl_stages = N_horizon
    epsilon = 0.01

    range_p_t = 2 * 2 * control_penalty[-1] / 2 * mass * gravity

    all_params = []
    for _ in range(n_samples):
        # Use extreme values (near 0 or near 1) to maximize condition number
        q_state_raw = rng.choice([0.01, 0.99], size=(n_state_stages, Q_STATE_SIZE))
        q_ctrl_raw = rng.choice([0.01, 0.99], size=(n_ctrl_stages, Q_CTRL_SIZE))
        p_x_raw = rng.choice([0.01, 0.99], size=(n_state_stages, P_X_SIZE))
        p_u_raw = rng.choice([0.01, 0.99], size=(n_ctrl_stages, P_U_SIZE))

        q_state_scaled = q_state_raw * 2 * state_penalty + epsilon
        q_ctrl_scaled = q_ctrl_raw * 2 * control_penalty + epsilon

        px = np.sqrt(state_penalty)
        pu = np.sqrt(control_penalty)
        state_scale = np.ones(12)
        action_scale = np.array([0.5, 0.5, 0.1, (0.48 - 0.0513) / 2])

        p_x_scaled = (p_x_raw - 0.5) * 2 * px * state_scale
        p_att = (p_u_raw[:, :3] - 0.5) * 2 * pu[:3] * action_scale[:3]
        p_t = -(p_u_raw[:, 3:4] * range_p_t + epsilon)
        p_u_scaled = np.concatenate([p_att, p_t], axis=-1)

        param = np.concatenate([
            q_state_scaled.flatten(),
            q_ctrl_scaled.flatten(),
            p_x_scaled.flatten(),
            p_u_scaled.flatten(),
        ])
        all_params.append(param)

    return np.array(all_params)


# --- Benchmark runner ---
def run_benchmark(
    diff_mpc: AcadosDiffMpcTorch,
    states: np.ndarray,
    params: np.ndarray,
    scale_x0: bool = False,
    u_scale: np.ndarray | None = None,
    label: str = "",
    batch_size: int = 64,
    backward: bool = False,
) -> dict:
    """Run benchmark on a set of states and parameters.

    Args:
        diff_mpc: The differentiable MPC module.
        states: Physical-space states, shape (N, NX).
        params: Cost parameters, shape (N, param_dim).
        scale_x0: Whether to scale x0 to scaled space (for scaled OCP).
        label: Label for printing.
        batch_size: Batch size for each solve.
        backward: Whether to run backward pass (gradient computation).

    Returns:
        Dictionary with benchmark results.
    """
    n_total = states.shape[0]
    n_batches = (n_total + batch_size - 1) // batch_size

    all_status = []
    all_fwd_times = []
    all_bwd_times = []
    all_success_rates = []
    all_retry_rates = []
    grad_norms = []
    grad_has_nan = []
    status_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_total)
        batch_states = states[start_idx:end_idx]
        batch_params = params[start_idx:end_idx]
        actual_batch = batch_states.shape[0]

        x0 = torch.tensor(batch_states, dtype=torch.float32)
        if scale_x0:
            x0 = x0 * torch.tensor(STATE_SCALE_INV, dtype=torch.float32)

        p_global = torch.tensor(batch_params, dtype=torch.float32, requires_grad=backward)

        # Get non-learnable params
        p_stagewise = diff_mpc.param_manager.combine_non_learnable_parameter_values(
            batch_size=actual_batch
        ) if hasattr(diff_mpc, 'param_manager') else None

        # Forward pass
        t_fwd_start = time.perf_counter()
        ctx, u0, x_traj, u_traj, value = diff_mpc(
            x0=x0,
            p_global=p_global,
            p_stagewise=p_stagewise,
        )
        t_fwd = time.perf_counter() - t_fwd_start
        all_fwd_times.append(t_fwd)

        # Backward pass
        if backward:
            # Loss = sum of u0 in physical units (arbitrary differentiable scalar)
            if u_scale is not None:
                u0_phys = u0 * torch.tensor(u_scale, dtype=torch.float32)
            else:
                u0_phys = u0
            loss = u0_phys.sum()
            t_bwd_start = time.perf_counter()
            loss.backward()
            t_bwd = time.perf_counter() - t_bwd_start
            all_bwd_times.append(t_bwd)

            if p_global.grad is not None:
                gn = p_global.grad.norm().item()
                grad_norms.append(gn)
                grad_has_nan.append(bool(torch.isnan(p_global.grad).any()))
            else:
                grad_norms.append(0.0)
                grad_has_nan.append(True)

        # Collect stats
        if ctx.log:
            all_success_rates.append(ctx.log.get("success_rate", 1.0))
            all_retry_rates.append(ctx.log.get("retry_rate", 0.0))

        # Count status codes
        for s in ctx.status:
            s_int = int(s)
            if s_int in status_counts:
                status_counts[s_int] += 1
            else:
                status_counts[s_int] = status_counts.get(s_int, 0) + 1

    total_solves = n_total
    results = {
        "label": label,
        "total_solves": total_solves,
        "fwd_time": sum(all_fwd_times),
        "bwd_time": sum(all_bwd_times) if all_bwd_times else 0,
        "total_time": sum(all_fwd_times) + (sum(all_bwd_times) if all_bwd_times else 0),
        "avg_fwd_per_batch": np.mean(all_fwd_times) if all_fwd_times else 0,
        "avg_bwd_per_batch": np.mean(all_bwd_times) if all_bwd_times else 0,
        "avg_success_rate": np.mean(all_success_rates) if all_success_rates else 0,
        "avg_retry_rate": np.mean(all_retry_rates) if all_retry_rates else 0,
        "status_counts": status_counts,
        "success_count": status_counts.get(0, 0),
        "minstep_count": status_counts.get(3, 0),
        "maxiter_count": status_counts.get(2, 0),
        "qp_fail_count": status_counts.get(4, 0),
        "nan_count": status_counts.get(1, 0),
        "avg_grad_norm": np.mean(grad_norms) if grad_norms else 0,
        "nan_grad_pct": np.mean(grad_has_nan) * 100 if grad_has_nan else 0,
    }
    return results


def print_results(results: dict):
    """Print benchmark results."""
    print(f"\n{'='*60}")
    print(f"  {results['label']}")
    print(f"{'='*60}")
    print(f"  Total solves:      {results['total_solves']}")
    print(f"  Fwd time:          {results['fwd_time']:.3f} s  ({results['avg_fwd_per_batch']*1000:.2f} ms/batch)")
    print(f"  Bwd time:          {results['bwd_time']:.3f} s  ({results['avg_bwd_per_batch']*1000:.2f} ms/batch)")
    print(f"  Total time:        {results['total_time']:.3f} s")
    print(f"  Avg success rate:  {results['avg_success_rate']*100:.1f}%")
    print(f"  Avg retry rate:    {results['avg_retry_rate']*100:.1f}%")
    print(f"  ---")
    print(f"  Success (0):       {results['success_count']}")
    print(f"  MINSTEP (3):       {results['minstep_count']}")
    print(f"  MAXITER (2):       {results['maxiter_count']}")
    print(f"  QP_FAIL (4):       {results['qp_fail_count']}")
    print(f"  NaN (1):           {results['nan_count']}")
    fail_total = results['total_solves'] - results['success_count']
    print(f"  Total failures:    {fail_total} ({fail_total/results['total_solves']*100:.1f}%)")
    if results['avg_grad_norm'] > 0 or results['nan_grad_pct'] > 0:
        print(f"  ---")
        print(f"  Avg grad norm:     {results['avg_grad_norm']:.4f}")
        print(f"  NaN grads:         {results['nan_grad_pct']:.1f}%")
    print(f"{'='*60}")


# --- Main ---
def main():
    import sys
    N_HORIZON = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    DT = 0.01
    T_HORIZON = N_HORIZON * DT
    DRONE_MODEL = "cf2x_L250"
    N_BATCH_MAX = 256
    NUM_THREADS = 8
    ROLL_PITCH_MAX = 0.5
    YAW_MAX = 0.1

    N_RANDOM = 512      # Random parameter samples
    N_EXTREME = 512     # Extreme parameter samples
    BATCH_SIZE = 64

    print(f"N_HORIZON = {N_HORIZON}, DT = {DT}, T_HORIZON = {T_HORIZON}")

    rng = np.random.default_rng(42)

    print("Generating test data...")
    states_random = generate_test_states(N_RANDOM, rng)
    states_extreme = generate_test_states(N_EXTREME, rng)
    params_random = generate_test_params(N_RANDOM, N_HORIZON, rng)
    params_extreme = generate_extreme_params(N_EXTREME, N_HORIZON, rng)

    # --- Build UNSCALED (current) OCP ---
    print("\nBuilding UNSCALED (current) OCP...")
    from crazyflie_mape_crazyflow.leap_c.quadrotor_ocp import export_parametric_ocp

    params_list_unscaled = create_quadrotor_params(
        N_horizon=N_HORIZON, param_interface="stagewise", drone_model=DRONE_MODEL,
        roll_pitch_max=ROLL_PITCH_MAX, yaw_max=YAW_MAX,
    )
    pm_unscaled = AcadosParameterManager(parameters=params_list_unscaled, N_horizon=N_HORIZON)
    ocp_unscaled = export_parametric_ocp(
        param_manager=pm_unscaled,
        name="quadrotor_unscaled_bench",
        N_horizon=N_HORIZON,
        T_horizon=T_HORIZON,
        dt=DT,
        drone_model=DRONE_MODEL,
        roll_pitch_max=ROLL_PITCH_MAX,
        yaw_max=YAW_MAX,
    )

    drone_params = load_params("so_rpy", DRONE_MODEL)
    mass = float(drone_params["mass"])
    gravity = float(np.abs(drone_params["gravity_vec"][2]))

    init_unscaled = UnscaledHoverInitializer(ocp_unscaled, mass=mass, gravity=gravity)
    export_dir_unscaled = Path("/tmp/acados_bench_unscaled")
    export_dir_unscaled.mkdir(parents=True, exist_ok=True)

    diff_mpc_unscaled = AcadosDiffMpcTorch(
        ocp_unscaled,
        initializer=init_unscaled,
        export_directory=export_dir_unscaled,
        n_batch_max=N_BATCH_MAX,
        num_threads_batch_solver=NUM_THREADS,
        verbose=False,
        dtype=torch.float32,
    )

    # Store param_manager ref for the benchmark function
    diff_mpc_unscaled.param_manager = pm_unscaled

    # --- Build SCALED OCP ---
    print("Building SCALED OCP...")
    params_list_scaled = create_quadrotor_params(
        N_horizon=N_HORIZON, param_interface="stagewise", drone_model=DRONE_MODEL,
        roll_pitch_max=ROLL_PITCH_MAX, yaw_max=YAW_MAX,
    )
    pm_scaled = AcadosParameterManager(parameters=params_list_scaled, N_horizon=N_HORIZON)
    ocp_scaled = export_scaled_ocp_no_reg(
        param_manager=pm_scaled,
        name="quadrotor_scaled_bench",
        N_horizon=N_HORIZON,
        T_horizon=T_HORIZON,
        dt=DT,
        drone_model=DRONE_MODEL,
        roll_pitch_max=ROLL_PITCH_MAX,
        yaw_max=YAW_MAX,
    )

    init_scaled = ScaledHoverInitializer(ocp_scaled, mass=mass, gravity=gravity)
    export_dir_scaled = Path("/tmp/acados_bench_scaled")
    export_dir_scaled.mkdir(parents=True, exist_ok=True)

    diff_mpc_scaled = AcadosDiffMpcTorch(
        ocp_scaled,
        initializer=init_scaled,
        export_directory=export_dir_scaled,
        n_batch_max=N_BATCH_MAX,
        num_threads_batch_solver=NUM_THREADS,
        verbose=False,
        dtype=torch.float32,
    )
    diff_mpc_scaled.param_manager = pm_scaled

    # Generate max-condition params
    states_maxcond = generate_test_states(N_RANDOM, rng)
    params_maxcond = generate_max_condition_params(N_RANDOM, N_HORIZON, rng)

    # --- Run benchmarks ---
    print("\n" + "="*60)
    print("  BENCHMARK: Random Parameters")
    print("="*60)

    res_unscaled_random = run_benchmark(
        diff_mpc_unscaled, states_random, params_random,
        scale_x0=False, label="UNSCALED - Random Params", batch_size=BATCH_SIZE, backward=True
    )
    print_results(res_unscaled_random)

    res_scaled_random = run_benchmark(
        diff_mpc_scaled, states_random, params_random,
        scale_x0=True, u_scale=CTRL_SCALE, label="SCALED - Random Params", batch_size=BATCH_SIZE, backward=True
    )
    print_results(res_scaled_random)

    print("\n" + "="*60)
    print("  BENCHMARK: Extreme Parameters (random 0.01/0.99)")
    print("="*60)

    res_unscaled_extreme = run_benchmark(
        diff_mpc_unscaled, states_extreme, params_extreme,
        scale_x0=False, label="UNSCALED - Extreme Params", batch_size=BATCH_SIZE, backward=True
    )
    print_results(res_unscaled_extreme)

    res_scaled_extreme = run_benchmark(
        diff_mpc_scaled, states_extreme, params_extreme,
        scale_x0=True, u_scale=CTRL_SCALE, label="SCALED - Extreme Params", batch_size=BATCH_SIZE, backward=True
    )
    print_results(res_scaled_extreme)

    print("\n" + "="*60)
    print("  BENCHMARK: Max Condition Number (pos max, att min)")
    print("="*60)

    res_unscaled_maxcond = run_benchmark(
        diff_mpc_unscaled, states_maxcond, params_maxcond,
        scale_x0=False, label="UNSCALED - Max Condition", batch_size=BATCH_SIZE, backward=True
    )
    print_results(res_unscaled_maxcond)

    res_scaled_maxcond = run_benchmark(
        diff_mpc_scaled, states_maxcond, params_maxcond,
        scale_x0=True, u_scale=CTRL_SCALE, label="SCALED - Max Condition", batch_size=BATCH_SIZE, backward=True
    )
    print_results(res_scaled_maxcond)

    # --- Summary ---
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    all_results = [
        ("Random Params", res_unscaled_random, res_scaled_random),
        ("Extreme Params", res_unscaled_extreme, res_scaled_extreme),
        ("Max Condition", res_unscaled_maxcond, res_scaled_maxcond),
    ]

    print(f"\n  Failures:")
    print(f"  {'Scenario':<30} {'Unscaled':<15} {'Scaled':<15}")
    print(f"  {'-'*60}")
    for label, res_u, res_s in all_results:
        fail_u = res_u['total_solves'] - res_u['success_count']
        fail_s = res_s['total_solves'] - res_s['success_count']
        n = res_u['total_solves']
        print(f"  {label:<30} {fail_u}/{n:<10} {fail_s}/{n}")

    print(f"\n  Forward time:")
    print(f"  {'Scenario':<30} {'Unscaled':<15} {'Scaled':<15}")
    print(f"  {'-'*60}")
    for label, res_u, res_s in all_results:
        print(f"  {label:<30} {res_u['fwd_time']:.3f}s{'':<8} {res_s['fwd_time']:.3f}s")

    print(f"\n  Backward time:")
    print(f"  {'Scenario':<30} {'Unscaled':<15} {'Scaled':<15}")
    print(f"  {'-'*60}")
    for label, res_u, res_s in all_results:
        print(f"  {label:<30} {res_u['bwd_time']:.3f}s{'':<8} {res_s['bwd_time']:.3f}s")

    print(f"\n  Avg gradient norm:")
    print(f"  {'Scenario':<30} {'Unscaled':<15} {'Scaled':<15}")
    print(f"  {'-'*60}")
    for label, res_u, res_s in all_results:
        print(f"  {label:<30} {res_u['avg_grad_norm']:.4f}{'':<9} {res_s['avg_grad_norm']:.4f}")

    print(f"\n  NaN gradient %:")
    print(f"  {'Scenario':<30} {'Unscaled':<15} {'Scaled':<15}")
    print(f"  {'-'*60}")
    for label, res_u, res_s in all_results:
        print(f"  {label:<30} {res_u['nan_grad_pct']:.1f}%{'':<10} {res_s['nan_grad_pct']:.1f}%")

    print()


if __name__ == "__main__":
    main()
