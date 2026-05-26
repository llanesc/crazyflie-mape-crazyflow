"""Quadrotor planner using leap-c AcadosPlanner with so_rpy dynamics.

This module provides a high-level planner interface for the quadrotor MPC
that integrates with the leap-c library for differentiable MPC.

Supports two state representations:
- Quaternion (13D): [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
- Euler (12D): [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]

Control: [roll, pitch, yaw, thrust] (4D)

Supports two cost formulations:
- QP: J = 0.5 * x'Qx + p'x (default)
- LINEAR_LS: J = 0.5 * ||Vx*x + Vu*u - y_ref||_W^2 (decoupled W and y_ref)
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_iterate import AcadosOcpFlattenedIterate
from drone_models.core import load_params
from leap_c.ocp.acados.data import AcadosOcpSolverInput
from leap_c.ocp.acados.initializer import AcadosDiffMpcInitializer
from leap_c.ocp.acados.parameters import AcadosParameter, AcadosParameterManager
from leap_c.ocp.acados.planner import AcadosPlanner
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch

# Import from qp by default
from .quadrotor_ocp_qp import (
    NX,
    NU,
    Q_STATE_SIZE,
    Q_CTRL_SIZE,
    P_X_SIZE,
    P_U_SIZE,
    QuadrotorAcadosParamInterface,
)

# Cost type literal for type hints
CostType = Literal["qp", "linear_ls"]
StateType = Literal["quat", "euler"]
IntegratorType = Literal["rk4", "euler"]


class QuadrotorHoverInitializer(AcadosDiffMpcInitializer):
    """Initializer that sets state to x0 and control to hover thrust.

    For each problem instance, the state trajectory is initialized to the
    initial state broadcast across all stages, and the control trajectory
    is initialized to hover [0, 0, 0, hover_thrust_cmd].

    The hover thrust command accounts for the thrust coefficient from the
    so_rpy model: thrust_cmd = (mass * gravity) / cmd_f_coef
    """

    def __init__(self, ocp: AcadosOcp, mass: float, gravity: float, cmd_f_coef: float):
        """Initialize with OCP and drone physical parameters.

        Args:
            ocp: The acados OCP for obtaining default iterate structure.
            mass: Drone mass [kg].
            gravity: Gravitational acceleration magnitude [m/s^2].
            cmd_f_coef: Thrust command coefficient from so_rpy model.
        """
        self.default_iterate = ocp.create_default_initial_iterate().flatten()
        self.N = ocp.solver_options.N_horizon
        self.nx = ocp.dims.nx
        self.nu = ocp.dims.nu
        # Hover thrust command: MPC dynamics use force = cmd_f_coef * cmd
        # so to get force = mass * gravity, need cmd = (mass * gravity) / cmd_f_coef
        self.hover_thrust = (mass * gravity) / cmd_f_coef
        self.hover_u = np.zeros(self.nu)
        self.hover_u[-1] = self.hover_thrust  # thrust is last element

        # Precompute tiled hover control for single and batch use
        self._hover_u_tiled = np.tile(self.hover_u, self.N)

    def single_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedIterate:
        """Generate initial iterate with state=x0 and control=hover.

        Args:
            solver_input: Solver input with x0 of shape (1, nx).

        Returns:
            Flattened iterate with hover initialization.
        """
        iterate = deepcopy(self.default_iterate)

        # State trajectory: tile x0 across (N+1) stages
        x0 = solver_input.x0.flatten()  # (nx,)
        iterate.x = np.tile(x0, self.N + 1)

        # Control trajectory: precomputed hover
        iterate.u = self._hover_u_tiled.copy()

        return iterate

    def batch_iterate(self, solver_input: AcadosOcpSolverInput) -> "AcadosOcpFlattenedBatchIterate":
        """Vectorized batch initialization (avoids per-sample Python loop).

        Args:
            solver_input: Batch solver input with x0 of shape (B, nx).

        Returns:
            Batched iterate with hover initialization for all samples.
        """
        from acados_template.acados_ocp_iterate import AcadosOcpFlattenedBatchIterate

        B = solver_input.batch_size

        # State trajectory: tile each x0 across (N+1) stages -> (B, (N+1)*nx)
        x_batch = np.tile(solver_input.x0, (1, self.N + 1))

        # Control trajectory: same hover for all samples -> (B, N*nu)
        u_batch = np.tile(self._hover_u_tiled, (B, 1))

        # Other fields: zeros matching default iterate dimensions
        z_size = self.default_iterate.z.size
        sl_size = self.default_iterate.sl.size
        su_size = self.default_iterate.su.size
        pi_size = self.default_iterate.pi.size
        lam_size = self.default_iterate.lam.size

        return AcadosOcpFlattenedBatchIterate(
            x=x_batch,
            u=u_batch,
            z=np.zeros((B, z_size)) if z_size > 0 else np.zeros((B, 0)),
            sl=np.zeros((B, sl_size)) if sl_size > 0 else np.zeros((B, 0)),
            su=np.zeros((B, su_size)) if su_size > 0 else np.zeros((B, 0)),
            pi=np.zeros((B, pi_size)) if pi_size > 0 else np.zeros((B, 0)),
            lam=np.zeros((B, lam_size)) if lam_size > 0 else np.zeros((B, 0)),
            N_batch=B,
        )


@dataclass(kw_only=True)
class QuadrotorPlannerConfig:
    """Configuration for quadrotor MPC planner with so_rpy Euler dynamics.

    Attributes:
        N_horizon: Number of MPC horizon steps.
        T_horizon: Total horizon time [s].
        dt: Integration timestep [s].
        param_interface: "global" for same params all stages, "stagewise" for varying.
        cost_type: Cost formulation - "qp" (J=0.5*x'Qx + p'x) or "linear_ls" (J=0.5*||y-y_ref||_W^2).
        n_batch_max: Maximum batch size for parallel solves.
        num_threads: Number of parallel threads for batch solver.
        drone_model: Drone model identifier for parameter loading.
        velocity_max: Maximum velocity magnitude [m/s]. None to disable constraint.
        roll_pitch_max: Maximum roll/pitch command [rad].
        yaw_max: Maximum yaw command [rad].
        thrust_min: Minimum collective thrust [N]. None to load from drone_model.
        thrust_max: Maximum collective thrust [N]. None to load from drone_model.
        mass: Drone mass [kg]. None to load from drone_model.
        gravity: Gravitational acceleration [m/s^2]. None to load from drone_model.
    """

    N_horizon: int = 2
    dt: float = 0.01  # 100Hz MPC
    T_horizon: float = None  # Will be set in __post_init__
    param_interface: QuadrotorAcadosParamInterface = "stagewise"
    cost_type: CostType = "qp"  # "qp" or "linear_ls"
    state_type: StateType = "quat"  # "quat" (13D) or "euler" (12D, for old runs)
    integrator: IntegratorType = "rk4"  # "rk4" or "euler" (forward Euler)
    n_batch_max: int = 4096
    num_threads: int = 8
    drone_model: str = "cf2x_L250"
    mpc_model: str = "so_rpy"  # "so_rpy" or "so_rpy_rotor_drag"
    velocity_max: float | None = None  # m/s, None to disable
    roll_pitch_max: float = 0.5
    yaw_max: float = 0.1
    thrust_min: float | None = None  # N, None to load from drone_model
    thrust_max: float | None = None  # N, None to load from drone_model
    mass: float | None = None  # kg, None to load from drone_model
    gravity: float | None = None  # m/s^2, None to load from drone_model
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.T_horizon is None:
            self.T_horizon = self.N_horizon * self.dt
        if self.cost_type not in ("qp", "linear_ls"):
            raise ValueError(f"cost_type must be 'qp' or 'linear_ls', got '{self.cost_type}'")
        if self.state_type not in ("quat", "euler"):
            raise ValueError(f"state_type must be 'quat' or 'euler', got '{self.state_type}'")
        if self.state_type == "euler" and self.cost_type == "qp":
            raise ValueError("Euler state_type is only supported with cost_type='linear_ls'")


class QuadrotorPlanner(AcadosPlanner[AcadosDiffMpcCtx]):
    """Differentiable MPC planner for quadrotor with so_rpy dynamics.

    Integrates with leap-c's AcadosPlanner pattern for use with RL frameworks.

    State: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz] (13D)
    Control: [roll, pitch, yaw, thrust] (4D)

    For stagewise mode, the planner expects parameters in the order:
    - q_state for each stage (N+1 stages): 13 * (N+1)
    - q_control for each stage (N stages): 4 * N
    - p_x for each stage (N+1 stages): 13 * (N+1)
    - p_u for each stage (N stages): 4 * N

    Attributes:
        cfg: Planner configuration.
        param_manager: leap-c parameter manager.
        diff_mpc: Differentiable MPC PyTorch module.
    """

    cfg: QuadrotorPlannerConfig

    def __init__(
        self,
        cfg: QuadrotorPlannerConfig | None = None,
        params: list[AcadosParameter] | None = None,
        export_directory: Path | None = None,
    ):
        """Initialize the quadrotor planner.

        Args:
            cfg: Planner configuration. Uses defaults if None.
            params: Optional custom parameters. Uses defaults if None.
            export_directory: Directory for acados code generation.
        """
        self.cfg = QuadrotorPlannerConfig() if cfg is None else cfg

        # Load drone physical parameters from drone-models
        self.drone_params = load_params(self.cfg.mpc_model, self.cfg.drone_model)

        # Import correct OCP module based on cost type and state type
        if self.cfg.cost_type == "qp":
            from .quadrotor_ocp_qp import (
                create_quadrotor_params_qp as create_quadrotor_params,
                export_parametric_ocp_qp as export_parametric_ocp,
                get_learnable_param_dim_qp as get_learnable_param_dim,
            )
            ocp_name = "quadrotor_so_rpy_qp"
        elif self.cfg.state_type == "euler":
            from .quadrotor_ocp_linear_ls import (
                create_quadrotor_params_linear_ls_euler as create_quadrotor_params,
                export_parametric_ocp_linear_ls_euler as export_parametric_ocp,
                get_learnable_param_dim_linear_ls_euler as get_learnable_param_dim,
                NX_EULER,
            )
            ocp_name = "quadrotor_so_rpy_euler_linear_ls"
        else:  # linear_ls + quat
            from .quadrotor_ocp_linear_ls import (
                create_quadrotor_params_linear_ls as create_quadrotor_params,
                export_parametric_ocp_linear_ls as export_parametric_ocp,
                get_learnable_param_dim_linear_ls as get_learnable_param_dim,
            )
            ocp_name = "quadrotor_so_rpy_linear_ls"

        # Store get_learnable_param_dim and state dim for later use
        self._get_learnable_param_dim = get_learnable_param_dim
        self._nx = NX_EULER if self.cfg.state_type == "euler" else NX

        # Build kwargs shared by both cost types
        param_kwargs = dict(
            N_horizon=self.cfg.N_horizon,
            param_interface=self.cfg.param_interface,
            drone_model=self.cfg.drone_model,
            roll_pitch_max=self.cfg.roll_pitch_max,
            yaw_max=self.cfg.yaw_max,
            thrust_min=self.cfg.thrust_min,
            thrust_max=self.cfg.thrust_max,
            mass=self.cfg.mass,
            gravity=self.cfg.gravity,
        )
        ocp_kwargs = dict(
            name=ocp_name,
            N_horizon=self.cfg.N_horizon,
            T_horizon=self.cfg.T_horizon,
            dt=self.cfg.dt,
            drone_model=self.cfg.drone_model,
            velocity_max=self.cfg.velocity_max,
            roll_pitch_max=self.cfg.roll_pitch_max,
            yaw_max=self.cfg.yaw_max,
            thrust_min=self.cfg.thrust_min,
            thrust_max=self.cfg.thrust_max,
            mass=self.cfg.mass,
            gravity=self.cfg.gravity,
        )
        # linear_ls supports mpc_model and integrator selection
        if self.cfg.cost_type == "linear_ls":
            param_kwargs["mpc_model"] = self.cfg.mpc_model
            ocp_kwargs["mpc_model"] = self.cfg.mpc_model
            ocp_kwargs["integrator"] = self.cfg.integrator

        # Create parameters
        params = (
            create_quadrotor_params(**param_kwargs)
            if params is None
            else params
        )

        # Create parameter manager
        # Euler state type uses SX dynamics, quaternion uses MX from drone-models
        casadi_type = "SX" if self.cfg.state_type == "euler" else "MX"
        param_manager = AcadosParameterManager(
            parameters=params,
            N_horizon=self.cfg.N_horizon,
            casadi_type=casadi_type,
        )

        # Create OCP
        ocp = export_parametric_ocp(
            param_manager=param_manager,
            **ocp_kwargs,
        )

        # Create hover initializer for solver warm-start
        # Use provided mass/gravity or fall back to drone_params
        mass = self.cfg.mass if self.cfg.mass is not None else float(self.drone_params["mass"])
        gravity = self.cfg.gravity if self.cfg.gravity is not None else float(np.abs(self.drone_params["gravity_vec"][2]))
        cmd_f_coef = float(self.drone_params["cmd_f_coef"])
        initializer = QuadrotorHoverInitializer(ocp, mass=mass, gravity=gravity, cmd_f_coef=cmd_f_coef)

        # Create differentiable MPC
        diff_mpc = AcadosDiffMpcTorch(
            ocp,
            initializer=initializer,
            export_directory=export_directory,
            n_batch_max=self.cfg.n_batch_max,
            num_threads_batch_solver=self.cfg.num_threads,
            dtype=self.cfg.dtype,
        )

        super().__init__(param_manager=param_manager, diff_mpc=diff_mpc)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        param: torch.Tensor | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: solve MPC with given state and parameters.

        Args:
            obs: Initial state x0 with shape (B, 13).
                 Expected order: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
            action: Not used (action comes from MPC solution).
            param: Learnable parameters with shape (B, n_learnable).
            ctx: Optional context for warmstarting.

        Returns:
            ctx: Updated context with solution info.
            u0: Optimal control at first step, shape (B, 4).
                 Order: [roll, pitch, yaw, thrust]
            x_traj: State trajectory, shape (B, N+1, 13).
            u_traj: Control trajectory, shape (B, N, 4).
            value: Cost value, shape (B, 1).
        """
        # Get non-learnable parameters
        p_stagewise = self.param_manager.combine_non_learnable_parameter_values(
            batch_size=obs.shape[0]
        )

        # Extract state (first nx elements)
        x0 = obs[:, :self._nx]

        return self.diff_mpc(
            x0=x0,
            u0=action,
            p_global=param,
            p_stagewise=p_stagewise,
            ctx=ctx,
        )

    def get_learnable_param_dim(self) -> int:
        """Get total dimension of learnable parameters.

        Returns:
            Total number of learnable parameters.
        """
        return self._get_learnable_param_dim(self.cfg.N_horizon, self.cfg.param_interface)

    def get_default_params(self, batch_size: int = 1) -> np.ndarray:
        """Get default parameter values.

        Args:
            batch_size: Number of parameter sets to return.

        Returns:
            Default parameters with shape (batch_size, param_dim).
        """
        default = self.param_manager.learnable_parameters_default.cat.full().flatten()
        if batch_size > 1:
            return np.tile(default, (batch_size, 1))
        return default[None, :]

    def get_param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds.

        Returns:
            Tuple of (lower_bounds, upper_bounds) each with shape (param_dim,).
        """
        lb = self.param_manager.learnable_parameters_lb.cat.full().flatten()
        ub = self.param_manager.learnable_parameters_ub.cat.full().flatten()
        return lb, ub

    def get_param_structure_info(self) -> dict:
        """Get information about parameter structure for stagewise mode.

        Returns:
            Dictionary with parameter structure information.
        """
        N = self.cfg.N_horizon
        return {
            'N_horizon': N,
            'param_interface': self.cfg.param_interface,
            'q_state_size': Q_STATE_SIZE,
            'q_ctrl_size': Q_CTRL_SIZE,
            'p_x_size': P_X_SIZE,
            'p_u_size': P_U_SIZE,
            'n_state_stages': N + 1,
            'n_ctrl_stages': N,
            'total_dim': self.get_learnable_param_dim(),
        }
