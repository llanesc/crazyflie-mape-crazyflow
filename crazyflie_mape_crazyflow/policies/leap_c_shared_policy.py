"""SKRL policies with LEAP-C MPC integration and parameter sharing.

This module provides SKRL-compatible policies that use LEAP-C's differentiable MPC
as the action generation mechanism. The policy network outputs MPC cost parameters
which are then used by the MPC solver to compute optimal control actions.

Adapted for so_rpy Euler dynamics with attitude commands [roll, pitch, yaw, thrust].

State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw] (12D)
Control: [roll, pitch, yaw, thrust] (4D)
"""

from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model

from crazyflie_mape_crazyflow.leap_c import (
    QuadrotorPlanner,
    QuadrotorPlannerConfig,
    NX,
    NU,
)

if TYPE_CHECKING:
    from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx
from crazyflie_mape_crazyflow.leap_c.quadrotor_ocp import (
    Q_STATE_SIZE,
    Q_CTRL_SIZE,
    P_X_SIZE,
    P_U_SIZE,
)


def get_activation(name: str) -> nn.Module:
    """Get activation module by name.

    Args:
        name: Activation function name (relu, tanh, elu, leaky_relu, gelu).

    Returns:
        Activation module instance.

    Raises:
        ValueError: If activation name is not recognized.
    """
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
    }
    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(activations.keys())}"
        )
    return activations[name.lower()]


class LeapCMPCLayer(nn.Module):
    """Neural network layer that wraps LEAP-C differentiable MPC.

    The layer takes observation features and outputs MPC cost parameters,
    which are then passed to the MPC solver to compute optimal attitude control.

    State: [pos(3), rpy(3), vel(3), drpy(3)] = 12D
    Control: [roll, pitch, yaw, thrust] = 4D

    Attributes:
        planner: LEAP-C QuadrotorPlanner instance.
        cost_net: Neural network that outputs cost parameters.
    """

    def __init__(
        self,
        observation_dim: int,
        mpc_horizon: int = 2,
        mpc_dt: float = 0.01,
        hidden_dim: int = 256,
        device: Union[str, torch.device] = "cuda",
        roll_pitch_max: float = 0.5,
        yaw_max: float = 0.1,
        drone_model: str = "cf2x_L250",
        n_batch_max: int = 4096,
        num_threads: int = 8,
        velocity_max: Optional[float] = None,
        activation: str = "relu",
        verbose: bool = True,
    ):
        """Initialize the MPC layer.

        Args:
            observation_dim: Dimension of input observations.
            mpc_horizon: MPC prediction horizon.
            mpc_dt: MPC timestep [s].
            hidden_dim: Hidden layer dimension.
            device: Computation device.
            roll_pitch_max: Maximum roll/pitch command in rad.
            yaw_max: Maximum yaw command in rad.
            drone_model: Drone model identifier.
            n_batch_max: Maximum batch size for parallel MPC solves.
            num_threads: Number of threads for parallel MPC solves.
            velocity_max: Maximum velocity constraint [m/s]. None to disable.
            activation: Hidden layer activation function name
                (relu, tanh, elu, leaky_relu, gelu).
            verbose: Whether to print acados build output.
        """
        super().__init__()

        self.device = device
        self.mpc_horizon = mpc_horizon
        self.mpc_dt = mpc_dt

        # Create LEAP-C planner with stagewise parameters
        planner_cfg = QuadrotorPlannerConfig(
            N_horizon=mpc_horizon,
            dt=mpc_dt,
            param_interface="stagewise",
            n_batch_max=n_batch_max,
            num_threads=num_threads,
            drone_model=drone_model,
            velocity_max=velocity_max,
            roll_pitch_max=roll_pitch_max,
            yaw_max=yaw_max,
            verbose=verbose,
        )
        self.planner = QuadrotorPlanner(cfg=planner_cfg)

        # Get physical parameters from planner's loaded drone params
        self.mass = float(self.planner.drone_params["mass"])
        self.gravity = float(np.abs(self.planner.drone_params["gravity_vec"][2]))
        min_thrust = float(self.planner.drone_params["thrust_min"]) * 4  # Per motor -> collective
        max_thrust = float(self.planner.drone_params["thrust_max"]) * 4


        # Get parameter dimensions
        self.param_dim = self.planner.get_learnable_param_dim()
        self.param_info = self.planner.get_param_structure_info()
        self.n_state_stages = self.param_info['n_state_stages']
        self.n_ctrl_stages = self.param_info['n_ctrl_stages']

        # Scaling constants
        # self.epsilon = 0.1
        # self.range_Q = 100000.0
        # self.range_p = 100000.0
        # self.range_p_t = 2 * self.range_Q / 2 * mass * gravity

        # Action scaling for normalized output to match environment
        # Environment expects: roll/pitch in [-roll_pitch_max, roll_pitch_max]
        #                      yaw in [-yaw_max, yaw_max]
        #                      thrust in [min_thrust, max_thrust]
        # Normalized action = (raw - mean) / scale, where scale = (max - min) / 2
        thrust_mean = (min_thrust + max_thrust) / 2.0
        thrust_scale = (max_thrust - min_thrust) / 2.0
        self.register_buffer(
            'action_mean',
            torch.tensor([0.0, 0.0, 0.0, thrust_mean], dtype=torch.float32)
        )
        self.register_buffer(
            'action_scale',
            torch.tensor([roll_pitch_max, roll_pitch_max, yaw_max, thrust_scale], dtype=torch.float32)
        )

        state_penalty = torch.tensor([50., 50., 100., 1., 1., 1., 10., 10., 10., 5., 5., 5.])
        control_penalty = torch.tensor([1., 1., 1., 50.])
        state_scale = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

        self.register_buffer('state_penalty', state_penalty)
        self.register_buffer('control_penalty', control_penalty)
        self.register_buffer('state_scale', state_scale)

        # Cost parameter network
        self.cost_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, self.param_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        u0_guess: torch.Tensor | None = None,
        ctx: Optional["AcadosDiffMpcCtx"] = None,
    ) -> tuple[torch.Tensor, "AcadosDiffMpcCtx"]:
        """Forward pass through MPC layer.

        Args:
            obs: Observations with shape (B, obs_dim).
            state: MPC state with shape (B, 12) [pos, rpy, vel, drpy].
            u0_guess: Initial control guess with shape (B, 4) [roll, pitch, yaw, thrust].
                If None, solver uses its default initialization.
            ctx: Context from previous solve for warmstarting. If provided, the solver
                will use the previous solution as initial guess for faster convergence.

        Returns:
            Tuple of:
                - action_normalized: Normalized control action with shape (B, 4).
                - ctx: Context object for warmstarting subsequent solves.
        """
        batch_size = obs.shape[0]

        # Get cost parameters from network
        cost_net_out = self.cost_net(obs)

        # Scale parameters
        mpc_params = self._scale_parameters(cost_net_out, batch_size)

        # Solve MPC with optional initial guess and warmstart
        ctx, u0, x_traj, u_traj, value = self.planner(
            obs=state,
            action=u0_guess,
            param=mpc_params,
            ctx=ctx,
        )

        # Normalize action to [-1, 1]
        action_normalized = (u0 - self.action_mean) / self.action_scale

        return action_normalized, ctx

    def _scale_parameters(self, net_out: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Scale network output to MPC parameter space.

        Args:
            net_out: Network output in [0, 1], shape (B, param_dim).
            batch_size: Batch size.

        Returns:
            Scaled MPC parameters.
        """
        # Split into parameter groups
        q_state_total = Q_STATE_SIZE * self.n_state_stages
        q_ctrl_total = Q_CTRL_SIZE * self.n_ctrl_stages
        p_x_total = P_X_SIZE * self.n_state_stages
        p_u_total = P_U_SIZE * self.n_ctrl_stages

        idx = 0
        q_state_raw = net_out[:, idx:idx + q_state_total]
        idx += q_state_total
        q_ctrl_raw = net_out[:, idx:idx + q_ctrl_total]
        idx += q_ctrl_total
        p_x_raw = net_out[:, idx:idx + p_x_total]
        idx += p_x_total
        p_u_raw = net_out[:, idx:idx + p_u_total]

        q_nom = torch.cat((self.state_penalty, self.control_penalty))
        px = torch.sqrt(self.state_penalty)
        pu = torch.sqrt(self.control_penalty)
        # p_nom = torch.cat((px, pu)) * torch.cat((self.state_scale, self.action_scale))
        # range_Q = 2.*q_nom
        # range_p = 2.*p_nom
        range_p_t = 2 * 2 * self.control_penalty[-1] / 2 * self.mass * self.gravity
        epsilon = 0.01

        # Reshape to per-stage
        q_state_per_stage = q_state_raw.reshape(batch_size, self.n_state_stages, Q_STATE_SIZE)
        q_ctrl_per_stage = q_ctrl_raw.reshape(batch_size, self.n_ctrl_stages, Q_CTRL_SIZE)
        p_x_per_stage = p_x_raw.reshape(batch_size, self.n_state_stages, P_X_SIZE)
        p_u_per_stage = p_u_raw.reshape(batch_size, self.n_ctrl_stages, P_U_SIZE)

        # Scale quadratic costs
        q_state_scaled = q_state_per_stage * 2 * self.state_penalty + epsilon
        q_ctrl_scaled = q_ctrl_per_stage * 2 * self.control_penalty + epsilon

        # Scale linear costs (centered)
        p_x_scaled = (p_x_per_stage - 0.5) * 2 * px * self.state_scale

        # p_u: attitude centered, thrust positive bias
        p_att = (p_u_per_stage[..., :3] - 0.5) * 2 * pu[:3] * self.action_scale[:3]
        p_t = -(p_u_per_stage[..., 3:4] * range_p_t + epsilon)
        p_u_scaled = torch.cat([p_att, p_t], dim=-1)

        # Flatten and concatenate
        q_state_flat = q_state_scaled.reshape(batch_size, -1)
        q_ctrl_flat = q_ctrl_scaled.reshape(batch_size, -1)
        p_x_flat = p_x_scaled.reshape(batch_size, -1)
        p_u_flat = p_u_scaled.reshape(batch_size, -1)

        return torch.cat([q_state_flat, q_ctrl_flat, p_x_flat, p_u_flat], dim=-1)


class LeapCSharedGaussianPolicy(GaussianMixin, Model):
    """SKRL Gaussian policy using LEAP-C MPC with parameter sharing.

    This policy is designed for multi-agent MAPPO training where all agents
    share the same policy network (parameter sharing). The policy uses
    LEAP-C's differentiable MPC to compute actions.

    The observation is expected to contain state information that can be
    extracted to form the MPC initial state.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: Union[str, torch.device] = "cuda",
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: Union[float, Sequence[float]] = 0.0,
        mpc_horizon: int = 2,
        mpc_dt: float = 0.01,
        hidden_dim: int = 256,
        state_indices: Optional[dict] = None,
        roll_pitch_max: float = 0.5,
        yaw_max: float = 0.1,
        drone_model: str = "cf2x_L250",
        n_batch_max: int = 4096,
        num_threads: int = 8,
        velocity_max: Optional[float] = None,
        activation: str = "relu",
        verbose: bool = True,
        **kwargs,
    ):
        """Initialize the policy.

        Args:
            observation_space: Environment observation space.
            action_space: Environment action space.
            device: Computation device.
            clip_actions: Whether to clip actions.
            clip_log_std: Whether to clip log std.
            min_log_std: Minimum log std.
            max_log_std: Maximum log std.
            initial_log_std: Initial log std. Can be a scalar (applied to all
                action dimensions) or an array of size action_dim.
            mpc_horizon: MPC prediction horizon.
            mpc_dt: MPC timestep [s].
            hidden_dim: Hidden layer dimension.
            state_indices: Dictionary mapping state components to observation indices.
            roll_pitch_max: Maximum roll/pitch command in rad.
            yaw_max: Maximum yaw command in rad.
            drone_model: Drone model identifier.
            n_batch_max: Maximum batch size for parallel MPC solves.
            num_threads: Number of threads for parallel MPC solves.
            velocity_max: Maximum velocity constraint [m/s]. None to disable.
            activation: Hidden layer activation function name
                (relu, tanh, elu, leaky_relu, gelu).
            verbose: Whether to print acados build output.
        """
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        self.mpc_horizon = mpc_horizon
        self.mpc_dt = mpc_dt
        self.state_indices = state_indices or self._default_state_indices()

        # Get dimensions
        obs_dim = gymnasium.spaces.flatdim(observation_space)
        action_dim = gymnasium.spaces.flatdim(action_space)

        # MPC layer
        self.mpc_layer = LeapCMPCLayer(
            observation_dim=obs_dim,
            mpc_horizon=mpc_horizon,
            mpc_dt=mpc_dt,
            hidden_dim=hidden_dim,
            device=device,
            roll_pitch_max=roll_pitch_max,
            yaw_max=yaw_max,
            drone_model=drone_model,
            n_batch_max=n_batch_max,
            num_threads=num_threads,
            velocity_max=velocity_max,
            activation=activation,
            verbose=verbose,
        )

        # Log std parameter - handle scalar or array input
        if isinstance(initial_log_std, (list, tuple, np.ndarray)):
            initial_log_std_tensor = torch.tensor(
                initial_log_std, dtype=torch.float32, device=device
            )
            if initial_log_std_tensor.shape[0] != action_dim:
                raise ValueError(
                    f"initial_log_std array has size {initial_log_std_tensor.shape[0]}, "
                    f"expected {action_dim} (action_dim)"
                )
        else:
            initial_log_std_tensor = torch.full(
                (action_dim,), initial_log_std, dtype=torch.float32, device=device
            )
        self.log_std_parameter = nn.Parameter(initial_log_std_tensor)

    def _default_state_indices(self) -> dict:
        """Default state indices for RedVsBlueEnv observation format.

        The observation format is:
        - [0:3] = position (x, y, z)
        - [3:6] = velocity (vx, vy, vz)
        - [6:9] = attitude (roll, pitch, yaw)
        - [9:12] = angular velocity (droll, dpitch, dyaw)

        Returns:
            Dictionary with indices for position, velocity, attitude, and rpy rates.
        """
        return {
            'position': [0, 1, 2],
            'velocity': [3, 4, 5],
            'attitude': [6, 7, 8],  # RPY (roll, pitch, yaw)
            'rpy_rates': [9, 10, 11],  # Euler angle rates (droll, dpitch, dyaw)
        }

    def compute(
        self,
        inputs: Mapping[str, torch.Tensor],
        role: str = "",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Mapping[str, torch.Tensor]]:
        """Compute actions from observations.

        Args:
            inputs: Dictionary containing observations.
            role: Model role (unused).

        Returns:
            Tuple of (actions, log_prob, outputs).
        """
        obs = inputs["states"]

        # Convert to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        # Extract MPC state from observations
        state = self._extract_state(obs)

        # Get mean action from MPC (ignore context for now, could be used for warmstarting)
        mean_actions, _ = self.mpc_layer(obs, state)

        # Get log std
        log_std = self.log_std_parameter
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)

        # Store for distribution computation
        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        return mean_actions, log_std, {}

    def _extract_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract MPC state from observation.

        Args:
            obs: Full observation tensor.

        Returns:
            MPC state [pos, rpy, vel, drpy] with shape (B, 12).
        """
        batch_size = obs.shape[0]

        # Extract components
        pos_idx = self.state_indices['position']
        vel_idx = self.state_indices['velocity']
        att_idx = self.state_indices['attitude']

        position = obs[:, pos_idx]
        velocity = obs[:, vel_idx]
        attitude = obs[:, att_idx]  # RPY directly

        # Get RPY rates if available, otherwise assume zero
        if 'rpy_rates' in self.state_indices:
            rpy_rates_idx = self.state_indices['rpy_rates']
            rpy_rates = obs[:, rpy_rates_idx]
        else:
            rpy_rates = torch.zeros(batch_size, 3, device=obs.device)

        # State: [pos, rpy, vel, drpy]
        return torch.cat([position, attitude, velocity, rpy_rates], dim=-1)
