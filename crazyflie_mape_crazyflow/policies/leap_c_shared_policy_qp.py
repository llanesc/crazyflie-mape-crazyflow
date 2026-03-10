"""SKRL policies with LEAP-C MPC integration and parameter sharing.

This module provides SKRL-compatible policies that use LEAP-C's differentiable MPC
as the action generation mechanism. The policy network outputs MPC cost parameters
which are then used by the MPC solver to compute optimal control actions.

Adapted for so_rpy Euler dynamics with attitude commands [roll, pitch, yaw, thrust].

State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw] (12D)
Control: [roll, pitch, yaw, thrust] (4D)
"""

from typing import Mapping, Optional, Sequence, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model

from crazyflie_mape_crazyflow.leap_c import (
    QuadrotorPlanner,
    QuadrotorPlannerConfig,
)
from crazyflie_mape_crazyflow.leap_c.quadrotor_ocp_qp import (
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


class LeapCMPCLayerQP(nn.Module):
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
        thrust_min: float = 1.23,
        thrust_max: float = 3.68,
        mass: Optional[float] = None,
        gravity: Optional[float] = None,
        drone_model: str = "cf2x_L250",
        n_batch_max: int = 4096,
        num_threads: int = 8,
        velocity_max: Optional[float] = None,
        activation: str = "relu",
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
            thrust_min: Minimum collective thrust [N].
            thrust_max: Maximum collective thrust [N].
            mass: Drone mass [kg]. None to load from drone_model.
            gravity: Gravitational acceleration [m/s^2]. None to load from drone_model.
            drone_model: Drone model identifier.
            n_batch_max: Maximum batch size for parallel MPC solves.
            num_threads: Number of threads for parallel MPC solves.
            velocity_max: Maximum velocity constraint [m/s]. None to disable.
            activation: Hidden layer activation function name
                (relu, tanh, elu, leaky_relu, gelu).
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
            thrust_min=thrust_min,
            thrust_max=thrust_max,
            mass=mass,
            gravity=gravity,
        )
        self.planner = QuadrotorPlanner(cfg=planner_cfg)

        # Get physical parameters (use provided values or fall back to drone_params)
        self.mass = mass if mass is not None else float(self.planner.drone_params["mass"])
        self.gravity = gravity if gravity is not None else float(np.abs(self.planner.drone_params["gravity_vec"][2]))
        self.cmd_f_coef = float(self.planner.drone_params["cmd_f_coef"])


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
        #                      thrust in [thrust_min, thrust_max]
        # Normalized action = (raw - mean) / scale, where scale = (max - min) / 2
        thrust_mean = (thrust_min + thrust_max) / 2.0
        thrust_scale = (thrust_max - thrust_min) / 2.0
        self.register_buffer(
            'action_mean',
            torch.tensor([0.0, 0.0, 0.0, thrust_mean], dtype=torch.float32)
        )
        self.register_buffer(
            'action_scale',
            torch.tensor([roll_pitch_max, roll_pitch_max, yaw_max, thrust_scale], dtype=torch.float32)
        )

        state_penalty = torch.tensor([50., 50., 100., 1., 1., 1., 10., 10., 10., 5., 5., 5.])
        control_penalty = torch.tensor([1., 1., 1., 5.])
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
    ) -> torch.Tensor:
        """Forward pass through MPC layer.

        Args:
            obs: Observations with shape (B, obs_dim).
            state: MPC state with shape (B, 12) [pos, rpy, vel, drpy].

        Returns:
            Normalized control action with shape (B, 4).
        """
        batch_size = obs.shape[0]

        # Get cost parameters from network
        cost_net_out = self.cost_net(obs)

        # Scale parameters
        mpc_params = self._scale_parameters(cost_net_out, batch_size)

        # Solve MPC (initialization handled by QuadrotorHoverInitializer)
        _, u0, x_traj, u_traj, value = self.planner(
            obs=state,
            param=mpc_params,
        )

        # Normalize action to [-1, 1]
        action_normalized = (u0 - self.action_mean) / self.action_scale

        return action_normalized

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
        range_p_t = 2 * 2 * self.control_penalty[-1] / 2 * (self.mass * self.gravity) / self.cmd_f_coef
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


class LeapCSharedGaussianPolicyQP(GaussianMixin, Model):
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
        roll_pitch_max: float = 0.5,
        yaw_max: float = 0.1,
        thrust_min: float = 1.23,
        thrust_max: float = 3.68,
        mass: Optional[float] = None,
        gravity: Optional[float] = None,
        drone_model: str = "cf2x_L250",
        n_batch_max: int = 4096,
        num_threads: int = 8,
        velocity_max: Optional[float] = None,
        activation: str = "relu",
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
            roll_pitch_max: Maximum roll/pitch command in rad.
            yaw_max: Maximum yaw command in rad.
            thrust_min: Minimum collective thrust [N].
            thrust_max: Maximum collective thrust [N].
            mass: Drone mass [kg]. None to load from drone_model.
            gravity: Gravitational acceleration [m/s^2]. None to load from drone_model.
            drone_model: Drone model identifier.
            n_batch_max: Maximum batch size for parallel MPC solves.
            num_threads: Number of threads for parallel MPC solves.
            velocity_max: Maximum velocity constraint [m/s]. None to disable.
            activation: Hidden layer activation function name
                (relu, tanh, elu, leaky_relu, gelu).
        """
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        # Explicitly store GaussianMixin parameters to avoid nn.Module attribute issues
        self._g_clip_log_std = clip_log_std
        self._g_min_log_std = min_log_std
        self._g_max_log_std = max_log_std

        self.mpc_horizon = mpc_horizon
        self.mpc_dt = mpc_dt

        # Get dimensions
        obs_dim = gymnasium.spaces.flatdim(observation_space)
        action_dim = gymnasium.spaces.flatdim(action_space)

        # MPC layer
        self.mpc_layer = LeapCMPCLayerQP(
            observation_dim=obs_dim,
            mpc_horizon=mpc_horizon,
            mpc_dt=mpc_dt,
            hidden_dim=hidden_dim,
            device=device,
            roll_pitch_max=roll_pitch_max,
            yaw_max=yaw_max,
            thrust_min=thrust_min,
            thrust_max=thrust_max,
            mass=mass,
            gravity=gravity,
            drone_model=drone_model,
            n_batch_max=n_batch_max,
            num_threads=num_threads,
            velocity_max=velocity_max,
            activation=activation,
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

    def compute(
        self,
        inputs: Mapping[str, torch.Tensor],
        role: str = "",
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Compute actions from observations.

        Args:
            inputs: Dictionary containing observations (SKRL 2.0 format).
            role: Model role (unused).

        Returns:
            Tuple of (mean_actions, outputs_dict) where outputs_dict contains "log_std".
        """
        # SKRL 2.0: per-agent observations are under "observations" key
        obs = inputs["observations"]

        # Convert to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        state = inputs["mpc_state"]

        # Get mean action from MPC (uses hover as initial guess internally)
        mean_actions = self.mpc_layer(obs, state)

        # Get log std
        log_std = self.log_std_parameter
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_min_log_std, self._g_max_log_std)

        # Store for distribution computation
        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        # SKRL 2.0: return (mean_actions, outputs_dict) with "log_std" in outputs
        return mean_actions, {"log_std": log_std}

