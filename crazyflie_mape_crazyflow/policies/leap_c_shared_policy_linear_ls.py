"""SKRL policies with LEAP-C MPC integration using LINEAR_LS cost.

This module provides SKRL-compatible policies that use LEAP-C's differentiable MPC
with LINEAR_LS cost formulation. The policy network outputs W (weights) and
y_ref (references) separately, which is more numerically stable.

Cost: J = 0.5 * ||Vx*x + Vu*u - y_ref||_W^2

The NN outputs:
1. W (weights) - log scaled for orders of magnitude control
2. y_ref (references) - position uses RELATIVE offsets from current position,
   other states use absolute bounds

Position reference: y_ref_pos = current_pos + offset, where offset in [-pos_offset_max, +pos_offset_max]
This "nudging" approach is similar to EXTERNAL cost and works better for trajectory tracking.

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
from crazyflie_mape_crazyflow.leap_c.quadrotor_ocp_linear_ls import (
    NX,
    NU,
)


def _rotation_matrix_to_euler(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to RPY Euler angles (XYZ intrinsic convention).

    Args:
        rotmat: Rotation matrices, shape (..., 3, 3).

    Returns:
        RPY angles [roll, pitch, yaw], shape (..., 3).
    """
    sy = torch.sqrt(rotmat[..., 0, 0] ** 2 + rotmat[..., 1, 0] ** 2)
    pitch = torch.atan2(-rotmat[..., 2, 0], sy)
    roll = torch.atan2(rotmat[..., 2, 1], rotmat[..., 2, 2])
    yaw = torch.atan2(rotmat[..., 1, 0], rotmat[..., 0, 0])
    return torch.stack([roll, pitch, yaw], dim=-1)


def _body_rates_to_euler_rates(rpy: torch.Tensor, body_rates: torch.Tensor) -> torch.Tensor:
    """Convert body angular velocity [p, q, r] to Euler rates [droll, dpitch, dyaw].

    Args:
        rpy: RPY angles, shape (..., 3).
        body_rates: Body angular velocity [p, q, r], shape (..., 3).

    Returns:
        Euler rates [droll, dpitch, dyaw], shape (..., 3).
    """
    phi = rpy[..., 0]
    theta = rpy[..., 1]
    p = body_rates[..., 0]
    q = body_rates[..., 1]
    r = body_rates[..., 2]

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    tan_theta = torch.tan(theta)

    droll = p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
    dpitch = cos_phi * q - sin_phi * r
    dyaw = (sin_phi / cos_theta) * q + (cos_phi / cos_theta) * r
    return torch.stack([droll, dpitch, dyaw], dim=-1)


def get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
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


class LeapCMPCLayerLinearLS(nn.Module):
    """Neural network layer that wraps LEAP-C differentiable MPC with LINEAR_LS cost.

    The layer outputs W (weights) and y_ref (references) separately,
    which are then passed to the MPC solver.

    State: [pos(3), rpy(3), vel(3), drpy(3)] = 12D
    Control: [roll, pitch, yaw, thrust] = 4D
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
        pos_offset_max: float = 1.0,
    ):
        """Initialize the MPC layer with LINEAR_LS cost.

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
            activation: Hidden layer activation function name.
            pos_offset_max: Maximum relative position offset [m]. NN outputs offset
                from current position in [-pos_offset_max, +pos_offset_max] range.
        """
        super().__init__()

        self.device = device
        self.mpc_horizon = mpc_horizon
        self.mpc_dt = mpc_dt

        # Create LEAP-C planner with stagewise parameters and LINEAR_LS cost
        planner_cfg = QuadrotorPlannerConfig(
            N_horizon=mpc_horizon,
            dt=mpc_dt,
            param_interface="stagewise",
            cost_type="linear_ls",  # Use LINEAR_LS cost
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

        # Get physical parameters
        self.mass = mass if mass is not None else float(self.planner.drone_params["mass"])
        self.gravity = gravity if gravity is not None else float(np.abs(self.planner.drone_params["gravity_vec"][2]))
        self.cmd_f_coef = float(self.planner.drone_params["cmd_f_coef"])
        # Hover thrust in Newtons (same units as thrust_min/thrust_max from config)
        self.hover_thrust = self.mass * self.gravity

        # Get parameter dimensions
        self.param_dim = self.planner.get_learnable_param_dim()
        self.param_info = self.planner.get_param_structure_info()
        self.n_state_stages = self.param_info['n_state_stages']
        self.n_ctrl_stages = self.param_info['n_ctrl_stages']

        # Action scaling for normalized output
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

        # === Log scaling bounds for W (weights) ===
        self.register_buffer(
            'w_state_min_log',
            torch.tensor([-1., -1., -1., -2., -2., -2., -1., -1., -1., -1., -1., -1.])
        )
        self.register_buffer(
            'w_state_max_log',
            torch.tensor([2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1.])
        )
        self.register_buffer(
            'w_ctrl_min_log',
            torch.tensor([-1., -1., -1., -1.])
        )
        self.register_buffer(
            'w_ctrl_max_log',
            torch.tensor([1., 1., 1., 1.])
        )

        # === Linear scaling bounds for y_ref (references) ===
        # Position uses RELATIVE offsets from current position (added in _scale_parameters)
        # Other states use absolute bounds
        self.register_buffer(
            'pos_offset_min',
            torch.tensor([-pos_offset_max, -pos_offset_max, -pos_offset_max])  # Relative position offset bounds [m]
        )
        self.register_buffer(
            'pos_offset_max_buf',
            torch.tensor([pos_offset_max, pos_offset_max, pos_offset_max])  # Relative position offset bounds [m]
        )
        self.register_buffer(
            'yref_state_min',
            torch.tensor([0., 0., 0., -roll_pitch_max, -roll_pitch_max, -yaw_max, -5., -5., -5., -10., -10., -10.])
        )
        self.register_buffer(
            'yref_state_max',
            torch.tensor([0., 0., 0., roll_pitch_max, roll_pitch_max, yaw_max, 5., 5., 5., 10., 10., 10.])
        )
        self.register_buffer(
            'yref_ctrl_min',
            torch.tensor([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min])
        )
        self.register_buffer(
            'yref_ctrl_max',
            torch.tensor([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max])
        )

        # Store hover thrust and thrust bounds for centered scaling
        self.register_buffer('hover_thrust_buf', torch.tensor(self.hover_thrust, dtype=torch.float32))
        self.register_buffer('thrust_min_buf', torch.tensor(thrust_min, dtype=torch.float32))
        self.register_buffer('thrust_max_buf', torch.tensor(thrust_max, dtype=torch.float32))

        # Position references are relative offsets from current position

        # Cost parameter network
        # Output: [w_state, w_ctrl, yref_state, yref_ctrl] per stage
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

        # Scale parameters using LINEAR_LS approach
        mpc_params = self._scale_parameters_linear_ls(cost_net_out, batch_size, state)

        # Solve MPC
        _, u0, x_traj, u_traj, value = self.planner(
            obs=state,
            param=mpc_params,
        )

        # Normalize action to [-1, 1]
        action_normalized = (u0 - self.action_mean) / self.action_scale

        return action_normalized

    def _scale_parameters_linear_ls(
        self,
        net_out: torch.Tensor,
        batch_size: int,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Scale network output to MPC parameters using LINEAR_LS formulation.

        The NN outputs are scaled as:
        - W (weights): log scaling -> W = 10^(min_log + raw * (max_log - min_log))
        - y_ref (references): linear scaling -> y_ref = min + raw * (max - min)

        Args:
            net_out: Network output in [0, 1], shape (B, param_dim).
            batch_size: Batch size.
            state: Current state for reference tracking.

        Returns:
            Scaled MPC parameters [w_state, w_ctrl, yref_state, yref_ctrl].
        """
        # Split into parameter groups
        w_state_total = NX * self.n_state_stages
        w_ctrl_total = NU * self.n_ctrl_stages
        yref_state_total = NX * self.n_state_stages
        yref_ctrl_total = NU * self.n_ctrl_stages

        idx = 0
        w_state_raw = net_out[:, idx:idx + w_state_total]
        idx += w_state_total
        w_ctrl_raw = net_out[:, idx:idx + w_ctrl_total]
        idx += w_ctrl_total
        yref_state_raw = net_out[:, idx:idx + yref_state_total]
        idx += yref_state_total
        yref_ctrl_raw = net_out[:, idx:idx + yref_ctrl_total]

        # Reshape to per-stage
        w_state_per_stage = w_state_raw.reshape(batch_size, self.n_state_stages, NX)
        w_ctrl_per_stage = w_ctrl_raw.reshape(batch_size, self.n_ctrl_stages, NU)
        yref_state_per_stage = yref_state_raw.reshape(batch_size, self.n_state_stages, NX)
        yref_ctrl_per_stage = yref_ctrl_raw.reshape(batch_size, self.n_ctrl_stages, NU)

        # === Log scaling for W (weights) ===
        # W = 10^(min_log + raw * (max_log - min_log))
        log_w_state = self.w_state_min_log + w_state_per_stage * (self.w_state_max_log - self.w_state_min_log)
        W_state = torch.pow(10., log_w_state)

        log_w_ctrl = self.w_ctrl_min_log + w_ctrl_per_stage * (self.w_ctrl_max_log - self.w_ctrl_min_log)
        W_ctrl = torch.pow(10., log_w_ctrl)

        # === Linear scaling for y_ref (references) ===
        # Position: scale as RELATIVE offset, then add current position
        # pos_offset in [-pos_offset_max, +pos_offset_max] range
        pos_offset = self.pos_offset_min + yref_state_per_stage[..., :3] * (self.pos_offset_max_buf - self.pos_offset_min)

        # Get current position from state: state is [pos(3), rpy(3), vel(3), drpy(3)]
        current_pos = state[:, :3]  # (batch, 3)

        # Broadcast current position across all stages and add offset
        # current_pos: (batch, 3) -> (batch, 1, 3) for broadcasting
        yref_pos_absolute = current_pos[:, None, :] + pos_offset  # (batch, n_state_stages, 3)

        # Other state refs (rpy, vel, drpy) use absolute bounds
        yref_other = self.yref_state_min[3:] + yref_state_per_stage[..., 3:] * (self.yref_state_max[3:] - self.yref_state_min[3:])

        # Combine position (absolute) with other state refs
        yref_state = torch.cat([yref_pos_absolute, yref_other], dim=-1)

        # Linear scaling for roll, pitch, yaw (symmetric around 0)
        yref_ctrl = self.yref_ctrl_min + yref_ctrl_per_stage * (self.yref_ctrl_max - self.yref_ctrl_min)

        # Thrust scaling centered on hover: raw=0.5 -> hover_thrust
        # Piecewise linear: [0, 0.5] -> [thrust_min, hover], [0.5, 1] -> [hover, thrust_max]
        thrust_raw = yref_ctrl_per_stage[..., 3]  # (batch, n_ctrl_stages)
        thrust_below = self.thrust_min_buf + 2.0 * thrust_raw * (self.hover_thrust_buf - self.thrust_min_buf)
        thrust_above = self.hover_thrust_buf + 2.0 * (thrust_raw - 0.5) * (self.thrust_max_buf - self.hover_thrust_buf)
        yref_ctrl[..., 3] = torch.where(thrust_raw <= 0.5, thrust_below, thrust_above)

        # Flatten and concatenate
        w_state_flat = W_state.reshape(batch_size, -1)
        w_ctrl_flat = W_ctrl.reshape(batch_size, -1)
        yref_state_flat = yref_state.reshape(batch_size, -1)
        yref_ctrl_flat = yref_ctrl.reshape(batch_size, -1)

        return torch.cat([w_state_flat, w_ctrl_flat, yref_state_flat, yref_ctrl_flat], dim=-1)


class LeapCSharedGaussianPolicyLinearLS(GaussianMixin, Model):
    """SKRL Gaussian policy using LEAP-C MPC with LINEAR_LS cost.

    This policy uses the LINEAR_LS cost formulation which is more numerically
    stable for neural network integration. The NN learns W (weights) and
    y_ref (references) separately.
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
        thrust_min: float = 1.23,
        thrust_max: float = 3.68,
        mass: Optional[float] = None,
        gravity: Optional[float] = None,
        drone_model: str = "cf2x_L250",
        n_batch_max: int = 4096,
        num_threads: int = 8,
        velocity_max: Optional[float] = None,
        activation: str = "relu",
        pos_offset_max: float = 1.0,
        **kwargs,
    ):
        """Initialize the policy with LINEAR_LS cost."""
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        self._g_clip_log_std = clip_log_std
        self._g_min_log_std = min_log_std
        self._g_max_log_std = max_log_std

        self.mpc_horizon = mpc_horizon
        self.mpc_dt = mpc_dt
        self.state_indices = state_indices or self._default_state_indices()

        obs_dim = gymnasium.spaces.flatdim(observation_space)
        action_dim = gymnasium.spaces.flatdim(action_space)

        # MPC layer with LINEAR_LS cost
        self.mpc_layer = LeapCMPCLayerLinearLS(
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
            pos_offset_max=pos_offset_max,
        )

        # Log std parameter
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
        - [6:15] = rotation matrix (flattened 3x3)
        - [15:18] = body angular velocity (p, q, r)
        """
        return {
            'position': [0, 1, 2],
            'velocity': [3, 4, 5],
            'rotation_matrix': [6, 7, 8, 9, 10, 11, 12, 13, 14],
            'body_rates': [15, 16, 17],
        }

    def compute(
        self,
        inputs: Mapping[str, torch.Tensor],
        role: str = "",
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Compute actions from observations."""
        obs = inputs["observations"]

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        # Use raw MPC state from inputs if available (bypasses observation normalization),
        # otherwise extract from (potentially normalized) observations as fallback
        if "mpc_state" in inputs:
            state = inputs["mpc_state"]
        else:
            state = self._extract_state(obs)
        mean_actions = self.mpc_layer(obs, state)

        log_std = self.log_std_parameter
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_min_log_std, self._g_max_log_std)

        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        return mean_actions, {"log_std": log_std}

    def _extract_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract MPC state from observation (fallback when inputs["mpc_state"] unavailable).

        Converts rotation matrix → RPY and body rates → Euler rates for MPC.
        """
        position = obs[:, self.state_indices['position']]
        velocity = obs[:, self.state_indices['velocity']]

        if 'rotation_matrix' in self.state_indices:
            rotmat = obs[:, self.state_indices['rotation_matrix']].reshape(-1, 3, 3)
            body_rates = obs[:, self.state_indices['body_rates']]
            rpy = _rotation_matrix_to_euler(rotmat)
            rpy_rates = _body_rates_to_euler_rates(rpy, body_rates)
        elif 'attitude' in self.state_indices:
            rpy = obs[:, self.state_indices['attitude']]
            if 'rpy_rates' in self.state_indices:
                rpy_rates = obs[:, self.state_indices['rpy_rates']]
            else:
                rpy_rates = torch.zeros(obs.shape[0], 3, device=obs.device)
        else:
            rpy = torch.zeros(obs.shape[0], 3, device=obs.device)
            rpy_rates = torch.zeros(obs.shape[0], 3, device=obs.device)

        return torch.cat([position, rpy, velocity, rpy_rates], dim=-1)
