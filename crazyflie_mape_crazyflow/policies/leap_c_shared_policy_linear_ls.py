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

Supports two state representations:
- Quaternion (13D): [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
- Euler (12D): [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]

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
    NX_EULER,
    NU,
)


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

    State: [pos(3), quat(4:xyzw), vel(3), ang_vel(3)] = 13D
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
        mpc_model: str = "so_rpy",
        state_type: str = "quat",
        integrator: str = "rk4",
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
            mpc_model: Physics model for MPC dynamics ("so_rpy" or "so_rpy_rotor_drag").
            state_type: MPC state type ("quat" for 13D or "euler" for 12D).
            integrator: Integration method ("rk4" or "euler").
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
        self.state_type = state_type
        self.nx = NX_EULER if state_type == "euler" else NX

        # Create LEAP-C planner with stagewise parameters and LINEAR_LS cost
        planner_cfg = QuadrotorPlannerConfig(
            N_horizon=mpc_horizon,
            dt=mpc_dt,
            param_interface="stagewise",
            cost_type="linear_ls",
            state_type=state_type,
            integrator=integrator,
            n_batch_max=n_batch_max,
            num_threads=num_threads,
            drone_model=drone_model,
            mpc_model=mpc_model,
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
        if state_type == "euler":
            # State: [pos(3), rpy(3), vel(3), drpy(3)] = 12D
            self.register_buffer(
                'w_state_min_log',
                torch.tensor([-1., -1., -1., -2., -2., -2., -1., -1., -1., -1., -1., -1.])
            )
            self.register_buffer(
                'w_state_max_log',
                torch.tensor([2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1.])
            )
        else:
            # State: [pos(3), quat(4:xyzw), vel(3), ang_vel(3)] = 13D
            self.register_buffer(
                'w_state_min_log',
                torch.tensor([-1., -1., -1., -2., -2., -2., -2., -1., -1., -1., -1., -1., -1.])
            )
            self.register_buffer(
                'w_state_max_log',
                torch.tensor([2., 2., 2., 1., 1., 1., 1., 2., 2., 2., 1., 1., 1.])
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
            torch.tensor([-pos_offset_max, -pos_offset_max, -pos_offset_max])
        )
        self.register_buffer(
            'pos_offset_max_buf',
            torch.tensor([pos_offset_max, pos_offset_max, pos_offset_max])
        )
        if state_type == "euler":
            # State refs: [pos(3), rpy(3), vel(3), drpy(3)] = 12D
            # pos indices 0:3 are placeholders (replaced by relative offsets)
            self.register_buffer(
                'yref_state_min',
                torch.tensor([0., 0., 0., -np.pi, -np.pi/2, -np.pi, -5., -5., -5., -10., -10., -10.])
            )
            self.register_buffer(
                'yref_state_max',
                torch.tensor([0., 0., 0., np.pi, np.pi/2, np.pi, 5., 5., 5., 10., 10., 10.])
            )
        else:
            # State refs: [pos(3), quat(4:xyzw), vel(3), ang_vel(3)] = 13D
            # pos indices 0:3 are placeholders (replaced by relative offsets)
            self.register_buffer(
                'yref_state_min',
                torch.tensor([0., 0., 0., -1., -1., -1., -1., -5., -5., -5., -10., -10., -10.])
            )
            self.register_buffer(
                'yref_state_max',
                torch.tensor([0., 0., 0., 1., 1., 1., 1., 5., 5., 5., 10., 10., 10.])
            )
        self.register_buffer(
            'yref_ctrl_min',
            torch.tensor([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min])
        )
        self.register_buffer(
            'yref_ctrl_max',
            torch.tensor([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max])
        )

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
            state: MPC state with shape (B, 13) [pos, quat(xyzw), vel, ang_vel].

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

        # Normalize action to [-1, 1] (avoid division by zero for disabled axes)
        safe_scale = self.action_scale.clone()
        safe_scale[safe_scale == 0] = 1.0
        action_normalized = (u0 - self.action_mean) / safe_scale

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
        nx = self.nx
        w_state_total = nx * self.n_state_stages
        w_ctrl_total = NU * self.n_ctrl_stages
        yref_state_total = nx * self.n_state_stages
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
        w_state_per_stage = w_state_raw.reshape(batch_size, self.n_state_stages, nx)
        w_ctrl_per_stage = w_ctrl_raw.reshape(batch_size, self.n_ctrl_stages, NU)
        yref_state_per_stage = yref_state_raw.reshape(batch_size, self.n_state_stages, nx)
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

        # Get current position from state: state is [pos(3), quat(4:xyzw), vel(3), ang_vel(3)]
        current_pos = state[:, :3]  # (batch, 3)

        # Broadcast current position across all stages and add offset
        # current_pos: (batch, 3) -> (batch, 1, 3) for broadcasting
        yref_pos_absolute = current_pos[:, None, :] + pos_offset  # (batch, n_state_stages, 3)

        # Other state refs (rpy, vel, drpy) use absolute bounds
        yref_other = self.yref_state_min[3:] + yref_state_per_stage[..., 3:] * (self.yref_state_max[3:] - self.yref_state_min[3:])

        # Combine position (absolute) with other state refs
        yref_state = torch.cat([yref_pos_absolute, yref_other], dim=-1)

        # Linear scaling for control references
        yref_ctrl = self.yref_ctrl_min + yref_ctrl_per_stage * (self.yref_ctrl_max - self.yref_ctrl_min)

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
        roll_pitch_max: float = 0.5,
        yaw_max: float = 0.1,
        thrust_min: float = 1.23,
        thrust_max: float = 3.68,
        mass: Optional[float] = None,
        gravity: Optional[float] = None,
        drone_model: str = "cf2x_L250",
        mpc_model: str = "so_rpy",
        state_type: str = "quat",
        integrator: str = "rk4",
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
            mpc_model=mpc_model,
            state_type=state_type,
            integrator=integrator,
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

    def compute(
        self,
        inputs: Mapping[str, torch.Tensor],
        role: str = "",
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Compute actions from observations."""
        obs = inputs["observations"]

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        state = inputs["mpc_state"]
        mean_actions = self.mpc_layer(obs, state)

        log_std = self.log_std_parameter
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_min_log_std, self._g_max_log_std)

        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        return mean_actions, {"log_std": log_std}

