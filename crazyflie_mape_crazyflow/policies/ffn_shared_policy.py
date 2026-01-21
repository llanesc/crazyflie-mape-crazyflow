"""SKRL feedforward neural network policy with parameter sharing.

This module provides a simple feedforward neural network policy for
multi-agent reinforcement learning without MPC.

Action: [roll, pitch, yaw, thrust] normalized to [-1, 1]
"""

from typing import Mapping, Optional, Sequence, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model


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


class FFNSharedGaussianPolicy(GaussianMixin, Model):
    """SKRL Gaussian policy using a feedforward neural network.

    This policy is designed for multi-agent MAPPO training where all agents
    share the same policy network (parameter sharing). The policy uses
    a simple feedforward neural network to compute actions.

    Architecture:
        - Hidden layers: [256, 256] (configurable)
        - Hidden activation: Configurable (default: ReLU)
        - Output activation: Tanh (bounds output to [-1, 1])
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
        hidden_sizes: Tuple[int, ...] = (256, 256),
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
            hidden_sizes: Tuple of hidden layer sizes.
            activation: Hidden layer activation function name
                (relu, tanh, elu, leaky_relu, gelu).
        """
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        # Get dimensions
        obs_dim = gymnasium.spaces.flatdim(observation_space)
        action_dim = gymnasium.spaces.flatdim(action_space)

        # Build feedforward network
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())  # Bound output to [-1, 1]

        self.policy_net = nn.Sequential(*layers)

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

        # Get mean action from network
        mean_actions = self.policy_net(obs)

        # Get log std
        log_std = self.log_std_parameter
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)

        # Store for distribution computation
        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        return mean_actions, log_std, {}
