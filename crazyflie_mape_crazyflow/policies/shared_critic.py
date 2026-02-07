"""Shared critic network for MAPPO centralized training.

This module provides a centralized critic that uses shared state information
for estimating state values in multi-agent settings.
"""

from typing import Mapping, Tuple, Union

import gymnasium
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, Model


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


class SharedCritic(DeterministicMixin, Model):
    """SKRL Deterministic value network for MAPPO critic.

    This value network estimates the state value for centralized training
    with decentralized execution (CTDE) in MAPPO. It uses the shared state
    that includes information about all agents.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: Union[str, torch.device] = "cuda",
        clip_actions: bool = False,
        hidden_dim: int = 256,
        activation: str = "relu",
        **kwargs,
    ):
        """Initialize the value network.

        Args:
            observation_space: Environment observation space (shared state).
            action_space: Environment action space.
            device: Computation device.
            clip_actions: Whether to clip actions (unused for value).
            hidden_dim: Hidden layer dimension.
            activation: Hidden layer activation function name
                (relu, tanh, elu, leaky_relu, gelu).
        """
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        obs_dim = gymnasium.spaces.flatdim(observation_space)

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, 1),
        )

    def compute(
        self,
        inputs: Mapping[str, torch.Tensor],
        role: str = "",
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Compute state value.

        Args:
            inputs: Dictionary containing shared states.
            role: Model role (unused).

        Returns:
            Tuple of (value, outputs).
        """
        # For MAPPO, use shared state for centralized critic
        if "states" in inputs:
            state = inputs["states"]
        else:
            state = inputs.get("shared_states", inputs.get("observations", None))

        if state is None:
            raise ValueError("No state found in inputs")

        value = self.value_net(state)

        return value, {}
