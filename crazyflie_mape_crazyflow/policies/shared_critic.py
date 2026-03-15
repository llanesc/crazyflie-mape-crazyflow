"""Shared critic network for MAPPO centralized training.

This module provides a centralized critic that uses shared state information
for estimating state values in multi-agent settings.
"""

from typing import List, Mapping, Optional, Tuple, Union

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
        binary_dims: Optional[List[int]] = None,
        binary_embed_dim: int = 8,
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
            binary_dims: Indices of binary/one-hot dimensions in the state.
                These are projected through a learned linear embedding instead
                of being passed directly, avoiding scale mismatch with the
                normalized continuous features.
            binary_embed_dim: Output dimension of the binary embedding layer.
        """
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        obs_dim = gymnasium.spaces.flatdim(observation_space)

        binary_dims = sorted(binary_dims) if binary_dims else []
        continuous_dims = [i for i in range(obs_dim) if i not in set(binary_dims)]
        n_binary = len(binary_dims)

        self.register_buffer("_binary_idx", torch.tensor(binary_dims, dtype=torch.long))
        self.register_buffer("_continuous_idx", torch.tensor(continuous_dims, dtype=torch.long))

        if n_binary > 0:
            self.binary_embed = nn.Linear(n_binary, binary_embed_dim)
            value_net_input_dim = len(continuous_dims) + binary_embed_dim
        else:
            self.binary_embed = None
            value_net_input_dim = obs_dim

        self.value_net = nn.Sequential(
            nn.Linear(value_net_input_dim, hidden_dim),
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

        if self.binary_embed is not None:
            state = torch.cat([
                state[:, self._continuous_idx],
                self.binary_embed(state[:, self._binary_idx]),
            ], dim=-1)

        value = self.value_net(state)

        return value, {}
