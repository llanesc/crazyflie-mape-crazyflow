"""Wrappers for multi-agent environments."""

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces


@jax.jit
def _jit_rescale_actions(actions: jnp.ndarray, scale: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled action rescaling from [-1, 1] to physical bounds.

    Args:
        actions: Normalized actions in [-1, 1], shape (n_worlds, n_agents, action_dim).
        scale: Scale factors, shape (action_dim,).
        mean: Mean offsets, shape (action_dim,).

    Returns:
        Rescaled actions in physical bounds.
    """
    clipped = jnp.clip(actions, -1.0, 1.0)
    return clipped * scale + mean


class RescaleActionWrapper(gym.Wrapper):
    """Rescale actions from [-1, 1] to the environment's physical action space.

    This wrapper is designed for multi-agent environments with dict-based action spaces.
    It converts normalized actions from policies (in [-1, 1]) to the physical action
    bounds expected by the environment.

    Uses JAX for efficient JIT-compiled rescaling.

    Example usage:
        env = RedVsBlueEnv(cfg)
        env = RescaleActionWrapper(env)
        # Now env.action_space is [-1, 1] for all agents
        # Policy outputs [-1, 1], wrapper rescales to physical bounds
    """

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap. Must have dict-based action spaces
                 with Box spaces for each agent.
        """
        super().__init__(env)

        # Build normalized action space and compute rescaling parameters
        normalized_action_space = {}
        sample_space = None

        for agent, space in env.action_space.items():
            if not isinstance(space, spaces.Box):
                raise ValueError(f"RescaleActionWrapper only supports Box action spaces, got {type(space)}")

            sample_space = space

            # Create normalized action space [-1, 1]
            normalized_action_space[agent] = spaces.Box(
                low=-np.ones_like(space.low),
                high=np.ones_like(space.high),
                dtype=space.dtype,
            )

        # Compute scale and mean for rescaling: physical = normalized * scale + mean
        # All agents have the same action space, so we only need one set of scale/mean
        self._action_scale = jnp.array((sample_space.high - sample_space.low) / 2.0)
        self._action_mean = jnp.array((sample_space.high + sample_space.low) / 2.0)

        self.action_space = spaces.Dict(normalized_action_space)
        self.action_spaces = self.action_space  # PettingZoo alias

        # Forward multi-agent attributes from underlying env
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self.num_agents = env.num_agents
        self.cfg = env.cfg  # Forward config for eval scripts

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        """Execute one environment step with rescaled actions.

        Args:
            actions: Dict mapping agent names to normalized actions in [-1, 1].

        Returns:
            observations, rewards, terminated, truncated, info
        """
        rescaled_actions = self._rescale_actions(actions)
        return self.env.step(rescaled_actions)

    def _rescale_actions(self, actions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Rescale actions from [-1, 1] to physical bounds using JAX.

        Args:
            actions: Dict of normalized actions in [-1, 1].

        Returns:
            Dict of actions rescaled to physical bounds.
        """
        # Stack actions into array for efficient JAX processing
        # Shape: (n_worlds, n_agents, action_dim)
        action_list = [actions[agent] for agent in self.possible_agents]
        stacked = jnp.stack(action_list, axis=1)

        # JIT-compiled rescaling
        rescaled = _jit_rescale_actions(stacked, self._action_scale, self._action_mean)

        # Convert back to dict with numpy arrays
        return {
            agent: np.asarray(rescaled[:, i])
            for i, agent in enumerate(self.possible_agents)
        }

    def state(self) -> np.ndarray:
        """Get global/shared state for centralized critic (SKRL interface).

        Returns:
            Shared state array for all agents.
        """
        return self.env.state()
