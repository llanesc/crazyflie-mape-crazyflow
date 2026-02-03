"""Policy loader for FFN and ACMPC checkpoints.

Loads SKRL-trained policies for use in hardware experiments.
"""

import json
from pathlib import Path
from typing import Optional, Union

import gymnasium
import numpy as np
import torch


def load_policy(
    policy_type: str,
    checkpoint_path: str,
    config: dict,
    device: str = "cpu",
):
    """Load a trained policy from checkpoint.

    Args:
        policy_type: Type of policy ("ffn" or "acmpc").
        checkpoint_path: Path to the SKRL checkpoint (.pt file).
        config: Environment configuration dictionary.
        device: Device to load model on ("cpu" or "cuda").

    Returns:
        Loaded policy in eval mode.

    Raises:
        ValueError: If policy_type is not supported.
        ImportError: If required modules are not available.
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Get dimensions from config
    n_pairs = config.get('n_pairs', 2)
    obs_dim = config.get('observation_space', {}).get('per_agent_obs_dim', 46)
    roll_pitch_max = config.get('roll_pitch_max', 0.5)
    yaw_max = config.get('yaw_max', 0.1)
    drone_model = config.get('drone_model', 'cf2x_T350')

    # Create observation and action spaces
    observation_space = gymnasium.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dim,),
        dtype=np.float32
    )

    # Load drone params for action space bounds
    from drone_models.core import load_params
    drone_params = load_params("so_rpy", drone_model)
    # min_thrust = float(drone_params["thrust_min"]) * 4
    # max_thrust = float(drone_params["thrust_max"]) * 4
    mass = float(drone_params["mass"]) + 4.9/1000.0
    gravity = float(np.abs(drone_params["gravity_vec"][2]))
    # self.min_thrust = float(drone_params["thrust_min"]) * 4  # Per motor -> collective
    # self.max_thrust = float(drone_params["thrust_max"]) * 4
    min_thrust = mass * gravity * 0.5 
    max_thrust = mass * gravity * 1.5

    action_space = gymnasium.spaces.Box(
        low=np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, min_thrust], dtype=np.float32),
        high=np.array([roll_pitch_max, roll_pitch_max, yaw_max, max_thrust], dtype=np.float32),
        dtype=np.float32
    )

    if policy_type.lower() == "ffn":
        policy = _load_ffn_policy(
            observation_space,
            action_space,
            checkpoint_path,
            config,
            device,
        )
    elif policy_type.lower() == "acmpc":
        policy = _load_acmpc_policy(
            observation_space,
            action_space,
            checkpoint_path,
            config,
            device,
        )
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}. Use 'ffn' or 'acmpc'.")

    return policy


def _load_ffn_policy(
    observation_space: gymnasium.Space,
    action_space: gymnasium.Space,
    checkpoint_path: Path,
    config: dict,
    device: str,
):
    """Load FFN policy from checkpoint.

    Args:
        observation_space: Observation space.
        action_space: Action space.
        checkpoint_path: Path to checkpoint.
        config: Environment configuration.
        device: Computation device.

    Returns:
        FFNSharedGaussianPolicy in eval mode.
    """
    from crazyflie_mape_crazyflow.policies.ffn_shared_policy import FFNSharedGaussianPolicy

    # Get architecture from config
    hidden_sizes = tuple(config.get('policy_net_sizes', [256, 256]))
    activation = config.get('policy_activation', 'relu')

    # Create policy
    policy = FFNSharedGaussianPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=0.0,
        hidden_sizes=hidden_sizes,
        activation=activation,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    state_dict = _extract_policy_state_dict(checkpoint)

    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to(device)

    return policy


def _extract_policy_state_dict(checkpoint: dict) -> dict:
    """Extract policy state_dict from various checkpoint formats.

    Handles:
        - SKRL multi-agent format: checkpoint['blue_0']['policy']
        - SKRL single-agent format: checkpoint['policy']
        - Direct state_dict: checkpoint contains weight keys directly
        - model_state_dict format: checkpoint['model_state_dict']

    Args:
        checkpoint: Loaded checkpoint dictionary.

    Returns:
        Policy state_dict.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint

    # SKRL multi-agent format (e.g., blue_0, blue_1)
    agent_keys = [k for k in checkpoint.keys() if k.startswith(('blue_', 'red_', 'agent_'))]
    if agent_keys:
        # Use first agent (parameter sharing means all are the same)
        agent_data = checkpoint[agent_keys[0]]
        if isinstance(agent_data, dict) and 'policy' in agent_data:
            return agent_data['policy']
        return agent_data

    # SKRL single-agent format
    if 'policy' in checkpoint:
        return checkpoint['policy']

    # PyTorch standard format
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']

    # Direct state_dict (contains weight keys)
    return checkpoint


def _load_acmpc_policy(
    observation_space: gymnasium.Space,
    action_space: gymnasium.Space,
    checkpoint_path: Path,
    config: dict,
    device: str,
):
    """Load ACMPC policy from checkpoint.

    Requires leap_c and acados to be installed.

    Args:
        observation_space: Observation space.
        action_space: Action space.
        checkpoint_path: Path to checkpoint.
        config: Environment configuration.
        device: Computation device.

    Returns:
        LeapCSharedGaussianPolicy in eval mode.

    Raises:
        ImportError: If leap_c or acados is not available.
    """
    try:
        from crazyflie_mape_crazyflow.policies.leap_c_shared_policy import LeapCSharedGaussianPolicy
    except ImportError as e:
        raise ImportError(
            "ACMPC policy requires leap_c and acados to be installed. "
            f"Original error: {e}"
        )

    # Get architecture from config
    hidden_dim = config.get('cost_net_sizes', [256, 256])[0]  # Use first hidden size
    activation = config.get('cost_net_activation', 'relu')
    mpc_horizon = config.get('mpc_horizon', 2)
    mpc_dt = config.get('mpc_dt', 0.01)
    velocity_max = config.get('mpc_velocity_max', None)
    roll_pitch_max = config.get('roll_pitch_max', 0.5)
    yaw_max = config.get('yaw_max', 0.1)
    drone_model = config.get('drone_model', 'cf2x_T350')

    # Create policy
    policy = LeapCSharedGaussianPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=0.0,
        mpc_horizon=mpc_horizon,
        mpc_dt=mpc_dt,
        hidden_dim=hidden_dim,
        roll_pitch_max=roll_pitch_max,
        yaw_max=yaw_max,
        drone_model=drone_model,
        n_batch_max=16,  # Small batch for hardware
        num_threads=4,
        velocity_max=velocity_max,
        activation=activation,
        verbose=False,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    state_dict = _extract_policy_state_dict(checkpoint)

    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to(device)

    return policy


def infer_action(policy, observation: np.ndarray) -> np.ndarray:
    """Run deterministic inference on a policy.

    Args:
        policy: Loaded policy with compute() method.
        observation: Observation array, shape (batch, obs_dim) or (obs_dim,).

    Returns:
        Mean action array, shape (batch, action_dim) or (action_dim,).
    """
    # Ensure batch dimension
    squeeze = False
    if observation.ndim == 1:
        observation = observation[None, :]
        squeeze = True

    with torch.no_grad():
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        inputs = {"states": obs_tensor}
        mean_actions, _, _ = policy.compute(inputs)

        if isinstance(mean_actions, torch.Tensor):
            actions = mean_actions.numpy()
        else:
            actions = mean_actions

    if squeeze:
        actions = actions.squeeze(0)

    return actions


def load_config(config_path: str) -> dict:
    """Load environment configuration from JSON file.

    Args:
        config_path: Path to environment_config.json.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config file is invalid JSON.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config
