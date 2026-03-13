"""Policy loader for FFN and ACMPC checkpoints.

Loads SKRL-trained policies for use in hardware experiments.
"""

import json
from pathlib import Path
from typing import Optional, Union

import gymnasium
import numpy as np
import torch


def find_checkpoint(checkpoint_path: str) -> Path:
    """Find a .pt checkpoint file from a path.

    Args:
        checkpoint_path: Path to a .pt file or directory containing .pt files.

    Returns:
        Path to the checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint file is found.
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_file() and checkpoint_path.suffix == '.pt':
        return checkpoint_path

    if checkpoint_path.is_dir():
        # Search for .pt files in the directory
        pt_files = list(checkpoint_path.glob('*.pt'))
        if not pt_files:
            # Also check subdirectories
            pt_files = list(checkpoint_path.glob('**/*.pt'))

        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in: {checkpoint_path}")

        # Prefer 'best_agent.pt' if available
        for pt_file in pt_files:
            if pt_file.name == 'best_agent.pt':
                return pt_file

        # Otherwise return the first one found
        return pt_files[0]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    raise FileNotFoundError(f"Invalid checkpoint path: {checkpoint_path}")


def load_policy(
    policy_type: str,
    checkpoint_path: str,
    config: dict,
    device: str = "cpu",
):
    """Load a trained policy and observation preprocessor from checkpoint.

    Args:
        policy_type: Type of policy ("ffn" or "acmpc").
        checkpoint_path: Path to a .pt file or directory containing .pt files.
        config: Environment configuration dictionary.
        device: Device to load model on ("cpu" or "cuda").

    Returns:
        Tuple of (policy, obs_preprocessor). obs_preprocessor may be None if
        no preprocessor was saved in the checkpoint.

    Raises:
        ValueError: If policy_type is not supported.
        ImportError: If required modules are not available.
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    checkpoint_path = find_checkpoint(checkpoint_path)

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

    # Load physical parameters from config, fallback to drone_params
    from drone_models.core import load_params
    drone_params = load_params("so_rpy", drone_model)
    gravity = float(np.abs(drone_params["gravity_vec"][2]))
    mass = config.get('mass', float(drone_params["mass"]))
    thrust_min = config.get('thrust_min', mass * gravity * 0.5)
    thrust_max = config.get('thrust_max', mass * gravity * 1.5)

    action_space = gymnasium.spaces.Box(
        low=np.array([-roll_pitch_max, -roll_pitch_max, -yaw_max, thrust_min], dtype=np.float32),
        high=np.array([roll_pitch_max, roll_pitch_max, yaw_max, thrust_max], dtype=np.float32),
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

    # Load observation preprocessor from checkpoint
    obs_preprocessor = _load_obs_preprocessor(checkpoint_path, obs_dim, device)

    return policy, obs_preprocessor


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

    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    policy.to(device)

    return policy


def _override_action_buffers(policy, roll_pitch_max, yaw_max, thrust_min, thrust_max):
    """Override action_scale/mean buffers that were overwritten by load_state_dict."""
    thrust_mean = (thrust_min + thrust_max) / 2.0
    thrust_scale = (thrust_max - thrust_min) / 2.0
    policy.action_scale = torch.tensor(
        [roll_pitch_max, roll_pitch_max, yaw_max, thrust_scale],
        dtype=torch.float32,
    )
    policy.action_mean = torch.tensor(
        [0.0, 0.0, 0.0, thrust_mean],
        dtype=torch.float32,
    )


def _load_obs_preprocessor(
    checkpoint_path: Path,
    obs_dim: int,
    device: str,
):
    """Load observation preprocessor from SKRL checkpoint.

    The checkpoint stores preprocessor state under
    checkpoint[agent_key]['observation_preprocessor']. We reconstruct a
    PartialRunningStandardScaler with the correct size and load its state.

    Args:
        checkpoint_path: Path to checkpoint file.
        obs_dim: Observation dimension.
        device: Computation device.

    Returns:
        Loaded preprocessor in eval mode, or None if not found.
    """
    from crazyflie_mape_crazyflow.preprocessors import PartialRunningStandardScaler

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        return None

    # Find the first agent key with an observation_preprocessor
    preprocessor_state = None
    for key in checkpoint:
        if key.startswith('_'):
            continue
        agent_data = checkpoint[key]
        if isinstance(agent_data, dict) and 'observation_preprocessor' in agent_data:
            preprocessor_state = agent_data['observation_preprocessor']
            break

    if preprocessor_state is None:
        return None

    # The state_dict contains _scale_indices and _skip_indices as buffers,
    # so we can reconstruct the preprocessor with the right skip_dims
    skip_indices = preprocessor_state.get('_skip_indices', torch.tensor([]))
    skip_dims = skip_indices.tolist() if len(skip_indices) > 0 else None

    preprocessor = PartialRunningStandardScaler(
        size=obs_dim,
        skip_dims=skip_dims,
        device=device,
    )
    preprocessor.load_state_dict(preprocessor_state)
    preprocessor.eval()
    preprocessor.to(device)

    return preprocessor


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
        from crazyflie_mape_crazyflow.policies.leap_c_shared_policy_linear_ls import LeapCSharedGaussianPolicyLinearLS as LeapCSharedGaussianPolicy
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
    pos_offset_max = config.get('pos_offset_max', 0.25)

    # Load physical parameters from config, fallback to drone-models
    from drone_models.core import load_params
    drone_params = load_params("so_rpy", drone_model)
    gravity = float(np.abs(drone_params["gravity_vec"][2]))

    # Use config values if provided, otherwise compute from drone_params
    mass = config.get('mass', float(drone_params["mass"]) + 4.9/1000.0)
    thrust_min = config.get('thrust_min', mass * gravity * 0.5)
    thrust_max = config.get('thrust_max', mass * gravity * 1.5)

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
        thrust_min=thrust_min,
        thrust_max=thrust_max,
        mass=mass,
        gravity=gravity,
        drone_model=drone_model,
        n_batch_max=16,  # Small batch for hardware
        num_threads=4,
        velocity_max=velocity_max,
        activation=activation,
        pos_offset_max=pos_offset_max,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    state_dict = _extract_policy_state_dict(checkpoint)

    policy.load_state_dict(state_dict, strict=False)

    # Override buffers so normalization uses config values, not checkpoint values
    _override_action_buffers(policy, roll_pitch_max, yaw_max, thrust_min, thrust_max)

    policy.eval()
    policy.to(device)

    return policy


def infer_action(policy, observation: np.ndarray, obs_preprocessor=None,
                 mpc_state: Optional[np.ndarray] = None) -> np.ndarray:
    """Run deterministic inference on a policy.

    Args:
        policy: Loaded policy with compute() method.
        observation: Raw observation array, shape (batch, obs_dim) or (obs_dim,).
        obs_preprocessor: Optional observation preprocessor (e.g. PartialRunningStandardScaler).
            If provided, normalizes the observation before passing to the policy.
        mpc_state: Raw MPC state [pos(3), rpy(3), vel(3), drpy(3)], shape (batch, 12) or (12,).
            Required for ACMPC policies. Must contain un-normalized physical values.

    Returns:
        Mean action array, shape (batch, action_dim) or (action_dim,).
    """
    # Ensure batch dimension
    squeeze = False
    if observation.ndim == 1:
        observation = observation[None, :]
        squeeze = True
    if mpc_state is not None and mpc_state.ndim == 1:
        mpc_state = mpc_state[None, :]

    with torch.no_grad():
        obs_tensor = torch.tensor(observation, dtype=torch.float32)

        # Apply observation preprocessor if available
        if obs_preprocessor is not None:
            obs_tensor = obs_preprocessor(obs_tensor)

        inputs = {"observations": obs_tensor}
        if mpc_state is not None:
            inputs["mpc_state"] = torch.tensor(mpc_state, dtype=torch.float32)
        mean_actions, _ = policy.compute(inputs)

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
