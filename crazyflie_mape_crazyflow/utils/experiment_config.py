"""Experiment configuration loading utilities.

This module provides utilities for loading experiment configurations from YAML files
and converting them to the appropriate dataclasses and argument namespaces.
"""

from pathlib import Path

import yaml

from crazyflie_mape_crazyflow.envs import RedVsBlueEnvConfig
from crazyflie_mape_crazyflow.envs.spawn import SpawnFn, create_spawn_fn_from_config


def load_experiment_config(experiment_path: Path) -> dict:
    """Load experiment configuration from YAML file.

    Args:
        experiment_path: Path to the experiment directory containing config.yaml.

    Returns:
        Dictionary containing the experiment configuration.

    Raises:
        FileNotFoundError: If config.yaml is not found.
    """
    config_path = experiment_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def config_to_env_config(config: dict, device: str | None = None) -> RedVsBlueEnvConfig:
    """Convert experiment config dictionary to RedVsBlueEnvConfig.

    Args:
        config: Experiment configuration dictionary.
        device: Override device if specified.

    Returns:
        RedVsBlueEnvConfig instance.

    Note:
        Spawn configuration is handled separately via get_spawn_fn_from_config().
    """
    env_cfg = config.get("environment", {})
    training_cfg = config.get("training", {})
    rewards_cfg = config.get("rewards", {})

    # Build kwargs for RedVsBlueEnvConfig
    kwargs = {}

    # Environment settings
    if "n_pairs" in env_cfg:
        kwargs["n_pairs"] = env_cfg["n_pairs"]
    if "pursuer_strategy" in env_cfg:
        kwargs["pursuer_strategy"] = env_cfg["pursuer_strategy"]
    if "drone_model" in env_cfg:
        kwargs["drone_model"] = env_cfg["drone_model"]

    # Simulation frequencies
    if "sim_freq" in env_cfg:
        kwargs["sim_freq"] = env_cfg["sim_freq"]
    if "mellinger_freq" in env_cfg:
        kwargs["mellinger_freq"] = env_cfg["mellinger_freq"]
    if "control_freq" in env_cfg:
        kwargs["control_freq"] = env_cfg["control_freq"]
    if "episode_length_s" in env_cfg:
        kwargs["episode_length_s"] = env_cfg["episode_length_s"]

    # Collision tolerances
    if "bb_collision_tolerance" in env_cfg:
        kwargs["bb_collision_tolerance"] = env_cfg["bb_collision_tolerance"]
    if "rr_collision_tolerance" in env_cfg:
        kwargs["rr_collision_tolerance"] = env_cfg["rr_collision_tolerance"]
    if "rb_collision_tolerance" in env_cfg:
        kwargs["rb_collision_tolerance"] = env_cfg["rb_collision_tolerance"]

    # Boundary settings
    if "boundary_size" in env_cfg:
        kwargs["boundary_size"] = env_cfg["boundary_size"]
    if "min_altitude" in env_cfg:
        kwargs["min_altitude"] = env_cfg["min_altitude"]
    if "max_altitude" in env_cfg:
        kwargs["max_altitude"] = env_cfg["max_altitude"]

    # Pursuit gains (ProNav)
    if "N_pronav_fb" in env_cfg:
        kwargs["N_pronav_fb"] = env_cfg["N_pronav_fb"]
    if "N_pronav_ff" in env_cfg:
        kwargs["N_pronav_ff"] = env_cfg["N_pronav_ff"]
    if "velocity_closure_threshold" in env_cfg:
        kwargs["velocity_closure_threshold"] = env_cfg["velocity_closure_threshold"]

    # Pursuit gains (Augmented ProNav)
    if "N_gain" in env_cfg:
        kwargs["N_gain"] = env_cfg["N_gain"]
    if "V_min" in env_cfg:
        kwargs["V_min"] = env_cfg["V_min"]
    if "K_v" in env_cfg:
        kwargs["K_v"] = env_cfg["K_v"]

    # Pursuit gains (Pure Pursuit)
    if "pp_k_pxy" in env_cfg:
        kwargs["pp_k_pxy"] = env_cfg["pp_k_pxy"]
    if "pp_k_vxy" in env_cfg:
        kwargs["pp_k_vxy"] = env_cfg["pp_k_vxy"]
    if "pp_k_pz" in env_cfg:
        kwargs["pp_k_pz"] = env_cfg["pp_k_pz"]
    if "pp_k_vz" in env_cfg:
        kwargs["pp_k_vz"] = env_cfg["pp_k_vz"]

    # Target assignment
    if "random_target_assignment" in env_cfg:
        kwargs["random_target_assignment"] = env_cfg["random_target_assignment"]

    # Physical parameters (mass override)
    if "mass" in env_cfg:
        kwargs["mass"] = env_cfg["mass"]

    # Domain randomization
    if "randomize_mass" in env_cfg:
        kwargs["randomize_mass"] = env_cfg["randomize_mass"]
    if "randomize_inertia" in env_cfg:
        kwargs["randomize_inertia"] = env_cfg["randomize_inertia"]
    if "mass_randomization_std" in env_cfg:
        kwargs["mass_randomization_std"] = env_cfg["mass_randomization_std"]
    if "inertia_randomization_std" in env_cfg:
        kwargs["inertia_randomization_std"] = env_cfg["inertia_randomization_std"]

    # Disturbance forces/torques
    if "enable_disturbance" in env_cfg:
        kwargs["enable_disturbance"] = env_cfg["enable_disturbance"]
    if "disturbance_force_std" in env_cfg:
        kwargs["disturbance_force_std"] = env_cfg["disturbance_force_std"]
    if "disturbance_torque_std" in env_cfg:
        kwargs["disturbance_torque_std"] = env_cfg["disturbance_torque_std"]

    # Reward settings
    if "capture" in rewards_cfg:
        kwargs["reward_capture"] = rewards_cfg["capture"]
    if "escape" in rewards_cfg:
        kwargs["reward_escape"] = rewards_cfg["escape"]
    if "red_crash" in rewards_cfg:
        kwargs["reward_red_crash"] = rewards_cfg["red_crash"]
    if "blue_crash" in rewards_cfg:
        kwargs["reward_blue_crash"] = rewards_cfg["blue_crash"]
    if "boundary" in rewards_cfg:
        kwargs["reward_boundary"] = rewards_cfg["boundary"]
    if "alive" in rewards_cfg:
        kwargs["reward_alive"] = rewards_cfg["alive"]
    if "pursuer_proximity" in rewards_cfg:
        kwargs["reward_pursuer_proximity"] = rewards_cfg["pursuer_proximity"]
    if "pursuer_proximity_decay" in rewards_cfg:
        kwargs["reward_pursuer_proximity_decay"] = rewards_cfg["pursuer_proximity_decay"]

    # Angle and action penalties
    if "angle_coef" in rewards_cfg:
        kwargs["reward_angle_coef"] = rewards_cfg["angle_coef"]
    if "velocity_coef" in rewards_cfg:
        kwargs["reward_velocity_coef"] = rewards_cfg["velocity_coef"]
    if "action_coef" in rewards_cfg:
        kwargs["reward_action_coef"] = rewards_cfg["action_coef"]
    if "action_smoothness_thrust" in rewards_cfg:
        kwargs["reward_action_smoothness_thrust"] = rewards_cfg["action_smoothness_thrust"]
    if "action_smoothness_rpy" in rewards_cfg:
        kwargs["reward_action_smoothness_rpy"] = rewards_cfg["action_smoothness_rpy"]

    # Training settings (n_worlds and device)
    if "n_worlds" in training_cfg:
        kwargs["n_worlds"] = training_cfg["n_worlds"]

    # Device: command line override > config > default
    if device is not None:
        kwargs["device"] = device
    elif "device" in training_cfg:
        kwargs["device"] = training_cfg["device"]

    return RedVsBlueEnvConfig(**kwargs)


def get_spawn_fn_from_config(config: dict) -> SpawnFn:
    """Create spawn function from experiment configuration.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        Spawn function with signature (key, N, B, R) -> (blue_pos, red_pos).
    """
    spawn_cfg = config.get("environment", {}).get("spawn", {})
    return create_spawn_fn_from_config(spawn_cfg)


def get_training_config(config: dict) -> dict:
    """Extract training configuration from experiment config.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        Dictionary with training parameters.
    """
    training_cfg = config.get("training", {})

    result = {
        "timesteps": training_cfg.get("timesteps", 1_000_000),
        "n_worlds": training_cfg.get("n_worlds", 256),
        "device": training_cfg.get("device", "cpu"),
        "rollouts": training_cfg.get("rollouts", 16),
        "learning_epochs": training_cfg.get("learning_epochs", 8),
        "mini_batches": training_cfg.get("mini_batches", 4),
        "learning_rate": training_cfg.get("learning_rate", 3e-4),
        "gamma": training_cfg.get("gamma", 0.99),
        "gae_lambda": training_cfg.get("gae_lambda", 0.95),
        "grad_norm_clip": training_cfg.get("grad_norm_clip", 0.5),
        "entropy_loss_scale": training_cfg.get("entropy_loss_scale", 0.01),
        "value_loss_scale": training_cfg.get("value_loss_scale", 2.0),
        "ratio_clip": training_cfg.get("ratio_clip", 0.2),
        "value_clip": training_cfg.get("value_clip", 0.2),
        "kl_threshold": training_cfg.get("kl_threshold", 0.0),
        # Value preprocessor (optional)
        # Supported: "RunningStandardScaler" or None
        "value_preprocessor": training_cfg.get("value_preprocessor", None),
        # Learning rate scheduler (optional)
        # Supported: "KLAdaptiveLR", "StepLR", or None
        "learning_rate_scheduler": training_cfg.get("learning_rate_scheduler", None),
        "learning_rate_scheduler_kwargs": training_cfg.get("learning_rate_scheduler_kwargs", {}),
    }

    return result


def get_policy_config(config: dict, policy_type: str) -> dict:
    """Extract policy configuration from experiment config.

    Args:
        config: Experiment configuration dictionary.
        policy_type: "acmpc" or "ffn".

    Returns:
        Dictionary with policy parameters.
    """
    policy_cfg = config.get("policy", {})

    if policy_type == "acmpc":
        # Get cost_net_sizes, default to [256, 256]
        cost_net_sizes = policy_cfg.get("cost_net_sizes", [256, 256])
        value_net_sizes = policy_cfg.get("value_net_sizes", [256, 256])
        return {
            # MPC settings
            "mpc_horizon": policy_cfg.get("mpc_horizon", 2),
            "mpc_dt": policy_cfg.get("mpc_dt", 0.01),
            "mpc_velocity_max": policy_cfg.get("mpc_velocity_max", None),
            # Control limits
            "roll_pitch_max": policy_cfg.get("roll_pitch_max", 0.5),
            "yaw_max": policy_cfg.get("yaw_max", 0.5),
            # Network architecture
            "cost_net_sizes": cost_net_sizes,
            "value_net_sizes": value_net_sizes,
            # Activation functions
            "cost_net_activation": policy_cfg.get("cost_net_activation", "relu"),
            "value_activation": policy_cfg.get("value_activation", "relu"),
            # Exploration
            "initial_log_std": policy_cfg.get("initial_log_std", -1.2),
        }
    elif policy_type == "ffn":
        policy_net_sizes = policy_cfg.get("policy_net_sizes", [256, 256])
        value_net_sizes = policy_cfg.get("value_net_sizes", [256, 256])
        return {
            # Control limits
            "roll_pitch_max": policy_cfg.get("roll_pitch_max", 0.5),
            "yaw_max": policy_cfg.get("yaw_max", 0.5),
            # Network architecture
            "policy_net_sizes": policy_net_sizes,
            "value_net_sizes": value_net_sizes,
            # Activation functions
            "policy_activation": policy_cfg.get("policy_activation", "relu"),
            "value_activation": policy_cfg.get("value_activation", "relu"),
            # Exploration
            "initial_log_std": policy_cfg.get("initial_log_std", -1.2),
        }
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def find_experiment_path(experiment_name: str, policy_type: str) -> Path:
    """Find the experiment directory path.

    Args:
        experiment_name: Name of the experiment (e.g., "deterministic_spawn").
        policy_type: "acmpc" or "ffn".

    Returns:
        Path to the experiment directory.

    Raises:
        FileNotFoundError: If experiment directory is not found.
    """
    # Get the project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent.parent

    experiment_path = project_root / "results" / policy_type / experiment_name

    if not experiment_path.exists():
        raise FileNotFoundError(
            f"Experiment not found: {experiment_path}\n"
            f"Available experiments in results/{policy_type}/:\n"
            + "\n".join(f"  - {d.name}" for d in (project_root / "results" / policy_type).iterdir() if d.is_dir())
        )

    return experiment_path
