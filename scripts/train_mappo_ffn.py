#!/usr/bin/env python3
"""Training script for Red vs Blue environment using MAPPO with FFN policy.

This script trains blue evaders using MAPPO with a feedforward neural network
policy against scripted red pursuers.

Usage:
    python scripts/train_mappo_ffn.py --experiment deterministic_spawn
"""

import argparse
import os
from pathlib import Path

import yaml


# Parse experiment and load config early to set JAX platform before imports
def _get_device_from_config():
    """Parse experiment argument and load config to get device."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment", type=str, required=True)
    args, _ = parser.parse_known_args()

    # Find and load config
    project_root = Path(__file__).parent.parent
    config_path = project_root / "results" / "ffn" / args.experiment / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("training", {}).get("device", "cpu")
    return "cpu"


_device = _get_device_from_config()
if _device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    # Suppress JAX CUDA plugin discovery errors when running on CPU
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
# For cuda, let JAX auto-detect

import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env
from torch.optim.lr_scheduler import StepLR

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig, RescaleActionWrapper
from crazyflie_mape_crazyflow.policies import (
    FFNSharedGaussianPolicy,
    SharedCritic,
)
from crazyflie_mape_crazyflow.envs.spawn import create_spawn_fn_from_config
from crazyflie_mape_crazyflow.utils import (
    load_experiment_config,
    config_to_env_config,
    get_spawn_fn_from_config,
    get_training_config,
    get_policy_config,
    find_experiment_path,
)
from crazyflie_mape_crazyflow.utils.curriculum import (
    CurriculumManager,
    CurriculumConfig,
    CurriculumLevel,
    load_curriculum_config,
)


class TerminationLoggingWrapper:
    """Wrapper that logs termination and collision events to TensorBoard via SKRL agent.

    Tracks:
    - Termination events: all_blue_dead, all_red_dead, out_of_bounds, max_steps
    - Collision events: bb_crash, rr_crash, br_crash

    Also handles curriculum learning by tracking blue wins and updating
    environment parameters when advancing levels.
    """

    def __init__(
        self,
        env,
        raw_env,
        log_interval: int = 100,
        curriculum_manager: CurriculumManager | None = None,
    ):
        """Initialize the wrapper.

        Args:
            env: Wrapped environment (SKRL PettingZoo wrapper).
            raw_env: Raw RedVsBlueEnv for accessing termination info directly.
            log_interval: Interval (in steps) for logging to TensorBoard.
            curriculum_manager: Optional curriculum manager for progressive training.
        """
        # Use object.__setattr__ to avoid triggering our custom __getattr__
        object.__setattr__(self, '_env', env)
        object.__setattr__(self, '_raw_env', raw_env)
        object.__setattr__(self, '_agent', None)  # Set later via set_agent()
        object.__setattr__(self, '_log_interval', log_interval)
        object.__setattr__(self, '_termination_counts', defaultdict(float))
        object.__setattr__(self, '_collision_counts', defaultdict(float))
        object.__setattr__(self, '_step_count', 0)
        object.__setattr__(self, '_total_episodes', 0)
        object.__setattr__(self, '_curriculum_manager', curriculum_manager)
        object.__setattr__(self, '_n_worlds', raw_env.cfg.n_worlds)

        # Register curriculum level change callback
        if curriculum_manager is not None:
            curriculum_manager.on_level_change(self._on_curriculum_level_change)

    def _on_curriculum_level_change(self, level_idx: int, level_config: CurriculumLevel):
        """Handle curriculum level change by updating environment params."""
        # Create spawn function if spawn config is provided
        spawn_fn = None
        if level_config.spawn:
            spawn_fn = create_spawn_fn_from_config(level_config.spawn)

        # Pass all params from the level config to the environment
        self._raw_env.update_curriculum_params(
            spawn_fn=spawn_fn,
            **level_config.params,
        )

    def set_agent(self, agent):
        """Set the SKRL agent for logging."""
        object.__setattr__(self, '_agent', agent)

    def __getattr__(self, name):
        """Forward attribute access to wrapped environment."""
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        """Reset environment and tracking."""
        return self._env.reset(*args, **kwargs)

    def step(self, actions):
        """Step environment and track termination/collision events."""
        obs, rewards, terminated, truncated, info = self._env.step(actions)

        # Get all termination events directly from the raw environment
        # The raw env stores last_termination_events as RATES (normalized by n_worlds)
        # We convert back to counts by multiplying by n_worlds
        term_events = self._raw_env.last_termination_events
        n_worlds = self._n_worlds

        # Collision events (not termination reasons, just individual agent collisions)
        # Divide by 2 because each collision involves 2 agents
        self._collision_counts["collision/bb_crash"] += term_events["bb_crash"] * n_worlds / 2
        self._collision_counts["collision/rr_crash"] += term_events["rr_crash"] * n_worlds / 2
        self._collision_counts["collision/br_crash"] += term_events["br_crash"] * n_worlds

        # Termination events (episode-ending reasons)
        self._termination_counts["termination/out_of_bounds"] += term_events["out_of_bounds"] * n_worlds
        self._termination_counts["termination/all_blue_dead"] += term_events["all_blue_dead"] * n_worlds
        self._termination_counts["termination/all_red_dead"] += term_events.get("all_red_dead", 0) * n_worlds
        self._termination_counts["termination/max_steps"] += term_events["max_steps"] * n_worlds

        # Track total episodes
        n_all_red_dead = int(round(term_events.get("all_red_dead", 0) * n_worlds))
        n_all_blue_dead = int(round(term_events["all_blue_dead"] * n_worlds))
        n_max_steps = int(round(term_events["max_steps"] * n_worlds))
        n_episodes_ended = n_all_red_dead + n_all_blue_dead + n_max_steps
        if n_episodes_ended > 0:
            object.__setattr__(self, '_total_episodes', self._total_episodes + n_episodes_ended)

        self._step_count += 1

        # Log to TensorBoard at intervals (only if agent is set)
        if self._agent is not None and self._step_count % self._log_interval == 0:
            self._log_events()

        return obs, rewards, terminated, truncated, info

    def _log_events(self):
        """Log accumulated termination and collision events to TensorBoard via SKRL agent."""
        # Log termination counts and rates (rates sum to 1 as proportions)
        total_terminations = sum(self._termination_counts.values())
        for key, count in self._termination_counts.items():
            self._agent.track_data(key, count)
            rate_key = key.replace("termination/", "termination_rate/")
            # Rate is proportion of total terminations (sums to 1)
            rate = count / total_terminations if total_terminations > 0 else 0.0
            self._agent.track_data(rate_key, rate)

        # Log collision counts and rates (per-episode rates)
        total_collisions = sum(self._collision_counts.values())
        for key, count in self._collision_counts.items():
            self._agent.track_data(key, count)
            rate_key = key.replace("collision/", "collision_rate/")
            # Rate is proportion of total collisions
            rate = count / total_collisions if total_collisions > 0 else 0.0
            self._agent.track_data(rate_key, rate)

        # Log total episodes
        self._agent.track_data("episode/total", self._total_episodes)

        # Compute blue win rate and check for curriculum advancement
        if self._curriculum_manager is not None:
            n_blue_wins = self._termination_counts["termination/all_red_dead"]
            # Win rate = blue wins / total episodes (blue wins + blue losses + max_steps)
            win_rate = n_blue_wins / total_terminations if total_terminations > 0 else 0.0

            # Check for advancement
            advanced = self._curriculum_manager.check_advancement(win_rate)
            if advanced:
                level = self._curriculum_manager.current_level
                level_name = self._curriculum_manager.current_level_config.name
                print(f"[Curriculum] Advanced to level {level} ({level_name}) at win rate {win_rate:.2%}")

            # Log curriculum stats
            self._agent.track_data("curriculum/level", self._curriculum_manager.current_level)
            self._agent.track_data("curriculum/win_rate", win_rate)
            self._agent.track_data("curriculum/total_episodes", self._total_episodes)

        # Reset counters
        object.__setattr__(self, '_termination_counts', defaultdict(float))
        object.__setattr__(self, '_collision_counts', defaultdict(float))

    def close(self):
        """Close environment."""
        return self._env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MAPPO with FFN policy on Red vs Blue environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/train_mappo_ffn.py --experiment deterministic_spawn
        """,
    )

    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name (e.g., 'deterministic_spawn')")

    return parser.parse_args()


def generate_run_id() -> str:
    """Generate a unique run identifier.

    Uses full timestamp to create a chronologically sortable ID.
    Format: YYYYMMDDHHmmss (e.g., "20240119143022").

    Returns:
        Unique run identifier string that increases with time.
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")


def main():
    """Main training function."""
    args = parse_args()

    # Find experiment path and load config
    experiment_path = find_experiment_path(args.experiment, "ffn")
    config = load_experiment_config(experiment_path)

    print(f"Loading experiment config from: {experiment_path / 'config.yaml'}")

    # Get training and policy configs
    training_cfg = get_training_config(config)
    policy_cfg = get_policy_config(config, "ffn")

    # Get settings from config
    device_str = training_cfg["device"]
    timesteps = training_cfg["timesteps"]
    n_worlds = training_cfg["n_worlds"]

    # Set torch device
    device = torch.device(device_str)

    # Create environment configuration from experiment config
    env_cfg = config_to_env_config(config, device=device_str)
    spawn_fn = get_spawn_fn_from_config(config)

    # Load curriculum configuration (if enabled)
    curriculum_cfg = load_curriculum_config(config)
    curriculum_manager = None
    if curriculum_cfg is not None:
        curriculum_manager = CurriculumManager(curriculum_cfg)
        print(f"[Curriculum] Enabled with {len(curriculum_cfg.levels)} levels")
        print(f"[Curriculum] Advance threshold: {curriculum_cfg.advance_threshold}")
        print(f"[Curriculum] Window size: {curriculum_cfg.window_size}")
        print(f"[Curriculum] Starting at level 0: {curriculum_cfg.levels[0].name}")

        # Apply initial level params to environment config
        initial_params = curriculum_manager.get_env_params()
        for param_name, param_value in initial_params.items():
            if param_name != "spawn" and hasattr(env_cfg, param_name):
                setattr(env_cfg, param_name, param_value)

        # Override spawn function if specified in initial level
        if "spawn" in initial_params and initial_params["spawn"]:
            spawn_fn = create_spawn_fn_from_config(initial_params["spawn"])

    # Create results directory inside experiment folder
    results_dir = experiment_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_id()
    run_name = f"run_{run_id}"
    # SKRL will create the run directory, we just define the path for saving configs
    experiment_dir = results_dir / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Track start time
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Experiment: {args.experiment}")
    print(f"Results directory: {experiment_dir}")

    # Create environment first (needed for obs_dim in config)
    env = RedVsBlueEnv(cfg=env_cfg, spawn_fn=spawn_fn)
    raw_env = env

    # Save environment config (parameters that define the environment/policy structure)
    environment_config = {
        "policy_type": "ffn",
        "experiment_name": args.experiment,
        "n_pairs": env_cfg.n_pairs,
        "drone_model": env_cfg.drone_model,
        "pursuer_strategy": env_cfg.pursuer_strategy,
        # Control limits
        "roll_pitch_max": policy_cfg["roll_pitch_max"],
        "yaw_max": policy_cfg["yaw_max"],
        # Network dimensions
        "policy_net_sizes": policy_cfg["policy_net_sizes"],
        "value_net_sizes": policy_cfg["value_net_sizes"],
        # Network activations
        "policy_activation": policy_cfg["policy_activation"],
        "value_activation": policy_cfg["value_activation"],
        # Frequencies
        "control_freq": env_cfg.control_freq,
        "mellinger_freq": env_cfg.mellinger_freq,
        "sim_freq": env_cfg.sim_freq,
        # Crash tolerances
        "bb_crash_tolerance": env_cfg.bb_crash_tolerance,
        "rr_crash_tolerance": env_cfg.rr_crash_tolerance,
        "br_crash_tolerance": env_cfg.br_crash_tolerance,
        # Boundary settings
        "boundary_size": env_cfg.boundary_size,
        "min_altitude": env_cfg.min_altitude,
        "max_altitude": env_cfg.max_altitude,
        # Spawn settings (from YAML config)
        "spawn": config.get("environment", {}).get("spawn", {}),
        # Pursuit gains
        "pursuit_gains": {
            "N_pronav_fb": env_cfg.N_pronav_fb,
            "N_pronav_ff": env_cfg.N_pronav_ff,
            "velocity_closure_threshold": env_cfg.velocity_closure_threshold,
            "pp_k_pxy": env_cfg.pp_k_pxy,
            "pp_k_vxy": env_cfg.pp_k_vxy,
            "pp_k_pz": env_cfg.pp_k_pz,
            "pp_k_vz": env_cfg.pp_k_vz,
            # Augmented ProNav parameters
            "N_gain": env_cfg.N_gain,
            "V_min": env_cfg.V_min,
            "K_v": env_cfg.K_v,
        },
        # Target assignment
        "random_target_assignment": env_cfg.random_target_assignment,
        # Observation space description
        "observation_space": {
            "per_agent_obs_dim": raw_env.obs_dim,
            "shared_state_dim": raw_env.shared_observation_space.shape[0],
            "components": [
                "own_state: pos(3) + vel(3) + rpy(3) + rpy_rates(3) = 12",
                "own_one_hot: n_blue",
                "all_blue_states: n_blue * (pos(3) + vel(3) + alive(1)) = n_blue * 7",
                "all_red_states: n_red * (pos(3) + vel(3) + alive(1)) = n_red * 7",
                "red_target_assignments: n_red * n_blue (one-hot per red)",
            ],
        },
        # Curriculum levels (if curriculum learning was enabled)
        "curriculum_levels": None,
    }

    # Add curriculum levels if enabled
    if curriculum_cfg is not None:
        environment_config["curriculum_levels"] = [
            {
                "name": level.name,
                "params": level.params,
                "spawn": level.spawn,
            }
            for level in curriculum_cfg.levels
        ]
    env_config_path = experiment_dir / "environment_config.json"
    with open(env_config_path, "w") as f:
        json.dump(environment_config, f, indent=2)
    print(f"Environment config saved to: {env_config_path}")

    # Save learning config (training hyperparameters and rewards)
    learning_config = {
        "timesteps": timesteps,
        "n_worlds": n_worlds,
        # Rewards
        "rewards": {
            "capture": env_cfg.reward_capture,
            "escape": env_cfg.reward_escape,
            "red_crash": env_cfg.reward_red_crash,
            "blue_crash": env_cfg.reward_blue_crash,
            "boundary": env_cfg.reward_boundary,
            "alive": env_cfg.reward_alive,
            "pursuer_proximity": env_cfg.reward_pursuer_proximity,
            "pursuer_proximity_decay": env_cfg.reward_pursuer_proximity_decay,
        },
        # PPO hyperparameters
        "hyperparameters": {
            "rollouts": training_cfg["rollouts"],
            "learning_epochs": training_cfg["learning_epochs"],
            "mini_batches": training_cfg["mini_batches"],
            "learning_rate": training_cfg["learning_rate"],
            "learning_rate_scheduler": training_cfg["learning_rate_scheduler"],
            "learning_rate_scheduler_kwargs": training_cfg["learning_rate_scheduler_kwargs"],
            "gamma": training_cfg["gamma"],
            "gae_lambda": training_cfg["gae_lambda"],
            "grad_norm_clip": training_cfg["grad_norm_clip"],
            "entropy_loss_scale": training_cfg["entropy_loss_scale"],
            "value_loss_scale": training_cfg["value_loss_scale"],
            "ratio_clip": training_cfg["ratio_clip"],
            "value_clip": training_cfg["value_clip"],
            "kl_threshold": training_cfg["kl_threshold"],
        },
        "initial_log_std": policy_cfg["initial_log_std"],
        "episode_length_s": env_cfg.episode_length_s,
        "start_time": start_time_str,
    }
    learning_config_path = experiment_dir / "learning_config.json"
    with open(learning_config_path, "w") as f:
        json.dump(learning_config, f, indent=2)
    print(f"Learning config saved to: {learning_config_path}")

    print(f"Training configuration:")
    print(f"  - experiment: {args.experiment}")
    print(f"  - n_pairs: {env_cfg.n_pairs}")
    print(f"  - n_worlds: {n_worlds}")
    print(f"  - pursuer_strategy: {env_cfg.pursuer_strategy}")
    print(f"  - device: {device_str}")
    print(f"  - timesteps: {timesteps}")
    print(f"  - policy_net_sizes: {policy_cfg['policy_net_sizes']}")
    print(f"  - value_net_sizes: {policy_cfg['value_net_sizes']}")

    # Wrap with action rescaling (policy outputs [-1, 1], env expects physical bounds)
    env = RescaleActionWrapper(env)

    # Wrap for SKRL (multi-agent wrapper)
    env = wrap_env(env, wrapper="pettingzoo")

    # Wrap with termination logging and curriculum (agent will be set after creation)
    env = TerminationLoggingWrapper(env, raw_env, log_interval=5000, curriculum_manager=curriculum_manager)

    print(f"Environment created with {env_cfg.n_drones} drones ({env_cfg.n_blue} blue, {env_cfg.n_red} red)")
    print(f"Possible agents: {env.possible_agents}")

    # Create memories for rollout buffer (one per agent for SKRL 1.4+)
    memories = {}
    for agent_name in env.possible_agents:
        memories[agent_name] = RandomMemory(
            memory_size=training_cfg["rollouts"],
            num_envs=n_worlds,
            device=device,
        )

    # Create shared policy and critic
    # Get observation and action spaces for a single agent (PettingZoo wrapper uses method)
    sample_agent = env.possible_agents[0]
    sample_obs_space = env.observation_space(sample_agent)
    sample_action_space = env.action_space(sample_agent)

    print(f"Creating shared FFN policy with obs_dim={sample_obs_space.shape[0]}, action_dim={sample_action_space.shape[0]}")
    print(f"Policy net sizes: {policy_cfg['policy_net_sizes']}")

    shared_policy = FFNSharedGaussianPolicy(
        observation_space=sample_obs_space,
        action_space=sample_action_space,
        device=device,
        hidden_sizes=tuple(policy_cfg["policy_net_sizes"]),
        initial_log_std=policy_cfg["initial_log_std"],
        activation=policy_cfg["policy_activation"],
    )

    shared_critic = SharedCritic(
        observation_space=raw_env.shared_observation_space,
        action_space=sample_action_space,
        device=device,
        hidden_dim=policy_cfg["value_net_sizes"][0],
        activation=policy_cfg["value_activation"],
    )

    # Create models dictionary for all agents (parameter sharing)
    models = {}
    for agent_name in env.possible_agents:
        models[agent_name] = {
            "policy": shared_policy,
            "value": shared_critic,
        }

    # MAPPO configuration
    mappo_cfg = MAPPO_DEFAULT_CONFIG.copy()
    mappo_cfg.update({
        "rollouts": training_cfg["rollouts"],
        "learning_epochs": training_cfg["learning_epochs"],
        "mini_batches": training_cfg["mini_batches"],
        "discount_factor": training_cfg["gamma"],
        "lambda": training_cfg["gae_lambda"],
        "learning_rate": training_cfg["learning_rate"],
        "grad_norm_clip": training_cfg["grad_norm_clip"],
        "entropy_loss_scale": training_cfg["entropy_loss_scale"],
        "value_loss_scale": training_cfg["value_loss_scale"],
        "ratio_clip": training_cfg["ratio_clip"],
        "value_clip": training_cfg["value_clip"],
        "kl_threshold": training_cfg["kl_threshold"],
        "experiment": {
            "directory": str(results_dir),
            "experiment_name": run_name,
            "write_interval": 100,
            "checkpoint_interval": 1000,
        },
    })

    # Add learning rate scheduler if configured
    lr_scheduler = training_cfg["learning_rate_scheduler"]
    if lr_scheduler == "KLAdaptiveLR":
        mappo_cfg["learning_rate_scheduler"] = KLAdaptiveLR
        mappo_cfg["learning_rate_scheduler_kwargs"] = training_cfg["learning_rate_scheduler_kwargs"]
        print(f"  - learning_rate_scheduler: KLAdaptiveLR")
        print(f"  - learning_rate_scheduler_kwargs: {training_cfg['learning_rate_scheduler_kwargs']}")
    elif lr_scheduler == "StepLR":
        mappo_cfg["learning_rate_scheduler"] = StepLR
        mappo_cfg["learning_rate_scheduler_kwargs"] = training_cfg["learning_rate_scheduler_kwargs"]
        print(f"  - learning_rate_scheduler: StepLR")
        print(f"  - learning_rate_scheduler_kwargs: {training_cfg['learning_rate_scheduler_kwargs']}")

    # Add value preprocessor if configured
    value_preprocessor = training_cfg["value_preprocessor"]
    if value_preprocessor == "RunningStandardScaler":
        mappo_cfg["value_preprocessor"] = RunningStandardScaler
        mappo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
        print(f"  - value_preprocessor: RunningStandardScaler")

    # Create MAPPO agent
    agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=mappo_cfg,
        observation_spaces={agent_name: env.observation_space(agent_name) for agent_name in env.possible_agents},
        action_spaces={agent_name: env.action_space(agent_name) for agent_name in env.possible_agents},
        shared_observation_spaces={agent_name: raw_env.shared_observation_space for agent_name in env.possible_agents},
        device=device,
    )

    # Connect agent to environment wrapper for termination logging
    env.set_agent(agent)

    print("Agent created successfully")
    print(f"Learnable parameters in policy: {sum(p.numel() for p in shared_policy.parameters() if p.requires_grad)}")
    print(f"Learnable parameters in critic: {sum(p.numel() for p in shared_critic.parameters() if p.requires_grad)}")

    # Create trainer
    trainer_cfg = {
        "timesteps": timesteps,
        "headless": True,
    }

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg=trainer_cfg,
    )

    print("Starting training...")
    trainer.train()

    # Calculate duration and update learning config
    end_time = datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()
    duration_str = str(end_time - start_time).split(".")[0]  # Remove microseconds

    learning_config["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    learning_config["duration"] = duration_str
    learning_config["duration_seconds"] = duration_seconds

    with open(learning_config_path, "w") as f:
        json.dump(learning_config, f, indent=2)

    # Save final checkpoint
    agent.save(str(experiment_dir / "final_checkpoint.pt"))
    print(f"Training complete in {duration_str}. Final checkpoint saved to {experiment_dir / 'final_checkpoint.pt'}")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
