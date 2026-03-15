#!/usr/bin/env python3
"""Training script for Red vs Blue environment using MAPPO with ACMPC policy.

This script trains blue evaders using MAPPO with LEAP-C MPC policy
against scripted red pursuers.

Usage:
    python scripts/train_mappo_acmpc.py --experiment deterministic_spawn
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
    config_path = project_root / "results" / "acmpc" / args.experiment / "config.yaml"
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
from skrl.multi_agents.torch.mappo import MAPPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from crazyflie_mape_crazyflow.preprocessors import PartialRunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env
from torch.optim.lr_scheduler import LinearLR, StepLR

from crazyflie_mape_crazyflow.agents import MAPPO_MPC
from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig, RescaleActionWrapper
from crazyflie_mape_crazyflow.envs.spawn import create_spawn_fn_from_config
from crazyflie_mape_crazyflow.policies import (
    LeapCSharedGaussianPolicyLinearLS,
    SharedCritic,
)
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
    - Termination events: blue_win, red_win, max_steps
    - Collision events: bb_collision, rr_collision, rb_collision

    Also handles curriculum learning by tracking blue wins and updating
    environment parameters when advancing levels.
    """

    def __init__(
        self,
        env,
        raw_env,
        log_interval: int = 100,
        curriculum_manager: CurriculumManager | None = None,
        experiment_dir: Path | None = None,
        # checkpoint_retention_interval: int = 500000,  # Disabled - using SKRL default checkpointing
        initial_timestep: int = 0,
    ):
        """Initialize the wrapper.

        Args:
            env: Wrapped environment (SKRL PettingZoo wrapper).
            raw_env: Raw RedVsBlueEnv for accessing termination info directly.
            log_interval: Interval (in steps) for logging to TensorBoard.
            curriculum_manager: Optional curriculum manager for progressive training.
            experiment_dir: Directory for experiment outputs.
            initial_timestep: Starting timestep when resuming training.
        """
        # Use object.__setattr__ to avoid triggering our custom __getattr__
        object.__setattr__(self, '_env', env)
        object.__setattr__(self, '_raw_env', raw_env)
        object.__setattr__(self, '_agent', None)  # Set later via set_agent()
        object.__setattr__(self, '_log_interval', log_interval)
        object.__setattr__(self, '_termination_counts', defaultdict(float))
        object.__setattr__(self, '_collision_counts', defaultdict(float))
        object.__setattr__(self, '_step_count', initial_timestep)
        object.__setattr__(self, '_total_episodes', 0)
        object.__setattr__(self, '_curriculum_manager', curriculum_manager)
        object.__setattr__(self, '_n_worlds', raw_env.cfg.n_worlds)
        object.__setattr__(self, '_experiment_dir', experiment_dir)
        # object.__setattr__(self, '_checkpoint_retention_interval', checkpoint_retention_interval)  # Disabled

        # Track cumulative reward components per world (episode returns)
        # Will be initialized on first step when we know the component names
        object.__setattr__(self, '_cumulative_components', None)
        object.__setattr__(self, '_component_names', None)
        # Completed episode returns (accumulated between logging intervals)
        object.__setattr__(self, '_completed_episode_components', defaultdict(list))

        # Register curriculum level change callback
        if curriculum_manager is not None:
            curriculum_manager.on_level_change(self._on_curriculum_level_change)

    def _on_curriculum_level_change(self, level_idx: int, level_config: CurriculumLevel):
        """Handle curriculum level change by updating environment params."""
        print(f"[DEBUG] Curriculum level change callback: level={level_idx}, name={level_config.name}")
        print(f"[DEBUG] Level params: {level_config.params}")

        # Create spawn function if spawn config is provided
        spawn_fn = None
        if level_config.spawn:
            spawn_fn = create_spawn_fn_from_config(level_config.spawn)

        # Pass all params from the level config to the environment
        self._raw_env.update_curriculum_params(
            spawn_fn=spawn_fn,
            **level_config.params,
        )
        print(f"[DEBUG] After update: bb_tol={self._raw_env.cfg.bb_collision_tolerance}, "
              f"rb_tol={self._raw_env.cfg.rb_collision_tolerance}, "
              f"disturbance={self._raw_env.cfg.enable_disturbance}")

    def set_agent(self, agent):
        """Set the SKRL agent for logging."""
        object.__setattr__(self, '_agent', agent)

    def set_initial_timestep(self, timestep: int):
        """Set the initial timestep for correct step tracking when resuming training.

        Args:
            timestep: The timestep to resume from.
        """
        object.__setattr__(self, '_step_count', timestep)

    def __getattr__(self, name):
        """Forward attribute access to wrapped environment."""
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        """Reset environment and tracking."""
        result = self._env.reset(*args, **kwargs)
        # Initialize agent's MPC state cache from reset info
        if self._agent is not None and hasattr(self._agent, '_current_mpc_state'):
            obs, infos = result
            if "mpc_state" in infos:
                for uid in self._agent.possible_agents:
                    if uid in infos["mpc_state"]:
                        self._agent._current_mpc_state[uid] = torch.as_tensor(
                            infos["mpc_state"][uid], dtype=torch.float32, device=self._agent.device
                        )
        return result

    def step(self, actions):
        """Step environment and track termination/collision events."""
        obs, rewards, terminated, truncated, info = self._env.step(actions)

        # Get all termination events directly from the raw environment
        # The raw env stores last_termination_events as RATES (normalized by n_worlds)
        # We convert back to counts by multiplying by n_worlds
        term_events = self._raw_env.last_termination_events
        n_worlds = self._n_worlds

        # Collision/violation events (not termination reasons, just individual agent events)
        # Counts per-agent: if 2 reds collide, that's 2 rr_collisions
        self._collision_counts["collision/bb_collision"] += term_events["bb_collision"] * n_worlds
        self._collision_counts["collision/rr_collision"] += term_events["rr_collision"] * n_worlds
        self._collision_counts["collision/rb_collision"] += term_events["rb_collision"] * n_worlds
        self._collision_counts["collision/out_of_bounds"] += term_events["out_of_bounds"] * n_worlds

        # Termination events (episode-ending reasons - these only fire once per episode)
        self._termination_counts["termination/red_win"] += term_events["red_win"] * n_worlds
        self._termination_counts["termination/blue_win"] += term_events.get("blue_win", 0) * n_worlds
        self._termination_counts["termination/max_steps"] += term_events["max_steps"] * n_worlds

        # Track total episodes (mutually exclusive outcomes to avoid double counting)
        n_blue_win = int(round(term_events.get("blue_win", 0) * n_worlds))
        n_red_win = int(round(term_events["red_win"] * n_worlds))
        n_max_steps = int(round(term_events["max_steps"] * n_worlds))
        n_episodes_ended = n_blue_win + n_red_win + n_max_steps
        if n_episodes_ended > 0:
            object.__setattr__(self, '_total_episodes', self._total_episodes + n_episodes_ended)

        # Track cumulative reward components per world (episode returns)
        if hasattr(self._raw_env, 'last_reward_components'):
            components = self._raw_env.last_reward_components

            # Initialize tracking arrays on first step
            if self._cumulative_components is None:
                names = list(components.keys())
                object.__setattr__(self, '_component_names', names)
                object.__setattr__(self, '_cumulative_components', {
                    name: np.zeros(n_worlds) for name in names
                })

            # Add current step's components to cumulative totals (per world)
            for name in self._component_names:
                self._cumulative_components[name] += components[name]

            # Check which worlds had an episode end this step
            # IMPORTANT: Use episode-level termination, not per-agent termination.
            # Per-agent terminated is True whenever the agent is dead (even mid-episode),
            # which would prematurely flush cumulative rewards.
            sample_agent = self._raw_env.possible_agents[0]
            episode_terminated = info.get("episode_terminated", np.zeros(n_worlds, dtype=bool))
            truncated_arr = truncated.get(sample_agent, np.zeros(n_worlds, dtype=bool))
            done_mask = np.asarray(episode_terminated).flatten() | np.asarray(truncated_arr).flatten()

            # Record completed episode returns and reset cumulative for done worlds
            if done_mask.any():
                for name in self._component_names:
                    # Store completed episode returns
                    completed_returns = self._cumulative_components[name][done_mask]
                    self._completed_episode_components[name].extend(completed_returns.tolist())
                    # Reset cumulative for done worlds
                    self._cumulative_components[name][done_mask] = 0.0

        self._step_count += 1

        # Log to TensorBoard at intervals (only if agent is set)
        if self._agent is not None and self._step_count % self._log_interval == 0:
            self._log_events()

        return obs, rewards, terminated, truncated, info

    def _log_events(self):
        """Log accumulated termination and collision events to TensorBoard via SKRL agent."""
        # Log termination counts and rates (rates sum to 1 as proportions)
        # Total episodes = blue_win + red_win + max_steps (mutually exclusive outcomes)
        # - blue_win: all reds dead AND at least one blue survives
        # - red_win: all blues dead (includes draws where both teams die)
        # - max_steps: neither team eliminated
        total_terminations = (
            self._termination_counts["termination/blue_win"] +
            self._termination_counts["termination/red_win"] +
            self._termination_counts["termination/max_steps"]
        )

        # DEBUG: Print termination counts
        blue_wins = self._termination_counts["termination/blue_win"]
        red_wins = self._termination_counts["termination/red_win"]
        max_steps = self._termination_counts["termination/max_steps"]
        blue_rate = blue_wins / total_terminations if total_terminations > 0 else 0.0
        print(f"[DEBUG] Step {self._step_count}: blue_win={blue_wins:.0f}, red_win={red_wins:.0f}, max_steps={max_steps:.0f}, total={total_terminations:.0f}, blue_rate={blue_rate:.2%}")

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

        # Log individual reward component episode returns (mean of completed episodes)
        total_mean_return = 0.0
        n_components = 0
        for name, returns in self._completed_episode_components.items():
            if len(returns) > 0:
                mean_return = np.mean(returns)
                self._agent.track_data(f"reward/{name}", mean_return)
                total_mean_return += mean_return
                n_components += 1

        # Compute blue win rate and check for curriculum advancement
        if self._curriculum_manager is not None:
            n_blue_wins = self._termination_counts["termination/blue_win"]
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
        object.__setattr__(self, '_completed_episode_components', defaultdict(list))

        # Disabled - using SKRL's default checkpoint behavior
        # # Clean up old checkpoints (keep only those at retention interval multiples)
        # self._cleanup_checkpoints()

    # Disabled - using SKRL's default checkpoint behavior
    # def _cleanup_checkpoints(self):
    #     """Delete checkpoints that are not at retention interval multiples.
    #
    #     Keeps checkpoints at multiples of checkpoint_retention_interval (e.g., 500k, 1M, 1.5M).
    #     Always keeps best_agent.pt and final_checkpoint.pt.
    #     """
    #     if self._experiment_dir is None:
    #         return
    #
    #     checkpoints_dir = self._experiment_dir / "checkpoints"
    #     if not checkpoints_dir.exists():
    #         return
    #
    #     retention_interval = self._checkpoint_retention_interval
    #
    #     # Find all numbered checkpoint files
    #     for checkpoint_file in checkpoints_dir.glob("agent_*.pt"):
    #         # Skip non-numbered files like best_agent.pt
    #         stem = checkpoint_file.stem
    #         if not stem.startswith("agent_"):
    #             continue
    #
    #         try:
    #             step_str = stem.split("_")[1]
    #             if not step_str.isdigit():
    #                 continue
    #             step = int(step_str)
    #         except (IndexError, ValueError):
    #             continue
    #
    #         # Delete if not at a retention interval multiple
    #         if step % retention_interval != 0:
    #             checkpoint_file.unlink()

    def close(self):
        """Close environment."""
        return self._env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MAPPO with ACMPC policy on Red vs Blue environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start fresh training
    python scripts/train_mappo_acmpc.py --experiment deterministic_spawn

    # Resume from a previous run (just specify run name)
    python scripts/train_mappo_acmpc.py --experiment curriculum_training --resume-run run_20260120033721

    # Resume from a run at a specific curriculum level
    python scripts/train_mappo_acmpc.py --experiment curriculum_training --resume-run run_20260120033721 --curriculum-level 5
        """,
    )

    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name (e.g., 'deterministic_spawn')")
    parser.add_argument("--resume-run", type=str, default=None,
                        help="Run name (e.g., 'run_20260120033721') or full path to resume from")
    parser.add_argument("--curriculum-level", type=int, default=None,
                        help="Curriculum level to start at (0-indexed). Required when using --resume-run with curriculum learning")

    # Rendering options (SKRL 2.0 feature)
    parser.add_argument("--render", action="store_true",
                        help="Enable periodic rendering during training (slower but useful for debugging)")
    parser.add_argument("--render-interval", type=int, default=10000,
                        help="Render every N timesteps (default: 10000). Only used if --render is set")

    return parser.parse_args()


def find_latest_checkpoint(run_dir: Path) -> tuple[Path, int]:
    """Find the latest checkpoint in a run directory.

    Prefers best_agent_*.pt over periodic checkpoints.

    Args:
        run_dir: Path to the run directory containing checkpoints/ subdirectory.

    Returns:
        Tuple of (checkpoint_path, step_number) for the latest checkpoint.

    Raises:
        FileNotFoundError: If no checkpoints are found.
    """
    # First check for best_agent_*.pt in run directory
    best_agents = list(run_dir.glob("best_agent_*.pt"))
    if best_agents:
        # Extract step from filename and return the one with highest step
        best_with_steps = []
        for f in best_agents:
            try:
                step = int(f.stem.split("_")[-1])
                best_with_steps.append((f, step))
            except (IndexError, ValueError):
                continue
        if best_with_steps:
            best_with_steps.sort(key=lambda x: x[1], reverse=True)
            return best_with_steps[0]

    # Fall back to periodic checkpoints in checkpoints/ subdirectory
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    checkpoint_files = []
    for f in checkpoints_dir.glob("agent_*.pt"):
        try:
            step = int(f.stem.split("_")[1])
            checkpoint_files.append((f, step))
        except (IndexError, ValueError):
            continue

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {run_dir}")

    # Sort by step number (highest first) and return the latest
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    return checkpoint_files[0]


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
    experiment_path = find_experiment_path(args.experiment, "acmpc")
    config = load_experiment_config(experiment_path)

    print(f"Loading experiment config from: {experiment_path / 'config.yaml'}")

    # Get training and policy configs
    training_cfg = get_training_config(config)
    policy_cfg = get_policy_config(config, "acmpc")

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

    # Set up results directory
    if args.resume_run is not None:
        # Check if it's just a run name or a full path
        resume_run_path = Path(args.resume_run)
        if not resume_run_path.exists():
            # Try treating it as just a run name within the current experiment
            resume_run_path = experiment_path / "results" / args.resume_run
        if not resume_run_path.exists():
            raise FileNotFoundError(
                f"Run directory not found: {args.resume_run}\n"
                f"Tried: {args.resume_run} and {experiment_path / 'results' / args.resume_run}"
            )
        experiment_dir = resume_run_path
        # Extract results_dir and run_name from the path
        results_dir = experiment_dir.parent
        run_name = experiment_dir.name
        print(f"Resuming training in existing run directory: {experiment_dir}")
    else:
        # Create new results directory inside experiment folder
        results_dir = experiment_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        run_id = generate_run_id()
        run_name = f"run_{run_id}"
        experiment_dir = results_dir / run_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

    # Track start time
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Experiment: {args.experiment}")
    print(f"Results directory: {experiment_dir}")

    # Create environment first (needed for obs_dim in config)
    # Enable rendering if --render flag is set (SKRL 2.0 feature)
    render_mode = "human" if args.render else None
    env = RedVsBlueEnv(cfg=env_cfg, spawn_fn=spawn_fn, render_mode=render_mode)
    raw_env = env

    # Save environment config (parameters that define the environment/policy structure)
    environment_config = {
        "policy_type": "acmpc",
        "cost_type": "linear_ls",
        "experiment_name": args.experiment,
        "n_pairs": env_cfg.n_pairs,
        "drone_model": env_cfg.drone_model,
        "pursuer_strategy": env_cfg.pursuer_strategy,
        # MPC settings
        "mpc_horizon": policy_cfg["mpc_horizon"],
        "mpc_dt": policy_cfg["mpc_dt"],
        "mpc_velocity_max": policy_cfg["mpc_velocity_max"],
        # Control limits
        "roll_pitch_max": policy_cfg["roll_pitch_max"],
        "yaw_max": policy_cfg["yaw_max"],
        # Network dimensions
        "cost_net_sizes": policy_cfg["cost_net_sizes"],
        "value_net_sizes": policy_cfg["value_net_sizes"],
        # Network activations
        "cost_net_activation": policy_cfg["cost_net_activation"],
        "value_activation": policy_cfg["value_activation"],
        "pos_offset_max": policy_cfg["pos_offset_max"],
        "binary_embed_dim": policy_cfg.get("binary_embed_dim", 0),
        # MPC dynamics model
        "mpc_model": policy_cfg.get("mpc_model", "so_rpy"),
        # Frequencies
        "control_freq": env_cfg.control_freq,
        "mellinger_freq": env_cfg.mellinger_freq,
        "sim_freq": env_cfg.sim_freq,
        # Crash tolerances
        "bb_collision_tolerance": env_cfg.bb_collision_tolerance,
        "rr_collision_tolerance": env_cfg.rr_collision_tolerance,
        "rb_collision_tolerance": env_cfg.rb_collision_tolerance,
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
        # Physical parameters
        "mass": env_cfg.mass,
        "thrust_min": env_cfg.thrust_min,
        "thrust_max": env_cfg.thrust_max,
        # Domain randomization
        "randomize_mass": env_cfg.randomize_mass,
        "randomize_inertia": env_cfg.randomize_inertia,
        "mass_randomization_std": env_cfg.mass_randomization_std,
        "inertia_randomization_std": env_cfg.inertia_randomization_std,
        # Disturbance forces/torques
        "enable_disturbance": env_cfg.enable_disturbance,
        "disturbance_force_std": env_cfg.disturbance_force_std,
        "disturbance_torque_std": env_cfg.disturbance_torque_std,
        # Target assignment
        "random_target_assignment": env_cfg.random_target_assignment,
        # Observation space description
        "observation_space": {
            "per_agent_obs_dim": raw_env.obs_dim,
            "shared_state_dim": raw_env.shared_observation_space.shape[0],
            "components": [
                "own_state: pos(3) + vel(3) + rotmat_flat(9) + body_rates(3) = 18",
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
            "red_crash": env_cfg.reward_red_crash,
            "blue_crash": env_cfg.reward_blue_crash,
            "boundary": env_cfg.reward_boundary,
            "pursuer_proximity": env_cfg.reward_pursuer_proximity,
            "pursuer_proximity_decay": env_cfg.reward_pursuer_proximity_decay,
            "angle_coef": env_cfg.reward_angle_coef,
            "velocity_coef": env_cfg.reward_velocity_coef,
            "ground_proximity_coef": env_cfg.reward_ground_proximity_coef,
            "ground_proximity_decay": env_cfg.reward_ground_proximity_decay,
            "action_coef": env_cfg.reward_action_coef,
            "action_smoothness_thrust": env_cfg.reward_action_smoothness_thrust,
            "action_smoothness_rpy": env_cfg.reward_action_smoothness_rpy,
            "rr_relative_velocity_coef": env_cfg.reward_rr_relative_velocity_coef,
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
        # Preprocessors
        "preprocessors": {
            "observation_preprocessor": training_cfg["observation_preprocessor"],
            "state_preprocessor": training_cfg["state_preprocessor"],
            "value_preprocessor": training_cfg["value_preprocessor"],
        },
        "initial_log_std": policy_cfg["initial_log_std"],
        "episode_length_s": env_cfg.episode_length_s,
        "start_time": start_time_str,
        # Resume info
        "resumed_from_run": args.resume_run,
        "initial_curriculum_level": args.curriculum_level,
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
    print(f"  - mpc_horizon: {policy_cfg['mpc_horizon']}")
    print(f"  - mpc_velocity_max: {policy_cfg['mpc_velocity_max']}")
    print(f"  - cost_net_sizes: {policy_cfg['cost_net_sizes']}")
    print(f"  - value_net_sizes: {policy_cfg['value_net_sizes']}")

    # Wrap with action rescaling (policy outputs [-1, 1], env expects physical bounds)
    env = RescaleActionWrapper(env)

    # Wrap for SKRL (multi-agent wrapper)
    env = wrap_env(env, wrapper="pettingzoo")

    # Wrap with termination logging and curriculum (agent will be set after creation)
    env = TerminationLoggingWrapper(env, raw_env, log_interval=5000, curriculum_manager=curriculum_manager, experiment_dir=experiment_dir)

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

    # MPC batch size: max of rollout collection and update phases (per agent)
    # Rollout: n_worlds
    # Update: (n_worlds * rollouts) / mini_batches
    rollout_batch = n_worlds
    update_batch = (n_worlds * training_cfg["rollouts"]) // training_cfg["mini_batches"]
    n_batch_max = max(rollout_batch, update_batch)

    print(f"Creating shared policy with obs_dim={sample_obs_space.shape[0]}, action_dim={sample_action_space.shape[0]}")
    print(f"MPC batch max: {n_batch_max} (rollout={rollout_batch}, update={update_batch})")

    binary_embed_dim = policy_cfg.get("binary_embed_dim", 0)

    shared_policy = LeapCSharedGaussianPolicyLinearLS(
        observation_space=sample_obs_space,
        action_space=sample_action_space,
        device=device,
        mpc_horizon=policy_cfg["mpc_horizon"],
        mpc_dt=policy_cfg["mpc_dt"],
        hidden_dim=policy_cfg["cost_net_sizes"][0],
        roll_pitch_max=policy_cfg["roll_pitch_max"],
        yaw_max=policy_cfg["yaw_max"],
        thrust_min=env_cfg.thrust_min,
        thrust_max=env_cfg.thrust_max,
        mass=env_cfg.mass,
        gravity=env_cfg.gravity,
        drone_model=env_cfg.drone_model,
        mpc_model=policy_cfg.get("mpc_model", "so_rpy"),
        n_batch_max=n_batch_max,
        initial_log_std=policy_cfg["initial_log_std"],
        velocity_max=policy_cfg["mpc_velocity_max"],
        activation=policy_cfg["cost_net_activation"],
        pos_offset_max=policy_cfg["pos_offset_max"],
        binary_dims=raw_env.obs_binary_dims if binary_embed_dim > 0 else None,
        binary_embed_dim=binary_embed_dim,
    )

    shared_critic = SharedCritic(
        observation_space=raw_env.shared_observation_space,
        action_space=sample_action_space,
        device=device,
        hidden_dim=policy_cfg["value_net_sizes"][0],
        activation=policy_cfg["value_activation"],
        binary_dims=raw_env.state_binary_dims if binary_embed_dim > 0 else None,
        binary_embed_dim=binary_embed_dim,
    )

    # Create models dictionary for all agents (parameter sharing)
    models = {}
    for agent_name in env.possible_agents:
        models[agent_name] = {
            "policy": shared_policy,
            "value": shared_critic,
        }

    # MAPPO configuration (SKRL 2.0 uses dataclass config)
    # SKRL 2.0 expand() requires agent-specific dicts for kwargs fields
    possible_agents = env.possible_agents

    # Prepare optional config values (as agent-specific dicts for SKRL 2.0)
    lr_scheduler_class = None
    lr_scheduler_kwargs = {agent: {} for agent in possible_agents}  # Agent-specific empty dicts
    lr_scheduler = training_cfg["learning_rate_scheduler"]
    if lr_scheduler == "KLAdaptiveLR":
        lr_scheduler_class = KLAdaptiveLR
        base_kwargs = training_cfg["learning_rate_scheduler_kwargs"]
        lr_scheduler_kwargs = {agent: base_kwargs.copy() for agent in possible_agents}
        print(f"  - learning_rate_scheduler: KLAdaptiveLR")
        print(f"  - learning_rate_scheduler_kwargs: {base_kwargs}")
    elif lr_scheduler == "StepLR":
        lr_scheduler_class = StepLR
        base_kwargs = training_cfg["learning_rate_scheduler_kwargs"]
        lr_scheduler_kwargs = {agent: base_kwargs.copy() for agent in possible_agents}
        print(f"  - learning_rate_scheduler: StepLR")
        print(f"  - learning_rate_scheduler_kwargs: {base_kwargs}")
    elif lr_scheduler == "LinearLR":
        lr_scheduler_class = LinearLR
        base_kwargs = training_cfg["learning_rate_scheduler_kwargs"]
        lr_scheduler_kwargs = {agent: base_kwargs.copy() for agent in possible_agents}
        print(f"  - learning_rate_scheduler: LinearLR")
        print(f"  - learning_rate_scheduler_kwargs: {base_kwargs}")

    # Observation preprocessor
    obs_preprocessor_class = None
    obs_preprocessor_kwargs = {agent: {} for agent in possible_agents}
    obs_preprocessor = training_cfg.get("observation_preprocessor")
    if obs_preprocessor == "RunningStandardScaler":
        obs_preprocessor_class = PartialRunningStandardScaler
        obs_size = raw_env.obs_dim
        obs_skip = raw_env.obs_binary_dims
        base_kwargs = {"size": obs_size, "skip_dims": obs_skip, "device": device}
        obs_preprocessor_kwargs = {agent: base_kwargs.copy() for agent in possible_agents}
        print(f"  - observation_preprocessor: PartialRunningStandardScaler (size={obs_size}, skip={len(obs_skip)} binary dims)")

    # State preprocessor (centralized critic input)
    state_preprocessor_class = None
    state_preprocessor_kwargs = {agent: {} for agent in possible_agents}
    state_preprocessor = training_cfg.get("state_preprocessor")
    if state_preprocessor == "RunningStandardScaler":
        state_preprocessor_class = PartialRunningStandardScaler
        state_size = raw_env.shared_observation_space.shape[0]
        state_skip = raw_env.state_binary_dims
        base_kwargs = {"size": state_size, "skip_dims": state_skip, "device": device}
        state_preprocessor_kwargs = {agent: base_kwargs.copy() for agent in possible_agents}
        print(f"  - state_preprocessor: PartialRunningStandardScaler (size={state_size}, skip={len(state_skip)} binary dims)")

    # Value preprocessor
    value_preprocessor_class = None
    value_preprocessor_kwargs = {agent: {} for agent in possible_agents}
    value_preprocessor = training_cfg.get("value_preprocessor")
    if value_preprocessor == "RunningStandardScaler":
        value_preprocessor_class = RunningStandardScaler
        base_kwargs = {"size": 1, "device": device}
        value_preprocessor_kwargs = {agent: base_kwargs.copy() for agent in possible_agents}
        print(f"  - value_preprocessor: RunningStandardScaler")

    mappo_cfg = MAPPO_CFG(
        rollouts=training_cfg["rollouts"],
        learning_epochs=training_cfg["learning_epochs"],
        mini_batches=training_cfg["mini_batches"],
        discount_factor=training_cfg["gamma"],
        lambda_=training_cfg["gae_lambda"],
        learning_rate=training_cfg["learning_rate"],
        learning_rate_scheduler=lr_scheduler_class,
        learning_rate_scheduler_kwargs=lr_scheduler_kwargs,
        observation_preprocessor=obs_preprocessor_class,
        observation_preprocessor_kwargs=obs_preprocessor_kwargs,
        state_preprocessor=state_preprocessor_class,
        state_preprocessor_kwargs=state_preprocessor_kwargs,
        grad_norm_clip=training_cfg["grad_norm_clip"],
        entropy_loss_scale=training_cfg["entropy_loss_scale"],
        value_loss_scale=training_cfg["value_loss_scale"],
        ratio_clip=training_cfg["ratio_clip"],
        value_clip=training_cfg["value_clip"],
        kl_threshold=training_cfg["kl_threshold"],
        value_preprocessor=value_preprocessor_class,
        value_preprocessor_kwargs=value_preprocessor_kwargs,
        experiment={
            "directory": str(results_dir),
            "experiment_name": run_name,
            "write_interval": 100,
            "checkpoint_interval": 5000,
        },
    )

    # Create MAPPO_MPC agent (extends MAPPO with MPC state passthrough)
    agent = MAPPO_MPC(
        mpc_state_size=12,  # [pos(3), rpy(3), vel(3), drpy(3)]
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=mappo_cfg,
        observation_spaces={agent_name: env.observation_space(agent_name) for agent_name in env.possible_agents},
        action_spaces={agent_name: env.action_space(agent_name) for agent_name in env.possible_agents},
        state_spaces={agent_name: raw_env.shared_observation_space for agent_name in env.possible_agents},  # SKRL 2.0: renamed from shared_observation_spaces
        device=device,
    )

    # Connect agent to environment wrapper for termination logging
    env.set_agent(agent)

    print("Agent created successfully")
    print(f"Learnable parameters in policy: {sum(p.numel() for p in shared_policy.parameters() if p.requires_grad)}")
    print(f"Learnable parameters in critic: {sum(p.numel() for p in shared_critic.parameters() if p.requires_grad)}")

    # Load checkpoint if resuming from a previous run
    initial_timestep = 0
    if args.resume_run is not None:
        checkpoint_path, resume_step = find_latest_checkpoint(experiment_dir)
        print(f"Loading checkpoint: {checkpoint_path} (step {resume_step})")
        agent.load(str(checkpoint_path))
        initial_timestep = resume_step
        # Update wrapper's step count for correct best_agent step tracking
        env.set_initial_timestep(initial_timestep)
        print(f"Checkpoint loaded successfully, resuming from step {initial_timestep}")

    # Set curriculum level if specified (must be after loading checkpoint)
    if args.curriculum_level is not None:
        if curriculum_manager is None:
            print(f"Warning: --curriculum-level specified but curriculum is not enabled in config")
        else:
            curriculum_manager.set_level(args.curriculum_level)
    elif args.resume_run is not None and curriculum_manager is not None:
        # If resuming but no curriculum level specified, warn user
        print(f"[Curriculum] Warning: Resuming from run but no --curriculum-level specified. Starting at level 0.")

    # Create trainer
    # When resuming, add initial_timestep to total so we train for the full additional duration
    total_timesteps = timesteps + initial_timestep
    trainer_cfg = {
        "timesteps": total_timesteps,
        "initial_timestep": initial_timestep,  # Resume from this timestep
        "headless": not args.render,  # Enable rendering if --render flag is set
        "render_interval": args.render_interval,  # SKRL 2.0: render every N timesteps
    }
    if args.render:
        print(f"Rendering enabled: rendering every {args.render_interval} timesteps")

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg=trainer_cfg,
    )

    if initial_timestep > 0:
        print(f"Training from step {initial_timestep} to {total_timesteps} ({timesteps} additional steps)")

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
