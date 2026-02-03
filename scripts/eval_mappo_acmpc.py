#!/usr/bin/env python3
"""Evaluation script for trained MAPPO models on Red vs Blue environment.

This script evaluates trained blue evader policies against scripted red pursuers.
"""

import argparse
import os

# Parse device argument early to set JAX platform before imports
def _get_device_from_args():
    """Parse just the device argument before other imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, default="cpu")
    args, _ = parser.parse_known_args()
    return args.device

_device = _get_device_from_args()
if _device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    # Suppress JAX CUDA plugin discovery errors when running on CPU
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
# For cuda, let JAX auto-detect

import json
import time
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
import torch

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig, RescaleActionWrapper
from crazyflie_mape_crazyflow.envs.spawn import create_spawn_fn_from_config
from crazyflie_mape_crazyflow.policies import (
    LeapCSharedGaussianPolicy,
    SharedCritic,
)


RESULTS_DIR = Path("results/acmpc")


def parse_step_string(step_str: str) -> int:
    """Parse step string like '500k', '1m', '1.5m', '2m' into integer.

    Args:
        step_str: Step string with optional k/m suffix (case-insensitive).

    Returns:
        Integer step value.

    Raises:
        ValueError: If string cannot be parsed.
    """
    step_str = step_str.strip().lower()

    if step_str.endswith('m'):
        # Million suffix
        value = float(step_str[:-1])
        return int(value * 1_000_000)
    elif step_str.endswith('k'):
        # Thousand suffix
        value = float(step_str[:-1])
        return int(value * 1_000)
    else:
        # Plain integer
        return int(step_str)


def find_checkpoints(search_dir: Path) -> list[Path]:
    """Find all checkpoint files in a directory.

    Searches for best_agent_*.pt, final_checkpoint.pt, and agent_*.pt files.
    Priority: best_agent_*.pt > final_checkpoint.pt > periodic checkpoints (by step number).

    Args:
        search_dir: Directory to search in.

    Returns:
        List of checkpoint paths, with best_agent first if it exists.
    """
    checkpoints = []

    def get_step(p: Path) -> int:
        """Extract step number from filename."""
        try:
            # Handle both best_agent_12345.pt and agent_12345.pt
            return int(p.stem.split("_")[-1])
        except (IndexError, ValueError):
            return 0

    # Priority 1: best_agent_*.pt (always preferred)
    best_agents = list(search_dir.glob("**/best_agent_*.pt"))
    if best_agents:
        # Sort by step number (highest first)
        best_agents.sort(key=get_step, reverse=True)
        checkpoints.extend(best_agents)

    # Priority 2: final_checkpoint.pt
    final_checkpoints = list(search_dir.glob("**/final_checkpoint.pt"))
    if final_checkpoints:
        final_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        checkpoints.extend(final_checkpoints)

    # Priority 3: periodic checkpoints (sorted by step number, highest first)
    periodic = list(search_dir.glob("**/checkpoints/agent_*.pt"))
    periodic.sort(key=get_step, reverse=True)
    checkpoints.extend(periodic)

    return checkpoints


def resolve_checkpoint(experiment: str, checkpoint_arg: str | None, step: str | None = None) -> Path:
    """Resolve checkpoint path from argument.

    Args:
        experiment: Experiment name (e.g., "box_random_spawn").
        checkpoint_arg: Either None (find latest), run folder name (e.g., "run_12345678"),
            or full checkpoint file path.
        step: Optional step string (e.g., "500k", "1m", "1.5m", "2m") to load a specific
            periodic checkpoint. Only used when checkpoint_arg is a run folder name.

    Returns:
        Path to the checkpoint file.

    Raises:
        FileNotFoundError: If checkpoint cannot be found.
    """
    results_dir = RESULTS_DIR / experiment / "results"

    if checkpoint_arg is None:
        # Find latest checkpoint in results directory
        if not results_dir.exists():
            raise FileNotFoundError(
                f"Results directory {results_dir} not found. "
                "Please run training first or specify --checkpoint."
            )
        checkpoints = find_checkpoints(results_dir)
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint found in {results_dir}/. "
                "Please run training first or specify --checkpoint."
            )
        return checkpoints[0]

    checkpoint_path = Path(checkpoint_arg)

    # If it's already a valid file path, use it directly
    if checkpoint_path.exists() and checkpoint_path.is_file():
        if step is not None:
            print(f"Warning: --step ignored when checkpoint is a file path")
        return checkpoint_path

    # Try as run folder name in results directory (e.g., "run_12345678")
    run_dir = results_dir / checkpoint_arg
    if run_dir.exists() and run_dir.is_dir():
        # If step is specified, look for that specific checkpoint
        if step is not None:
            step_num = parse_step_string(step)
            checkpoint_file = run_dir / "checkpoints" / f"agent_{step_num}.pt"
            if checkpoint_file.exists():
                return checkpoint_file
            # Also try without checkpoints subdirectory
            checkpoint_file_alt = run_dir / f"agent_{step_num}.pt"
            if checkpoint_file_alt.exists():
                return checkpoint_file_alt
            raise FileNotFoundError(
                f"Checkpoint for step {step} ({step_num}) not found in {run_dir}. "
                f"Tried: {checkpoint_file} and {checkpoint_file_alt}"
            )

        # No step specified, use priority search
        checkpoints = find_checkpoints(run_dir)
        if checkpoints:
            return checkpoints[0]
        raise FileNotFoundError(
            f"No checkpoint found in {run_dir}. "
            "Expected best_agent_*.pt, final_checkpoint.pt, or checkpoints/agent_*.pt"
        )

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_arg}\n"
        f"Tried: {checkpoint_path} and {run_dir}"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MAPPO ACMPC on Red vs Blue environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/eval_mappo_acmpc.py --experiment box_random_spawn
    python scripts/eval_mappo_acmpc.py --experiment box_random_spawn --checkpoint run_12345678
    python scripts/eval_mappo_acmpc.py --experiment box_random_spawn --checkpoint run_12345678 --step 1m
        """,
    )

    # Experiment (required)
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name (e.g., 'box_random_spawn')")

    # Checkpoint (optional - will find latest if not specified)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Run folder name (e.g., 'run_12345678') or full path. Omit for latest.")

    # Step selection (optional - load specific checkpoint step)
    parser.add_argument("--step", type=str, default=None,
                        help="Checkpoint step to load (e.g., '500k', '1m', '1.5m', '2m'). "
                             "Requires --checkpoint to specify run folder.")

    # Environment settings (only n_worlds and device can be changed for eval)
    parser.add_argument("--n-worlds", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")

    # Curriculum level selection
    parser.add_argument("--level", type=int, default=None,
                        help="Curriculum level to evaluate on (0, 1, 2, ...). Uses spawn and params from that level.")

    # Evaluation settings
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--render-fps", type=int, default=30, help="Render FPS")
    parser.add_argument("--record", action="store_true",
                        help="Record video. Saves to run directory as {experiment}_{run}_level_{N}.mp4. Requires --n-worlds 1.")

    # Camera settings
    parser.add_argument("--cam-distance", type=float, default=8.0,
                        help="Camera distance from scene center")
    parser.add_argument("--cam-azimuth", type=float, default=90.0,
                        help="Camera azimuth angle in degrees (0=front, 90=side)")
    parser.add_argument("--cam-elevation", type=float, default=-25.0,
                        help="Camera elevation angle in degrees (negative=above)")
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                        help="Camera lookat point (x y z)")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")

    return parser.parse_args()


def load_configs(checkpoint_path: Path) -> tuple[dict, dict | None]:
    """Load configuration from experiment directory.

    Supports both new format (environment_config.json + learning_config.json)
    and old format (training_config.json) for backwards compatibility.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Tuple of (environment_config, learning_config).
        learning_config may be None if using old format.

    Raises:
        FileNotFoundError: If no config files are found.
    """
    # The checkpoint is typically in the experiment directory or a subdirectory
    # Try to find config files by walking up the directory tree
    env_config = None
    learning_config = None
    search_dir = checkpoint_path.parent

    for _ in range(3):  # Search up to 3 levels
        # Try new format first
        env_config_path = search_dir / "environment_config.json"
        learning_config_path = search_dir / "learning_config.json"

        if env_config_path.exists():
            with open(env_config_path, "r") as f:
                env_config = json.load(f)
            print(f"Loaded environment config from: {env_config_path}")

            if learning_config_path.exists():
                with open(learning_config_path, "r") as f:
                    learning_config = json.load(f)
                print(f"Loaded learning config from: {learning_config_path}")
            return env_config, learning_config

        # Fall back to old format
        old_config_path = search_dir / "training_config.json"
        if old_config_path.exists():
            with open(old_config_path, "r") as f:
                env_config = json.load(f)
            print(f"Loaded training config (legacy format) from: {old_config_path}")
            return env_config, None

        search_dir = search_dir.parent

    raise FileNotFoundError(
        f"Could not find environment_config.json or training_config.json in or above {checkpoint_path.parent}. "
        "Make sure the checkpoint is in an experiment directory created by train_mappo.py."
    )


def evaluate(env, policy, n_episodes, deterministic=False, render=False, render_fps=30, record_path=None,
             cam_distance=8.0, cam_azimuth=90.0, cam_elevation=-25.0, cam_lookat=(0.0, 0.0, 1.0)):
    """Run evaluation episodes and collect metrics.

    Args:
        env: The environment to evaluate on.
        policy: The policy to evaluate.
        n_episodes: Number of episodes to run.
        deterministic: Whether to use deterministic actions.
        render: Whether to render episodes.
        render_fps: Frames per second for rendering.
        record_path: Path to save video recording (None for no recording).
        cam_distance: Camera distance from lookat point.
        cam_azimuth: Camera azimuth angle in degrees.
        cam_elevation: Camera elevation angle in degrees.
        cam_lookat: Camera lookat point (x, y, z).

    Returns:
        Dictionary of evaluation metrics.
    """
    # Get the raw environment (unwrap if needed)
    raw_env = env.env if hasattr(env, 'env') else env

    n_worlds = env.cfg.n_worlds
    n_blue = env.cfg.n_blue

    # Metrics accumulators
    all_episode_lengths = []
    all_rewards = []
    all_termination_reasons = []  # Track why each episode ended

    # Win rate tracking for overlay
    total_episodes_completed = 0
    blue_wins = 0
    red_wins = 0
    out_of_bounds_count = 0

    # Recording setup
    frames = [] if record_path else None
    recording = record_path is not None

    episodes_completed = 0
    render_interval = max(1, env.cfg.control_freq // render_fps) if (render or recording) else None
    # Time per render frame for real-time sync (in seconds)
    # Use simulation time per frame for accurate real-time playback
    sim_dt = 1.0 / env.cfg.control_freq  # Time per env step in simulation
    render_dt = render_interval * sim_dt if render else None  # Sim time per rendered frame

    # Camera settings to apply on first render
    camera_initialized = False

    print(f"Running evaluation for {n_episodes} episodes...")
    if render:
        print(f"  Render interval: every {render_interval} steps ({render_dt:.4f}s sim time per frame)")
    if recording:
        print(f"  Recording to: {record_path}")
    if render or recording:
        print(f"  Camera: distance={cam_distance}, azimuth={cam_azimuth}, elevation={cam_elevation}, lookat={cam_lookat}")

    while episodes_completed < n_episodes:
        # Reset environment
        obs_dict, info = env.reset()

        # Get initial positions of all drones (world 0 for display/print)
        initial_pos = np.asarray(raw_env.sim.data.states.pos[0])  # (n_drones, 3)
        current_episode = episodes_completed + 1

        # Print initial positions
        print(f"\n  Episode {current_episode}/{n_episodes} - Initial Positions:")
        for i in range(n_blue):
            pos = initial_pos[i]
            print(f"    Blue {i}: ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})")
        for i in range(len(initial_pos) - n_blue):
            pos = initial_pos[n_blue + i]
            print(f"    Red  {i}: ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})")

        episode_rewards = {agent: [] for agent in env.possible_agents}
        step = 0
        last_render_time = time.perf_counter() if render else None

        # Track cumulative termination events per world
        episode_bb_collision = np.zeros(n_worlds)
        episode_rb_collision = np.zeros(n_worlds)
        episode_out_of_bounds = np.zeros(n_worlds)

        done = False
        while not done:
            # Get actions from policy
            actions = {}
            with torch.no_grad():
                for agent_name in env.possible_agents:
                    obs = obs_dict[agent_name]
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=policy.device)

                    # Get action from policy
                    action, log_std, _ = policy.compute({"states": obs_tensor}, role="")
                    if not deterministic:
                        # Sample from distribution using returned log_std
                        std = torch.exp(log_std)
                        action = action + torch.randn_like(action) * std

                    actions[agent_name] = action.cpu().numpy()

            # Step environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

            # Accumulate rewards
            for agent_name, reward in rewards.items():
                episode_rewards[agent_name].append(reward)

            # Accumulate termination events (rates are per-step, we want totals)
            if "termination/bb_collision" in info:
                episode_bb_collision += info["termination/bb_collision"] * n_worlds
                episode_rb_collision += info["termination/rb_collision"] * n_worlds
                episode_out_of_bounds += info["termination/out_of_bounds"] * n_worlds

            # Check for done (any world terminated or truncated)
            sample_agent = env.possible_agents[0]
            done = terminated[sample_agent].any() or truncated[sample_agent].any()
            step += 1

            # Render/record if requested
            if render_interval and (step % render_interval) == 0:
                # Initialize camera on first render (need to render once to create viewer)
                if not camera_initialized:
                    env.render()  # Initialize viewer
                    # Access camera through MujocoRenderer -> internal viewer -> cam
                    if raw_env.sim.viewer is not None and raw_env.sim.viewer.viewer is not None:
                        cam = raw_env.sim.viewer.viewer.cam
                        cam.distance = cam_distance
                        cam.azimuth = cam_azimuth
                        cam.elevation = cam_elevation
                        cam.lookat[:] = cam_lookat
                    camera_initialized = True

                # Add overlays (only for live rendering, not recording - overlays accumulate in rgb_array mode)
                if render and not recording and raw_env.sim.viewer is not None and raw_env.sim.viewer.viewer is not None:
                    viewer = raw_env.sim.viewer.viewer

                    # Get velocities from sim state (world 0)
                    vel = np.asarray(raw_env.sim.data.states.vel[0])  # (n_drones, 3)
                    speeds = np.linalg.norm(vel, axis=-1)  # (n_drones,)

                    # Get alive status (world 0)
                    blue_alive = np.asarray(raw_env.blue_alive[0])  # (n_blue,)
                    red_alive = np.asarray(raw_env.red_alive[0])  # (n_red,)

                    # Win rates overlay
                    if total_episodes_completed > 0:
                        blue_win_rate = blue_wins / total_episodes_completed * 100
                        red_win_rate = red_wins / total_episodes_completed * 100
                        oob_rate = out_of_bounds_count / total_episodes_completed * 100
                    else:
                        blue_win_rate = red_win_rate = oob_rate = 0.0
                    viewer.add_overlay(
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Win Rates",
                        f"Blue: {blue_win_rate:.1f}%  Red: {red_win_rate:.1f}%  OOB: {oob_rate:.1f}%"
                    )
                    viewer.add_overlay(
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Episode",
                        f"{current_episode}/{n_episodes}"
                    )

                    # Initial positions overlay
                    blue_init_pos = [f"B{i}:({initial_pos[i][0]:+.2f},{initial_pos[i][1]:+.2f},{initial_pos[i][2]:+.2f})" for i in range(n_blue)]
                    viewer.add_overlay(
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Init Blue",
                        " ".join(blue_init_pos)
                    )
                    n_red = len(initial_pos) - n_blue
                    red_init_pos = [f"R{i}:({initial_pos[n_blue+i][0]:+.2f},{initial_pos[n_blue+i][1]:+.2f},{initial_pos[n_blue+i][2]:+.2f})" for i in range(n_red)]
                    viewer.add_overlay(
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Init Red",
                        " ".join(red_init_pos)
                    )

                    # Blue agents status: velocity + alive/dead
                    blue_status = []
                    for i in range(n_blue):
                        status = "●" if blue_alive[i] else "✗"
                        blue_status.append(f"B{i}:{status} {speeds[i]:.2f}")
                    viewer.add_overlay(
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Blue [m/s]",
                        "  ".join(blue_status)
                    )

                    # Red agents status: velocity + alive/dead
                    red_status = []
                    for i in range(len(red_alive)):
                        status = "●" if red_alive[i] else "✗"
                        red_status.append(f"R{i}:{status} {speeds[n_blue + i]:.2f}")
                    viewer.add_overlay(
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Red [m/s]",
                        "  ".join(red_status)
                    )

                    # Red target assignments
                    red_targets = np.asarray(raw_env.red_target[0])  # (n_red,)
                    target_status = []
                    for i in range(len(red_alive)):
                        if red_alive[i]:
                            target_status.append(f"R{i}→B{red_targets[i]}")
                        else:
                            target_status.append(f"R{i}→✗")
                    viewer.add_overlay(
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Targets",
                        "  ".join(target_status)
                    )

                if recording:
                    # Capture frame for recording
                    frame = env.render()
                    if frame is not None:
                        # Add text overlay to frame
                        frame = frame.copy()  # Make writable copy
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        color = (255, 255, 255)  # White text
                        bg_color = (0, 0, 0)  # Black background

                        # Build overlay text
                        lines = [
                            f"Episode: {current_episode}/{n_episodes}",
                            f"Step: {step}",
                        ]
                        # Add initial positions
                        for i in range(n_blue):
                            pos = initial_pos[i]
                            lines.append(f"B{i} init: ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})")
                        n_red_drones = len(initial_pos) - n_blue
                        for i in range(n_red_drones):
                            pos = initial_pos[n_blue + i]
                            lines.append(f"R{i} init: ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})")

                        # Add target assignments
                        red_targets = np.asarray(raw_env.red_target[0])  # (n_red,)
                        red_alive = np.asarray(raw_env.red_alive[0])  # (n_red,)
                        target_strs = []
                        for i in range(n_red_drones):
                            if red_alive[i]:
                                target_strs.append(f"R{i}→B{red_targets[i]}")
                            else:
                                target_strs.append(f"R{i}→✗")
                        lines.append(f"Targets: {' '.join(target_strs)}")

                        # Draw text with background
                        y_offset = 20
                        for line in lines:
                            (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
                            cv2.rectangle(frame, (5, y_offset - text_h - 2), (10 + text_w, y_offset + 4), bg_color, -1)
                            cv2.putText(frame, line, (7, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
                            y_offset += text_h + 8

                        frames.append(frame)
                if render:
                    if not recording:
                        env.render()
                    # Sync to real-time
                    current_time = time.perf_counter()
                    elapsed = current_time - last_render_time
                    sleep_time = render_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    last_render_time = time.perf_counter()

        # Episode finished - collect metrics
        # Determine termination reason for each world
        # Note: info contains the state BEFORE auto-reset
        max_steps = env.cfg.max_episode_steps
        sample_agent = env.possible_agents[0]

        # Get final alive status (before auto-reset)
        final_red_alive = info["red_alive"]  # (n_worlds, n_red)
        final_blue_alive = info["blue_alive"]  # (n_worlds, n_blue)

        for world_idx in range(min(n_worlds, n_episodes - episodes_completed)):
            all_episode_lengths.append(step)

            # Compute episode return
            episode_return = sum(
                sum(episode_rewards[agent][s][world_idx] for agent in env.possible_agents)
                for s in range(len(episode_rewards[env.possible_agents[0]]))
            )
            all_rewards.append(episode_return)

            # Determine termination reason and update win rate tracking
            all_red_dead = not final_red_alive[world_idx].any()
            all_blue_dead = not final_blue_alive[world_idx].any()

            if truncated[sample_agent][world_idx]:
                reason = "survived"  # Reached max steps
            elif all_red_dead and not all_blue_dead:
                reason = "blue_won"  # All reds eliminated, blues survive
                blue_wins += 1
            elif episode_rb_collision[world_idx] > 0:
                reason = "captured"  # Blue-red collision (blue captured)
                red_wins += 1
            elif episode_bb_collision[world_idx] > 0:
                reason = "blue_crashed"  # Blue-blue collision
                red_wins += 1
            elif episode_out_of_bounds[world_idx] > 0:
                reason = "out_of_bounds"  # Boundary violation
                out_of_bounds_count += 1
                red_wins += 1
            else:
                reason = "unknown"

            all_termination_reasons.append(reason)
            total_episodes_completed += 1

        episodes_completed += n_worlds
        print(f"  Completed {min(episodes_completed, n_episodes)}/{n_episodes} episodes")

    # Trim to exact number of episodes
    all_episode_lengths = all_episode_lengths[:n_episodes]
    all_rewards = all_rewards[:n_episodes]
    all_termination_reasons = all_termination_reasons[:n_episodes]

    # Compute termination reason counts
    reason_counts = {
        "survived": sum(1 for r in all_termination_reasons if r == "survived"),
        "blue_won": sum(1 for r in all_termination_reasons if r == "blue_won"),
        "captured": sum(1 for r in all_termination_reasons if r == "captured"),
        "blue_crashed": sum(1 for r in all_termination_reasons if r == "blue_crashed"),
        "out_of_bounds": sum(1 for r in all_termination_reasons if r == "out_of_bounds"),
        "unknown": sum(1 for r in all_termination_reasons if r == "unknown"),
    }

    # Compute statistics
    metrics = {
        "n_episodes": n_episodes,
        "episode_length": {
            "mean": float(np.mean(all_episode_lengths)),
            "std": float(np.std(all_episode_lengths)),
            "min": float(np.min(all_episode_lengths)),
            "max": float(np.max(all_episode_lengths)),
        },
        "episode_return": {
            "mean": float(np.mean(all_rewards)),
            "std": float(np.std(all_rewards)),
            "min": float(np.min(all_rewards)),
            "max": float(np.max(all_rewards)),
        },
        "termination_reasons": {
            "survived": reason_counts["survived"] / n_episodes,
            "blue_won": reason_counts["blue_won"] / n_episodes,
            "captured": reason_counts["captured"] / n_episodes,
            "blue_crashed": reason_counts["blue_crashed"] / n_episodes,
            "out_of_bounds": reason_counts["out_of_bounds"] / n_episodes,
        },
    }

    # Save video if recording
    if recording and frames:
        print(f"\nSaving video with {len(frames)} frames to {record_path}...")
        imageio.mimsave(record_path, frames, fps=render_fps)
        print(f"Video saved to: {record_path}")

    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Resolve checkpoint path
    checkpoint_path = resolve_checkpoint(args.experiment, args.checkpoint, args.step)
    print(f"Experiment: {args.experiment}")
    print(f"Using checkpoint: {checkpoint_path}")

    # Load configuration (supports both new and old formats)
    env_config, learning_config = load_configs(checkpoint_path)
    n_pairs = env_config["n_pairs"]
    pursuer_strategy = env_config["pursuer_strategy"]
    mpc_horizon = env_config.get("mpc_horizon", 2)
    drone_model = env_config.get("drone_model", "cf2x_T350")
    # Support old (hidden_dim/hidden_sizes) and new (cost_net_sizes) format
    if "cost_net_sizes" in env_config:
        hidden_dim = env_config["cost_net_sizes"][0]
    elif "hidden_sizes" in env_config:
        hidden_dim = env_config["hidden_sizes"][0]
    elif "hidden_dim" in env_config:
        hidden_dim = env_config["hidden_dim"]
    else:
        hidden_dim = 256  # Default

    # Get MPC velocity constraint
    mpc_velocity_max = env_config.get("mpc_velocity_max", None)

    # Get cost network activation
    cost_net_activation = env_config.get("cost_net_activation", "relu")

    # Get control limits and mass from saved config
    roll_pitch_max = env_config.get("roll_pitch_max", 0.5)
    yaw_max = env_config.get("yaw_max", 0.5)
    mass = env_config.get("mass", None)

    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Environment configuration:")
    print(f"  - n_pairs: {n_pairs}")
    print(f"  - pursuer_strategy: {pursuer_strategy}")
    print(f"  - mpc_horizon: {mpc_horizon}")
    print(f"  - mpc_velocity_max: {mpc_velocity_max}")
    print(f"  - cost_net_sizes: [{hidden_dim}, {hidden_dim}]")
    print(f"  - cost_net_activation: {cost_net_activation}")
    print(f"  - drone_model: {drone_model}")
    print(f"  - roll_pitch_max: {roll_pitch_max}")
    print(f"  - yaw_max: {yaw_max}")
    print(f"  - mass: {mass}")
    print(f"Evaluation configuration:")
    print(f"  - n_worlds: {args.n_worlds}")
    print(f"  - n_episodes: {args.n_episodes}")
    print(f"  - deterministic: {args.deterministic}")
    print(f"  - device: {args.device}")

    # Set device
    device = torch.device(args.device)

    # Create environment configuration from saved config
    # Use values from environment_config.json, with defaults for backwards compatibility
    # Note: MPC settings are passed directly to the policy, not to env config
    env_cfg = RedVsBlueEnvConfig(
        n_pairs=n_pairs,
        n_worlds=args.n_worlds,
        pursuer_strategy=pursuer_strategy,
        device=args.device,
        drone_model=drone_model,
        # Frequencies
        control_freq=env_config.get("control_freq", 100),
        sim_freq=env_config.get("sim_freq", 500),
        mellinger_freq=env_config.get("mellinger_freq", 500),
        # Crash tolerances
        bb_collision_tolerance=env_config.get("bb_collision_tolerance", 0.2),
        rr_collision_tolerance=env_config.get("rr_collision_tolerance", 0.2),
        rb_collision_tolerance=env_config.get("rb_collision_tolerance", 0.2),
        # Boundary settings
        boundary_size=env_config.get("boundary_size", 3.0),
        min_altitude=env_config.get("min_altitude", 0.1),
        max_altitude=env_config.get("max_altitude", 3.0),
        # Pursuit gains
        N_pronav_fb=env_config.get("pursuit_gains", {}).get("N_pronav_fb", 5.0),
        N_pronav_ff=env_config.get("pursuit_gains", {}).get("N_pronav_ff", 1.0),
        velocity_closure_threshold=env_config.get("pursuit_gains", {}).get("velocity_closure_threshold", 0.5),
        pp_k_pxy=env_config.get("pursuit_gains", {}).get("pp_k_pxy", 6.1624),
        pp_k_vxy=env_config.get("pursuit_gains", {}).get("pp_k_vxy", 3.39),
        pp_k_pz=env_config.get("pursuit_gains", {}).get("pp_k_pz", 20.0),
        pp_k_vz=env_config.get("pursuit_gains", {}).get("pp_k_vz", 10.0),
        # Augmented ProNav parameters
        N_gain=env_config.get("pursuit_gains", {}).get("N_gain", 3.0),
        V_min=env_config.get("pursuit_gains", {}).get("V_min", 0.5),
        K_v=env_config.get("pursuit_gains", {}).get("K_v", 2.5),
        # Control limits
        roll_pitch_max=roll_pitch_max,
        yaw_max=yaw_max,
        # Physical parameters
        mass=mass,  # None uses drone_model default
        # Target assignment
        random_target_assignment=env_config.get("random_target_assignment", False),
        # Rewards (from learning_config if available, else defaults)
        reward_capture=learning_config.get("rewards", {}).get("capture", -30.0) if learning_config else -30.0,
        reward_escape=learning_config.get("rewards", {}).get("escape", 20.0) if learning_config else 20.0,
        reward_red_crash=learning_config.get("rewards", {}).get("red_crash", 20.0) if learning_config else 20.0,
        reward_blue_crash=learning_config.get("rewards", {}).get("blue_crash", -20.0) if learning_config else -20.0,
        reward_boundary=learning_config.get("rewards", {}).get("boundary", -5.0) if learning_config else -5.0,
        reward_alive=learning_config.get("rewards", {}).get("alive", 0.1) if learning_config else 0.1,
        reward_pursuer_proximity=learning_config.get("rewards", {}).get("pursuer_proximity", 0.5) if learning_config else 0.5,
        reward_pursuer_proximity_decay=learning_config.get("rewards", {}).get("pursuer_proximity_decay", 2.0) if learning_config else 2.0,
        # Episode length
        episode_length_s=learning_config.get("episode_length_s", 20.0) if learning_config else 20.0,
    )

    # Handle curriculum level selection
    curriculum_levels = env_config.get("curriculum_levels")
    if args.level is not None:
        if curriculum_levels is None:
            raise ValueError(
                f"--level {args.level} specified but no curriculum_levels found in environment_config.json. "
                "This checkpoint may not have been trained with curriculum learning."
            )
        if args.level < 0 or args.level >= len(curriculum_levels):
            raise ValueError(
                f"--level {args.level} is out of range. "
                f"Available levels: 0-{len(curriculum_levels)-1}"
            )

        level_config = curriculum_levels[args.level]
        level_name = level_config.get("name", f"Level {args.level}")
        print(f"\nUsing curriculum level {args.level}: {level_name}")

        # Apply level params to env_cfg
        level_params = level_config.get("params", {})
        for param_name, param_value in level_params.items():
            if hasattr(env_cfg, param_name):
                setattr(env_cfg, param_name, param_value)
                print(f"  {param_name}: {param_value}")

        # Use level's spawn config
        spawn_config = level_config.get("spawn", {})
    else:
        # Use default spawn config from training
        spawn_config = env_config.get("spawn", {})

    # Create spawn function from config
    spawn_fn = create_spawn_fn_from_config(spawn_config)

    # Validate recording settings and construct video path
    record_path = None
    if args.record:
        if args.n_worlds != 1:
            raise ValueError("Recording requires --n-worlds 1")

        # Construct video path: {experiment}_{run}_level_{N}.mp4 in run directory
        run_dir = checkpoint_path.parent
        # Handle case where checkpoint is in a subdirectory (e.g., checkpoints/)
        if run_dir.name == "checkpoints":
            run_dir = run_dir.parent
        run_name = run_dir.name  # e.g., "run_20260121021742"

        if args.level is not None:
            video_filename = f"{args.experiment}_{run_name}_level_{args.level}.mp4"
        else:
            video_filename = f"{args.experiment}_{run_name}.mp4"

        record_path = run_dir / video_filename

    # Create environment
    # Use rgb_array mode for recording, human mode for live rendering
    if args.record:
        render_mode = "rgb_array"
    elif args.render:
        render_mode = "human"
    else:
        render_mode = None
    env = RedVsBlueEnv(cfg=env_cfg, render_mode=render_mode, spawn_fn=spawn_fn)

    # Wrap with action rescaling (policy outputs [-1, 1], env expects physical bounds)
    env = RescaleActionWrapper(env)

    print(f"Environment created with {env_cfg.n_drones} drones ({env_cfg.n_blue} blue, {env_cfg.n_red} red)")

    # Get observation and action spaces
    sample_obs_space = env.observation_space[env.possible_agents[0]]
    sample_action_space = env.action_space[env.possible_agents[0]]

    print(f"Observation space: {sample_obs_space}")
    print(f"Action space: {sample_action_space}")

    # Get MPC dt from config
    mpc_dt = env_config.get("mpc_dt", 0.01)

    # Create policy (same architecture as training)
    # Note: roll_pitch_max, yaw_max, and cost_net_activation were loaded earlier
    shared_policy = LeapCSharedGaussianPolicy(
        observation_space=sample_obs_space,
        action_space=sample_action_space,
        device=device,
        mpc_horizon=mpc_horizon,
        mpc_dt=mpc_dt,
        hidden_dim=hidden_dim,
        roll_pitch_max=roll_pitch_max,
        yaw_max=yaw_max,
        drone_model=drone_model,
        velocity_max=mpc_velocity_max,
        activation=cost_net_activation,
        verbose=False,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # SKRL saves models in various formats depending on version
    # Try different loading strategies
    loaded = False

    if "policy" in checkpoint:
        shared_policy.load_state_dict(checkpoint["policy"])
        print("Loaded policy weights from checkpoint['policy']")
        loaded = True
    elif "models" in checkpoint:
        # Multi-agent format with nested structure
        agent_name = env.possible_agents[0]
        if agent_name in checkpoint["models"] and "policy" in checkpoint["models"][agent_name]:
            shared_policy.load_state_dict(checkpoint["models"][agent_name]["policy"])
            print("Loaded policy weights from multi-agent checkpoint")
            loaded = True

    # SKRL 1.x format: {agent_name: {"policy": state_dict, ...}}
    if not loaded:
        agent_name = env.possible_agents[0]
        if agent_name in checkpoint:
            agent_data = checkpoint[agent_name]
            if isinstance(agent_data, dict) and "policy" in agent_data:
                shared_policy.load_state_dict(agent_data["policy"])
                print(f"Loaded policy weights from checkpoint['{agent_name}']['policy']")
                loaded = True
            elif isinstance(agent_data, dict):
                # Maybe the agent_data is directly the state dict
                try:
                    shared_policy.load_state_dict(agent_data)
                    print(f"Loaded policy weights from checkpoint['{agent_name}']")
                    loaded = True
                except:
                    pass

    if not loaded:
        # Try loading directly (might be just the state dict)
        try:
            shared_policy.load_state_dict(checkpoint)
            print("Loaded policy weights directly from checkpoint")
            loaded = True
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Available keys in checkpoint:", list(checkpoint.keys()))
            if env.possible_agents[0] in checkpoint:
                print(f"Keys in checkpoint['{env.possible_agents[0]}']:",
                      list(checkpoint[env.possible_agents[0]].keys()) if isinstance(checkpoint[env.possible_agents[0]], dict) else "not a dict")
            raise

    # Set to evaluation mode and ensure on correct device
    shared_policy.to(device)
    shared_policy.eval()

    print(f"Policy loaded with {sum(p.numel() for p in shared_policy.parameters())} parameters")

    # Run evaluation
    metrics = evaluate(
        env=env,
        policy=shared_policy,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        render=args.render,
        render_fps=args.render_fps,
        record_path=record_path,
        cam_distance=args.cam_distance,
        cam_azimuth=args.cam_azimuth,
        cam_elevation=args.cam_elevation,
        cam_lookat=tuple(args.cam_lookat),
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes evaluated: {metrics['n_episodes']}")
    print(f"\nEpisode Length:")
    print(f"  Mean: {metrics['episode_length']['mean']:.2f}")
    print(f"  Std:  {metrics['episode_length']['std']:.2f}")
    print(f"  Min:  {metrics['episode_length']['min']:.0f}")
    print(f"  Max:  {metrics['episode_length']['max']:.0f}")
    print(f"\nEpisode Return:")
    print(f"  Mean: {metrics['episode_return']['mean']:.2f}")
    print(f"  Std:  {metrics['episode_return']['std']:.2f}")
    print(f"  Min:  {metrics['episode_return']['min']:.2f}")
    print(f"  Max:  {metrics['episode_return']['max']:.2f}")
    print(f"\nTermination Reasons:")
    print(f"  Survived:      {metrics['termination_reasons']['survived']*100:.1f}%")
    print(f"  Blue Won:      {metrics['termination_reasons']['blue_won']*100:.1f}%")
    print(f"  Captured:      {metrics['termination_reasons']['captured']*100:.1f}%")
    print(f"  Blue Crashed:  {metrics['termination_reasons']['blue_crashed']*100:.1f}%")
    print(f"  Out of Bounds: {metrics['termination_reasons']['out_of_bounds']*100:.1f}%")
    print("=" * 60)

    # Save results to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Close environment
    env.close()

    return metrics


if __name__ == "__main__":
    main()
