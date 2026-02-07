#!/usr/bin/env python3
"""Test script to verify win rates using the EXACT same code path as training.

This script:
1. Sets up the environment exactly like train_mappo_ffn.py
2. Loads a checkpoint
3. Runs the policy for N timesteps
4. Tracks win rates using the same TerminationLoggingWrapper
5. Prints debug output to verify the 60% vs 13% discrepancy
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import torch
import numpy as np
from collections import defaultdict

from skrl.envs.wrappers.torch import wrap_env

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig, RescaleActionWrapper
from crazyflie_mape_crazyflow.envs.spawn import create_spawn_fn_from_config
from crazyflie_mape_crazyflow.policies import FFNSharedGaussianPolicy


def main():
    parser = argparse.ArgumentParser(description="Test win rates using trainer code path")
    parser.add_argument("--experiment", type=str, default="action_penalty", help="Experiment name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to load")
    parser.add_argument("--level", type=int, default=8, help="Curriculum level")
    parser.add_argument("--timesteps", type=int, default=10000, help="Number of timesteps to run")
    parser.add_argument("--n-worlds", type=int, default=128, help="Number of parallel worlds (match training)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--no-domain-rand", action="store_true", help="Disable domain randomization (mass/inertia)")
    parser.add_argument("--no-disturbance", action="store_true", help="Disable external force/torque disturbances")
    args = parser.parse_args()

    # Find experiment
    results_dir = Path("results/ffn") / args.experiment
    if not results_dir.exists():
        print(f"Experiment not found: {results_dir}")
        return

    # Find latest run
    run_dirs = sorted(results_dir.glob("results/run_*"), reverse=True)
    if not run_dirs:
        print(f"No runs found in {results_dir}")
        return
    run_dir = run_dirs[0]
    print(f"Using run: {run_dir}")

    # Load configs
    env_config_path = run_dir / "environment_config.json"
    with open(env_config_path) as f:
        env_config = json.load(f)

    # Get curriculum level config
    curriculum_levels = env_config.get("curriculum_levels", [])
    if args.level >= len(curriculum_levels):
        print(f"Level {args.level} not found, max is {len(curriculum_levels)-1}")
        return
    level_config = curriculum_levels[args.level]
    print(f"Using curriculum level {args.level}: {level_config.get('name', 'Unknown')}")

    # Extract level params
    if "params" in level_config and isinstance(level_config["params"], dict):
        level_params = level_config["params"]
    else:
        level_params = {k: v for k, v in level_config.items() if k not in ("name", "level", "spawn")}

    # Create environment config - MUST use same drone_model and mass as training!
    env_cfg = RedVsBlueEnvConfig(
        n_pairs=env_config["n_pairs"],
        n_worlds=args.n_worlds,
        control_freq=env_config["control_freq"],
        episode_length_s=20.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        drone_model=env_config.get("drone_model", "cf2x_T350"),  # Match training!
        mass=env_config.get("mass"),  # Explicit mass override from training
    )
    print(f"  drone_model: {env_cfg.drone_model}")
    print(f"  mass: {env_cfg.mass}")

    # Apply level params
    for param_name, param_value in level_params.items():
        if hasattr(env_cfg, param_name):
            setattr(env_cfg, param_name, param_value)
            print(f"  {param_name}: {param_value}")

    # Override domain randomization if requested
    if args.no_domain_rand:
        env_cfg.randomize_mass = False
        env_cfg.randomize_inertia = False
        print("\n  ** Domain randomization DISABLED **")

    # Override disturbances if requested
    if args.no_disturbance:
        env_cfg.enable_disturbance = False
        print("  ** Disturbances DISABLED **")

    # Get spawn config from level
    spawn_config = level_config.get("spawn", env_config.get("spawn", {}))
    spawn_fn = create_spawn_fn_from_config(spawn_config)

    # Create environment EXACTLY like training does
    print("\nCreating environment (training style)...")
    raw_env = RedVsBlueEnv(cfg=env_cfg, spawn_fn=spawn_fn, render_mode=None)
    env = RescaleActionWrapper(raw_env)
    env = wrap_env(env, wrapper="pettingzoo")  # SKRL wrapper

    print(f"  n_worlds: {env_cfg.n_worlds}")
    print(f"  n_blue: {env_cfg.n_blue}, n_red: {env_cfg.n_red}")
    print(f"  Collision tolerances: bb={env_cfg.bb_collision_tolerance}, rb={env_cfg.rb_collision_tolerance}")
    print(f"  Domain rand: mass={env_cfg.randomize_mass}, inertia={env_cfg.randomize_inertia}")
    print(f"  Disturbance: {env_cfg.enable_disturbance}")

    # Find and load checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Find latest checkpoint
        checkpoints = list((run_dir / "checkpoints").glob("agent_*.pt"))
        best_checkpoints = list(run_dir.glob("best_agent_*.pt"))
        all_checkpoints = checkpoints + best_checkpoints
        if not all_checkpoints:
            print("No checkpoints found")
            return
        # Sort by step number
        def get_step(p):
            try:
                return int(p.stem.split("_")[-1])
            except:
                return 0
        checkpoint_path = sorted(all_checkpoints, key=get_step, reverse=True)[0]

    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Create policy (use raw env's spaces - Dict keyed by agent names)
    sample_agent = raw_env.possible_agents[0]
    obs_space = raw_env.observation_space[sample_agent]
    act_space = raw_env.action_space[sample_agent]

    policy = FFNSharedGaussianPolicy(
        observation_space=obs_space,
        action_space=act_space,
        device=env_cfg.device,
        state_space=raw_env.shared_observation_space,
        num_layers=len(env_config.get("policy_net_sizes", [256, 256])),
        hidden_units=env_config.get("policy_net_sizes", [256, 256]),
        activation=env_config.get("policy_activation", "relu"),
    )

    # Load checkpoint (handle various SKRL formats)
    checkpoint = torch.load(checkpoint_path, map_location=env_cfg.device)
    agent_name = sample_agent  # e.g., "blue_0"

    loaded = False
    if "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
        print("Loaded from checkpoint['policy']")
        loaded = True
    elif "models" in checkpoint:
        # Multi-agent SKRL format
        if agent_name in checkpoint["models"] and "policy" in checkpoint["models"][agent_name]:
            policy.load_state_dict(checkpoint["models"][agent_name]["policy"])
            print(f"Loaded from checkpoint['models']['{agent_name}']['policy']")
            loaded = True
    elif agent_name in checkpoint:
        # Per-agent format
        agent_data = checkpoint[agent_name]
        if isinstance(agent_data, dict) and "policy" in agent_data:
            policy.load_state_dict(agent_data["policy"])
            print(f"Loaded from checkpoint['{agent_name}']['policy']")
            loaded = True
        elif isinstance(agent_data, dict):
            try:
                policy.load_state_dict(agent_data)
                print(f"Loaded from checkpoint['{agent_name}']")
                loaded = True
            except Exception:
                pass

    if not loaded:
        # Try direct load
        try:
            policy.load_state_dict(checkpoint)
            print("Loaded directly from checkpoint")
            loaded = True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print(f"Available keys: {list(checkpoint.keys())}")
            return

    policy.to(env_cfg.device)
    policy.eval()
    print("Checkpoint loaded successfully")

    # Initialize tracking (same as TerminationLoggingWrapper)
    # Window counts (reset every log_interval like training)
    window_counts = defaultdict(float)
    # Cumulative counts (for final summary)
    total_counts = defaultdict(float)
    total_episodes = 0
    log_interval = 5000  # Match training's log_interval

    # Run evaluation loop EXACTLY like training
    print(f"\nRunning {args.timesteps} timesteps...")
    print("=" * 60)

    # Reset environment (only once at start, like training)
    observations, infos = env.reset()

    for timestep in range(args.timesteps):
        # Get actions from policy (EXACTLY like training - use act() not compute())
        with torch.no_grad():
            actions = {}
            for agent_name in env.possible_agents:
                obs = observations[agent_name]
                if not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(obs, dtype=torch.float32, device=env_cfg.device)
                else:
                    obs = obs.to(env_cfg.device)

                if args.deterministic:
                    # For deterministic eval, use compute() which returns mean actions
                    action, _ = policy.compute({"observations": obs}, role="")
                else:
                    # For stochastic (like training), use act() which samples from distribution
                    action, _ = policy.act({"observations": obs}, role="")

                actions[agent_name] = action

        # Step environment
        observations, rewards, terminated, truncated, infos = env.step(actions)

        # Track termination events (EXACTLY like TerminationLoggingWrapper)
        term_events = raw_env.last_termination_events
        n_worlds = args.n_worlds

        # Window counts (for per-interval rates like TensorBoard)
        window_counts["blue_win"] += term_events.get("blue_win", 0) * n_worlds
        window_counts["red_win"] += term_events["red_win"] * n_worlds
        window_counts["max_steps"] += term_events["max_steps"] * n_worlds

        # Total counts (for final summary)
        total_counts["blue_win"] += term_events.get("blue_win", 0) * n_worlds
        total_counts["red_win"] += term_events["red_win"] * n_worlds
        total_counts["max_steps"] += term_events["max_steps"] * n_worlds

        # Track total episodes
        n_blue_win = int(round(term_events.get("blue_win", 0) * n_worlds))
        n_red_win = int(round(term_events["red_win"] * n_worlds))
        n_max_steps = int(round(term_events["max_steps"] * n_worlds))
        n_episodes_ended = n_blue_win + n_red_win + n_max_steps
        total_episodes += n_episodes_ended

        # Log at intervals (EXACTLY like TerminationLoggingWrapper._log_events)
        if (timestep + 1) % log_interval == 0:
            window_total = (
                window_counts["blue_win"] +
                window_counts["red_win"] +
                window_counts["max_steps"]
            )

            if window_total > 0:
                blue_rate = window_counts["blue_win"] / window_total
                red_rate = window_counts["red_win"] / window_total
                max_steps_rate = window_counts["max_steps"] / window_total

                print(f"Step {timestep+1} [WINDOW]: "
                      f"blue_win={window_counts['blue_win']:.0f} ({blue_rate:.1%}), "
                      f"red_win={window_counts['red_win']:.0f} ({red_rate:.1%}), "
                      f"max_steps={window_counts['max_steps']:.0f} ({max_steps_rate:.1%}), "
                      f"window_eps={window_total:.0f}")

            # Reset window counts (EXACTLY like training does after _log_events)
            window_counts = defaultdict(float)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS (CUMULATIVE):")
    print("=" * 60)

    total_terminations = (
        total_counts["blue_win"] +
        total_counts["red_win"] +
        total_counts["max_steps"]
    )

    if total_terminations > 0:
        blue_rate = total_counts["blue_win"] / total_terminations * 100
        red_rate = total_counts["red_win"] / total_terminations * 100
        max_steps_rate = total_counts["max_steps"] / total_terminations * 100

        print(f"Total episodes: {total_terminations:.0f}")
        print(f"Blue wins: {total_counts['blue_win']:.0f} ({blue_rate:.1f}%)")
        print(f"Red wins: {total_counts['red_win']:.0f} ({red_rate:.1f}%)")
        print(f"Max steps: {total_counts['max_steps']:.0f} ({max_steps_rate:.1f}%)")
    else:
        print("No episodes completed!")

    print("\nThis should match what TensorBoard shows during training.")
    print("If this shows ~60% but eval_mappo_ffn.py shows ~15%, the bug is in eval.")
    print("If this shows ~15%, the 60% in TensorBoard may be from old logged data.")


if __name__ == "__main__":
    main()
