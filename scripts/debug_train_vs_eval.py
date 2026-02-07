#!/usr/bin/env python3
"""Debug script to compare training vs eval environment behavior.

This script runs the same sequence of actions through both training-style
and eval-style environment setups to find where they diverge.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import torch
from skrl.envs.wrappers.torch import wrap_env

from crazyflie_mape_crazyflow.envs.red_vs_blue_env import RedVsBlueEnv
from crazyflie_mape_crazyflow.envs.red_vs_blue_config import RedVsBlueEnvConfig
from crazyflie_mape_crazyflow.envs.wrappers import RescaleActionWrapper
from crazyflie_mape_crazyflow.envs.spawn import create_spawn_fn_from_config


# Default spawn config (Level 8 style)
DEFAULT_SPAWN_CONFIG = {
    "blue": {
        "method": "deterministic",
        "x": 3.0,
        "teammate_spacing": 1.0,
        "initial_height": 1.0,
    },
    "red": {
        "method": "nominal_box",
        "x": 0.0,
        "teammate_spacing": 0.7,
        "initial_height": 1.0,
        "x_half": 0.6,
        "y_half": 0.6,
        "z_half": 0.4,
    },
    "min_spawn_distance": 0.5,
}


def create_env_training_style(cfg, spawn_config=None):
    """Create environment the way training does."""
    spawn_fn = create_spawn_fn_from_config(spawn_config or DEFAULT_SPAWN_CONFIG)
    env = RedVsBlueEnv(cfg=cfg, spawn_fn=spawn_fn, render_mode=None)
    raw_env = env
    env = RescaleActionWrapper(env)
    env = wrap_env(env, wrapper="pettingzoo")
    return env, raw_env


def create_env_eval_style(cfg, spawn_config=None):
    """Create environment the way eval does."""
    spawn_fn = create_spawn_fn_from_config(spawn_config or DEFAULT_SPAWN_CONFIG)
    env = RedVsBlueEnv(cfg=cfg, spawn_fn=spawn_fn, render_mode=None)
    raw_env = env
    env = RescaleActionWrapper(env)
    # Eval doesn't use wrap_env
    return env, raw_env


def run_episode(env, raw_env, actions_fn, max_steps=100, style="unknown"):
    """Run one episode and track outcomes.

    Args:
        env: The wrapped environment to step
        raw_env: The raw RedVsBlueEnv for checking state
        actions_fn: Function that returns actions given observations
        max_steps: Maximum steps to run
        style: "training" or "eval" for logging

    Returns:
        Dict with episode results
    """
    obs, info = env.reset()

    total_blue_wins = 0
    total_red_wins = 0
    total_max_steps = 0
    steps = 0

    for step in range(max_steps):
        # Generate actions (same for both)
        actions = actions_fn(obs)

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        steps += 1

        # Check termination events from raw env
        term_events = raw_env.last_termination_events
        n_worlds = raw_env.cfg.n_worlds

        blue_wins = term_events.get("blue_win", 0) * n_worlds
        red_wins = term_events["red_win"] * n_worlds
        max_steps_events = term_events["max_steps"] * n_worlds

        total_blue_wins += blue_wins
        total_red_wins += red_wins
        total_max_steps += max_steps_events

        # Check if all worlds are done
        sample_agent = raw_env.possible_agents[0]
        all_done = terminated[sample_agent].all() or truncated[sample_agent].all()

    total = total_blue_wins + total_red_wins + total_max_steps
    blue_rate = total_blue_wins / total if total > 0 else 0

    return {
        "style": style,
        "steps": steps,
        "blue_wins": total_blue_wins,
        "red_wins": total_red_wins,
        "max_steps": total_max_steps,
        "total_episodes": total,
        "blue_win_rate": blue_rate,
    }


def main():
    # Load config from a recent run
    run_dir = Path("results/ffn/action_penalty/results/run_20260203031242")
    env_config_path = run_dir / "environment_config.json"

    if not env_config_path.exists():
        print(f"Config not found: {env_config_path}")
        return

    with open(env_config_path) as f:
        env_config = json.load(f)

    # Create env config
    cfg = RedVsBlueEnvConfig(
        n_worlds=64,  # Same as training
        n_pairs=env_config["n_pairs"],
        control_freq=env_config["control_freq"],
        episode_length_s=20.0,
        bb_collision_tolerance=0.2,  # Level 8
        rr_collision_tolerance=0.2,  # Level 8
        rb_collision_tolerance=0.2,  # Level 8
        randomize_mass=True,
        randomize_inertia=True,
        mass_randomization_std=0.002,
        inertia_randomization_std=1e-6,
        enable_disturbance=True,
        disturbance_force_std=0.005,
        disturbance_torque_std=5e-5,
    )

    print("=" * 60)
    print("Comparing Training vs Eval Environment Behavior")
    print("=" * 60)
    print(f"n_worlds: {cfg.n_worlds}")
    print(f"n_blue: {cfg.n_blue}, n_red: {cfg.n_red}")
    print(f"Level 8 params: bb_collision={cfg.bb_collision_tolerance}, "
          f"domain_rand=True, disturbance=True")
    print()

    # Use random actions for this comparison (not using the trained policy)
    # This tests if the environments themselves behave differently
    np.random.seed(42)

    def random_actions_numpy(obs):
        """Generate random actions in [-1, 1] as numpy arrays."""
        sample_agent = list(obs.keys())[0]
        batch_size = obs[sample_agent].shape[0]
        action_dim = 4  # roll, pitch, yaw, thrust

        actions = {}
        for agent in obs.keys():
            if agent.startswith("blue"):
                actions[agent] = np.random.uniform(-1, 1, (batch_size, action_dim)).astype(np.float32)
            else:
                actions[agent] = np.zeros((batch_size, action_dim), dtype=np.float32)
        return actions

    def random_actions_torch(obs):
        """Generate random actions in [-1, 1] as torch tensors (for SKRL wrapper)."""
        sample_agent = list(obs.keys())[0]
        batch_size = obs[sample_agent].shape[0]
        action_dim = 4  # roll, pitch, yaw, thrust

        actions = {}
        for agent in obs.keys():
            if agent.startswith("blue"):
                actions[agent] = torch.rand(batch_size, action_dim) * 2 - 1  # [-1, 1]
            else:
                actions[agent] = torch.zeros(batch_size, action_dim)
        return actions

    # Test 1: Compare raw environment behavior (no SKRL wrapper difference)
    print("Test 1: Raw environment comparison (same wrapping)")
    print("-" * 60)

    # Both use the same setup: just RescaleActionWrapper
    env1, raw1 = create_env_eval_style(cfg)
    env2, raw2 = create_env_eval_style(cfg)

    # Seed both the same
    np.random.seed(42)
    result1 = run_episode(env1, raw1, random_actions_numpy, max_steps=500, style="env1")

    np.random.seed(42)
    result2 = run_episode(env2, raw2, random_actions_numpy, max_steps=500, style="env2")

    print(f"Env1: blue_wins={result1['blue_wins']:.0f}, red_wins={result1['red_wins']:.0f}, "
          f"max_steps={result1['max_steps']:.0f}, rate={result1['blue_win_rate']:.2%}")
    print(f"Env2: blue_wins={result2['blue_wins']:.0f}, red_wins={result2['red_wins']:.0f}, "
          f"max_steps={result2['max_steps']:.0f}, rate={result2['blue_win_rate']:.2%}")
    print()

    # Test 2: Compare with vs without SKRL wrapper
    print("Test 2: SKRL wrapper vs no wrapper")
    print("-" * 60)

    env_train, raw_train = create_env_training_style(cfg)
    env_eval, raw_eval = create_env_eval_style(cfg)

    # Use torch tensors for training (SKRL wrapper expects this)
    torch.manual_seed(42)
    result_train = run_episode(env_train, raw_train, random_actions_torch, max_steps=500, style="training")

    # Use numpy for eval (no SKRL wrapper)
    np.random.seed(42)
    result_eval = run_episode(env_eval, raw_eval, random_actions_numpy, max_steps=500, style="eval")

    print(f"Training style: blue_wins={result_train['blue_wins']:.0f}, red_wins={result_train['red_wins']:.0f}, "
          f"max_steps={result_train['max_steps']:.0f}, rate={result_train['blue_win_rate']:.2%}")
    print(f"Eval style:     blue_wins={result_eval['blue_wins']:.0f}, red_wins={result_eval['red_wins']:.0f}, "
          f"max_steps={result_eval['max_steps']:.0f}, rate={result_eval['blue_win_rate']:.2%}")

    if abs(result_train['blue_win_rate'] - result_eval['blue_win_rate']) > 0.05:
        print("\n*** SIGNIFICANT DIFFERENCE DETECTED ***")
        print("The SKRL PettingZoo wrapper is changing behavior!")
    else:
        print("\nNo significant difference from SKRL wrapper.")

    print()

    # Test 3: Check reset() vs _reset_done_worlds()
    print("Test 3: reset() vs _reset_done_worlds() behavior")
    print("-" * 60)

    env, raw = create_env_eval_style(cfg)

    # Run with only reset() (eval style)
    np.random.seed(42)
    obs, info = env.reset()

    blue_wins_reset = 0
    red_wins_reset = 0
    for ep in range(10):
        for step in range(100):
            actions = random_actions_numpy(obs)
            obs, rewards, terminated, truncated, info = env.step(actions)

            term = raw.last_termination_events
            blue_wins_reset += term.get("blue_win", 0) * cfg.n_worlds
            red_wins_reset += term["red_win"] * cfg.n_worlds

        # Full reset between episodes
        obs, info = env.reset()

    total_reset = blue_wins_reset + red_wins_reset
    rate_reset = blue_wins_reset / total_reset if total_reset > 0 else 0

    # Run with _reset_done_worlds() (training style)
    np.random.seed(42)
    obs, info = env.reset()

    blue_wins_partial = 0
    red_wins_partial = 0
    for ep in range(10):
        for step in range(100):
            actions = random_actions_numpy(obs)
            obs, rewards, terminated, truncated, info = env.step(actions)

            term = raw.last_termination_events
            blue_wins_partial += term.get("blue_win", 0) * cfg.n_worlds
            red_wins_partial += term["red_win"] * cfg.n_worlds

        # Partial reset (training style)
        done_mask = np.ones(cfg.n_worlds, dtype=bool)
        raw._reset_done_worlds(done_mask)
        raw.episode_steps = np.zeros(cfg.n_worlds, dtype=np.int32)
        obs = raw._get_observations()

    total_partial = blue_wins_partial + red_wins_partial
    rate_partial = blue_wins_partial / total_partial if total_partial > 0 else 0

    print(f"Full reset():        blue_wins={blue_wins_reset:.0f}, red_wins={red_wins_reset:.0f}, rate={rate_reset:.2%}")
    print(f"_reset_done_worlds(): blue_wins={blue_wins_partial:.0f}, red_wins={red_wins_partial:.0f}, rate={rate_partial:.2%}")

    if abs(rate_reset - rate_partial) > 0.05:
        print("\n*** SIGNIFICANT DIFFERENCE DETECTED ***")
        print("reset() and _reset_done_worlds() produce different results!")
    else:
        print("\nNo significant difference between reset methods.")

    print()
    print("=" * 60)
    print("Done")


if __name__ == "__main__":
    main()
