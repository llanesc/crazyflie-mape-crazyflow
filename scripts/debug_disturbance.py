#!/usr/bin/env python3
"""Debug script to check disturbance state in environment."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import torch

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig
from crazyflie_mape_crazyflow.envs.spawn import create_spawn_fn_from_config


def main():
    # Load config from action_penalty experiment
    run_dir = Path("results/ffn/action_penalty/results/run_20260203031242")
    env_config_path = run_dir / "environment_config.json"

    with open(env_config_path) as f:
        env_config = json.load(f)

    # Level 8 config
    level_config = env_config["curriculum_levels"][8]
    level_params = level_config["params"]
    spawn_config = level_config.get("spawn", env_config.get("spawn", {}))

    print("=" * 60)
    print("Level 8 Config:")
    print(f"  params: {level_params}")
    print(f"  spawn: {spawn_config}")
    print("=" * 60)

    # Create env config WITHOUT level 8 params (like initial training setup)
    print("\nTest 1: Create environment without level 8 disturbance params")
    env_cfg = RedVsBlueEnvConfig(
        n_pairs=env_config["n_pairs"],
        n_worlds=8,  # Small for testing
        control_freq=env_config["control_freq"],
        episode_length_s=20.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_disturbance=False,  # Initial state
    )

    spawn_fn = create_spawn_fn_from_config(spawn_config)
    env = RedVsBlueEnv(cfg=env_cfg, spawn_fn=spawn_fn, render_mode=None)

    print(f"  cfg.enable_disturbance: {env.cfg.enable_disturbance}")
    print(f"  _disturbance_enabled: {env._disturbance_enabled}")
    print(f"  step_pipeline length: {len(env.sim.step_pipeline)}")

    # Now apply level 8 params via update_curriculum_params
    print("\nTest 2: Apply level 8 params via update_curriculum_params")
    env.update_curriculum_params(spawn_fn=spawn_fn, **level_params)

    print(f"  cfg.enable_disturbance: {env.cfg.enable_disturbance}")
    print(f"  _disturbance_enabled: {env._disturbance_enabled}")
    print(f"  step_pipeline length: {len(env.sim.step_pipeline)}")

    # Create another env with level 8 params from the start
    print("\nTest 3: Create environment WITH level 8 disturbance params from start")
    env_cfg2 = RedVsBlueEnvConfig(
        n_pairs=env_config["n_pairs"],
        n_worlds=8,
        control_freq=env_config["control_freq"],
        episode_length_s=20.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_disturbance=True,  # Level 8 state
        disturbance_force_std=level_params["disturbance_force_std"],
        disturbance_torque_std=level_params["disturbance_torque_std"],
    )

    env2 = RedVsBlueEnv(cfg=env_cfg2, spawn_fn=spawn_fn, render_mode=None)

    print(f"  cfg.enable_disturbance: {env2.cfg.enable_disturbance}")
    print(f"  _disturbance_enabled: {env2._disturbance_enabled}")
    print(f"  step_pipeline length: {len(env2.sim.step_pipeline)}")

    # Compare step pipelines
    print("\nTest 4: Compare step pipelines")
    print(f"  Env1 (update_curriculum_params): {[type(f).__name__ for f in env.sim.step_pipeline]}")
    print(f"  Env2 (direct init):              {[type(f).__name__ for f in env2.sim.step_pipeline]}")

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    if env._disturbance_enabled == env2._disturbance_enabled:
        print("  Disturbance state matches after update_curriculum_params!")
    else:
        print("  BUG: Disturbance state DIFFERS!")
        print(f"    update_curriculum_params: {env._disturbance_enabled}")
        print(f"    direct init: {env2._disturbance_enabled}")


if __name__ == "__main__":
    main()
