#!/usr/bin/env python3
"""Test script to visualize roll, pitch, and yaw orientation.

This script applies different roll, pitch, and yaw commands to verify
the orientation conventions are correct.

Expected behavior (standard aerospace convention):
- Positive roll: right wing down (rotate about x-axis, clockwise from behind)
- Positive pitch: nose up (rotate about y-axis, clockwise from right)
- Positive yaw: nose right (rotate about z-axis, clockwise from above)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import numpy as np
from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig


def run_test():
    """Run the RPY orientation test."""
    # Create environment with rendering
    env_cfg = RedVsBlueEnvConfig(
        n_pairs=1,
        n_worlds=1,
        device='cpu',
        spawn_method="deterministic",
        initial_height=1.0,
    )
    env = RedVsBlueEnv(cfg=env_cfg, render_mode="human")

    # Get hover thrust (normalized to [-1, 1])
    hover_thrust = env_cfg.mass * env_cfg.gravity
    thrust_mean = (env_cfg.thrust_min + env_cfg.thrust_max) / 2.0
    thrust_scale = (env_cfg.thrust_max - env_cfg.thrust_min) / 2.0
    hover_thrust_normalized = (hover_thrust - thrust_mean) / thrust_scale

    print(f"Hover thrust: {hover_thrust:.4f} N")
    print(f"Normalized hover thrust: {hover_thrust_normalized:.4f}")
    print(f"Thrust range: [{env_cfg.thrust_min:.4f}, {env_cfg.thrust_max:.4f}] N")
    print()

    # Test sequence: (name, roll, pitch, yaw, duration)
    # Values are normalized [-1, 1] which map to [-max_angle, max_angle]
    test_sequence = [
        ("Hover (baseline)", 0.0, 0.0, 0.0, 2.0),
        ("Positive ROLL (+) - right wing down", 0.5, 0.0, 0.0, 2.0),
        ("Hover (reset)", 0.0, 0.0, 0.0, 1.0),
        ("Negative ROLL (-) - left wing down", -0.5, 0.0, 0.0, 2.0),
        ("Hover (reset)", 0.0, 0.0, 0.0, 1.0),
        ("Positive PITCH (+) - nose up", 0.0, 0.5, 0.0, 2.0),
        ("Hover (reset)", 0.0, 0.0, 0.0, 1.0),
        ("Negative PITCH (-) - nose down", 0.0, -0.5, 0.0, 2.0),
        ("Hover (reset)", 0.0, 0.0, 0.0, 1.0),
        ("Positive YAW (+) - nose right", 0.0, 0.0, 0.5, 2.0),
        ("Hover (reset)", 0.0, 0.0, 0.0, 1.0),
        ("Negative YAW (-) - nose left", 0.0, 0.0, -0.5, 2.0),
        ("Hover (reset)", 0.0, 0.0, 0.0, 1.0),
    ]

    # Reset environment
    obs, info = env.reset()
    print("Starting RPY orientation test...")
    print("=" * 60)
    print()

    control_dt = 1.0 / env_cfg.control_freq

    for test_name, roll, pitch, yaw, duration in test_sequence:
        print(f"Test: {test_name}")
        print(f"  Command: roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")

        n_steps = int(duration / control_dt)

        for step in range(n_steps):
            # Create action for blue agent (evader)
            # Action format: [roll, pitch, yaw, thrust] normalized to [-1, 1]
            # Note: Red agents are controlled internally by the environment's pursuit strategy
            action_blue = np.array([[roll, pitch, yaw, hover_thrust_normalized]], dtype=np.float32)

            actions = {
                env.possible_agents[0]: action_blue,  # blue_0
            }

            obs, rewards, terminated, truncated, info = env.step(actions)

            # Print state periodically
            if step == n_steps // 2:
                blue_obs = obs[env.possible_agents[0]]
                pos = blue_obs[0, :3]
                vel = blue_obs[0, 3:6]
                rpy = blue_obs[0, 6:9]
                print(f"  Mid-test state:")
                print(f"    Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"    RPY (rad): [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]")
                print(f"    RPY (deg): [{np.degrees(rpy[0]):.1f}, {np.degrees(rpy[1]):.1f}, {np.degrees(rpy[2]):.1f}]")

            # Small delay for visualization
            time.sleep(0.01)

        print()

    print("=" * 60)
    print("Test complete!")
    print()
    print("Verify the following:")
    print("  - Positive roll: drone tilted with RIGHT side down")
    print("  - Negative roll: drone tilted with LEFT side down")
    print("  - Positive pitch: drone nose UP")
    print("  - Negative pitch: drone nose DOWN")
    print("  - Positive yaw: drone rotates clockwise (from above), nose goes RIGHT")
    print("  - Negative yaw: drone rotates counter-clockwise, nose goes LEFT")

    env.close()


if __name__ == "__main__":
    run_test()
