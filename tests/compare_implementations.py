#!/usr/bin/env python3
"""Compare performance between Crazyflow env and custom NumPy implementation."""

import argparse
import os
import sys
import time

# Force CPU for fair comparison
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import numpy as np

# Add custom implementation to path
sys.path.insert(0, "/home/llanesc/multiagent_pursuit_evasion/crazyflie-mape-rl-custom")


def profile_crazyflow_env(n_pairs, n_worlds, n_steps):
    """Profile the Crazyflow-based environment."""
    import jax
    from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig

    print(f"\n{'='*60}")
    print("CRAZYFLOW ENVIRONMENT (JAX + First Principles Physics)")
    print(f"{'='*60}")

    env_cfg = RedVsBlueEnvConfig(
        n_pairs=n_pairs,
        n_worlds=n_worlds,
        pursuer_strategy="ProNav",
        device="cpu",
    )
    env = RedVsBlueEnv(cfg=env_cfg)

    # Warmup
    obs_dict, _ = env.reset()
    dummy_actions = {
        agent: np.zeros((n_worlds, 4), dtype=np.float32)
        for agent in env.possible_agents
    }
    for _ in range(10):
        obs_dict, _, _, _, _ = env.step(dummy_actions)
    jax.block_until_ready(env.sim.data.states.pos)

    # Timing
    step_times = []
    reset_times = []

    t0 = time.perf_counter()
    obs_dict, _ = env.reset()
    jax.block_until_ready(env.sim.data.states.pos)
    reset_times.append(time.perf_counter() - t0)

    for step in range(n_steps):
        t0 = time.perf_counter()
        obs_dict, rewards, terminated, truncated, info = env.step(dummy_actions)
        jax.block_until_ready(env.sim.data.states.pos)
        step_times.append(time.perf_counter() - t0)

    env.close()

    mean_step_time = np.mean(step_times) * 1000
    std_step_time = np.std(step_times) * 1000
    mean_reset_time = np.mean(reset_times) * 1000

    print(f"  n_worlds: {n_worlds}")
    print(f"  n_pairs: {n_pairs}")
    print(f"  n_steps: {n_steps}")
    print(f"")
    print(f"  Mean step time: {mean_step_time:.3f} ms (+/- {std_step_time:.3f} ms)")
    print(f"  Reset time: {mean_reset_time:.3f} ms")
    print(f"  Steps/second: {1000/mean_step_time:.1f}")
    print(f"  World-steps/second: {n_worlds * 1000/mean_step_time:.0f}")

    return {
        "mean_step_ms": mean_step_time,
        "std_step_ms": std_step_time,
        "reset_ms": mean_reset_time,
        "steps_per_sec": 1000/mean_step_time,
        "world_steps_per_sec": n_worlds * 1000/mean_step_time,
    }


def profile_custom_env(n_pairs, n_worlds, n_steps):
    """Profile the custom NumPy-based environment."""
    from crazyflie_mape_rl.envs.Crazyflie.RedVsBlue.RedVsBlueGym import RedVsBlueCrazyflieGym

    print(f"\n{'='*60}")
    print("CUSTOM ENVIRONMENT (Pure NumPy + Simplified Dynamics)")
    print(f"{'='*60}")

    env = RedVsBlueCrazyflieGym(n_batch=n_worlds, n_pairs=n_pairs, evader_control="ProNav")

    # Warmup
    obs, state = env.reset()
    dummy_actions = np.zeros((n_worlds, n_pairs, 4), dtype=np.float32)
    for _ in range(10):
        obs, state, reward, done = env.step(dummy_actions)

    # Timing
    step_times = []
    reset_times = []

    t0 = time.perf_counter()
    obs, state = env.reset()
    reset_times.append(time.perf_counter() - t0)

    for step in range(n_steps):
        t0 = time.perf_counter()
        obs, state, reward, done = env.step(dummy_actions)
        step_times.append(time.perf_counter() - t0)

    mean_step_time = np.mean(step_times) * 1000
    std_step_time = np.std(step_times) * 1000
    mean_reset_time = np.mean(reset_times) * 1000

    print(f"  n_worlds: {n_worlds}")
    print(f"  n_pairs: {n_pairs}")
    print(f"  n_steps: {n_steps}")
    print(f"")
    print(f"  Mean step time: {mean_step_time:.3f} ms (+/- {std_step_time:.3f} ms)")
    print(f"  Reset time: {mean_reset_time:.3f} ms")
    print(f"  Steps/second: {1000/mean_step_time:.1f}")
    print(f"  World-steps/second: {n_worlds * 1000/mean_step_time:.0f}")

    return {
        "mean_step_ms": mean_step_time,
        "std_step_ms": std_step_time,
        "reset_ms": mean_reset_time,
        "steps_per_sec": 1000/mean_step_time,
        "world_steps_per_sec": n_worlds * 1000/mean_step_time,
    }


def profile_custom_detailed(n_pairs, n_worlds, n_steps):
    """Detailed profiling of custom environment components."""
    from crazyflie_mape_rl.envs.Crazyflie.RedVsBlue.RedVsBlueGym import RedVsBlueCrazyflieGym

    print(f"\n{'='*60}")
    print("CUSTOM ENVIRONMENT - DETAILED BREAKDOWN")
    print(f"{'='*60}")

    env = RedVsBlueCrazyflieGym(n_batch=n_worlds, n_pairs=n_pairs, evader_control="ProNav")
    env.reset()

    dummy_actions = np.zeros((n_worlds, n_pairs, 4), dtype=np.float32)

    # Warmup
    for _ in range(5):
        env.step(dummy_actions)

    # Time individual components
    times = {
        "blue_dynamics": [],
        "pronav": [],
        "mellinger": [],
        "red_dynamics": [],
        "state_update": [],
        "collision_reward": [],
        "get_observations": [],
        "total": [],
    }

    for _ in range(n_steps):
        t_total_start = time.perf_counter()

        # Blue control and dynamics
        t0 = time.perf_counter()
        blue_action = dummy_actions.copy()
        blue_action *= np.array([0.4, 0.4, 0.1, 0.7*env.g])
        blue_action[:, :, 3] += env.g
        x_dot_blue = env.quadrotor_dynamics(env.states_blue[:, :, :].copy(), blue_action)
        times["blue_dynamics"].append(time.perf_counter() - t0)

        # ProNav
        t0 = time.perf_counter()
        accel_red = env.evader_augmented_pronav(x_dot_blue[env.batch_pair_meshgrid, env.red_target, 3:6])
        times["pronav"].append(time.perf_counter() - t0)

        # Mellinger control
        t0 = time.perf_counter()
        rpy_des, thrust_des = env.mellinger_control(accel_red, env.states_red)
        times["mellinger"].append(time.perf_counter() - t0)

        # Red dynamics
        t0 = time.perf_counter()
        red_thrust_c = np.clip(thrust_des, 0.0, 1.4*env.g)
        red_roll_c = np.clip(rpy_des[...,0], -0.15, 0.15)
        red_pitch_c = np.clip(rpy_des[...,1], -0.15, 0.15)
        red_yaw_c = np.clip(rpy_des[...,2], -0.1, 0.1)
        red_action = np.stack([red_roll_c, red_pitch_c, red_yaw_c, red_thrust_c], axis=-1)
        x_dot_red = env.quadrotor_dynamics(env.states_red[:, :, :].copy(), red_action)
        times["red_dynamics"].append(time.perf_counter() - t0)

        # State update
        t0 = time.perf_counter()
        env.states_blue[:,:,:] += x_dot_blue * env.dt * env.alive_blue[:, :, :]
        env.states_red[:,:,:] += x_dot_red * env.dt * env.alive_red[:, :, :]
        env.t += env.dt
        times["state_update"].append(time.perf_counter() - t0)

        # Collision and reward
        t0 = time.perf_counter()
        reward, done = env.is_done_get_reward(dummy_actions.copy())
        times["collision_reward"].append(time.perf_counter() - t0)

        # Observations
        t0 = time.perf_counter()
        obs = env.get_observations()
        state = env.get_state()
        times["get_observations"].append(time.perf_counter() - t0)

        times["total"].append(time.perf_counter() - t_total_start)

    print("\nComponent breakdown (mean time per step):")
    total = np.mean(times["total"]) * 1000
    for key, values in times.items():
        if key != "total":
            mean_ms = np.mean(values) * 1000
            pct = mean_ms / total * 100
            print(f"  {key:<20}: {mean_ms:8.3f} ms ({pct:5.1f}%)")
    print(f"  {'TOTAL':<20}: {total:8.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="Compare environment implementations")
    parser.add_argument("--n-pairs", type=int, default=2)
    parser.add_argument("--n-worlds", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--detailed", action="store_true", help="Show detailed breakdown")
    args = parser.parse_args()

    print(f"Configuration: n_pairs={args.n_pairs}, n_worlds={args.n_worlds}, n_steps={args.n_steps}")

    # Profile both environments
    crazyflow_results = profile_crazyflow_env(args.n_pairs, args.n_worlds, args.n_steps)
    custom_results = profile_custom_env(args.n_pairs, args.n_worlds, args.n_steps)

    if args.detailed:
        profile_custom_detailed(args.n_pairs, args.n_worlds, args.n_steps)

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"")
    print(f"{'Metric':<25} {'Crazyflow':<15} {'Custom':<15} {'Ratio':<10}")
    print("-" * 65)

    step_ratio = crazyflow_results["mean_step_ms"] / custom_results["mean_step_ms"]
    print(f"{'Step time (ms)':<25} {crazyflow_results['mean_step_ms']:<15.3f} {custom_results['mean_step_ms']:<15.3f} {step_ratio:<10.2f}x")

    reset_ratio = crazyflow_results["reset_ms"] / custom_results["reset_ms"]
    print(f"{'Reset time (ms)':<25} {crazyflow_results['reset_ms']:<15.3f} {custom_results['reset_ms']:<15.3f} {reset_ratio:<10.2f}x")

    throughput_ratio = custom_results["world_steps_per_sec"] / crazyflow_results["world_steps_per_sec"]
    print(f"{'World-steps/sec':<25} {crazyflow_results['world_steps_per_sec']:<15.0f} {custom_results['world_steps_per_sec']:<15.0f} {throughput_ratio:<10.2f}x")

    print(f"\n{'='*60}")
    print("KEY DIFFERENCES:")
    print(f"{'='*60}")
    print("""
1. PHYSICS MODEL:
   - Crazyflow: Full first-principles physics (500Hz internal)
   - Custom: Simplified 1st-order attitude dynamics (single-step)

2. FRAMEWORK:
   - Crazyflow: JAX with JIT compilation
   - Custom: Pure NumPy (no JIT overhead, no JAX dispatch)

3. COLLISION DETECTION:
   - Crazyflow: JIT-compiled vectorized operations
   - Custom: NumPy advanced indexing with precomputed meshgrids

4. QUATERNION TO RPY:
   - Crazyflow: JIT-compiled pure JAX
   - Custom: scipy Rotation.as_euler() (slower)
""")


if __name__ == "__main__":
    main()
