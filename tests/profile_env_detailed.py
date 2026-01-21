#!/usr/bin/env python3
"""Detailed profiling of environment step components."""

import argparse
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import jax
import jax.numpy as jnp
import numpy as np

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig


def profile_env_step(env, n_steps=50):
    """Profile each component of the environment step."""

    # Timing accumulators
    times = {
        "process_blue_actions": [],
        "apply_controls": [],
        "sim_step": [],
        "check_collisions": [],
        "update_alive": [],
        "compute_rewards": [],
        "check_terminated": [],
        "check_truncated": [],
        "reset_done_worlds": [],
        "get_observations": [],
        "total_step": [],
    }

    obs_dict, _ = env.reset()

    # Create dummy actions
    dummy_actions = {
        agent: np.zeros((env.cfg.n_worlds, 4), dtype=np.float32)
        for agent in env.possible_agents
    }

    # Warmup - run a few steps first
    for _ in range(5):
        env._process_blue_actions(dummy_actions)
        env._apply_controls()
        env.sim.step(n_steps=env.cfg.sim_steps_per_control)
        bb_crash, rr_crash, br_crash, out_of_bounds = env._check_collisions()
        env._update_alive_status(bb_crash, rr_crash, br_crash, out_of_bounds)
        _ = env._compute_rewards(bb_crash, rr_crash, br_crash, out_of_bounds)
        _ = env._get_observations()
    jax.block_until_ready(env.sim.data.states.pos)

    for step in range(n_steps):
        t_total_start = time.perf_counter()

        # 1. Process blue actions
        t0 = time.perf_counter()
        env._process_blue_actions(dummy_actions)
        jax.block_until_ready(env.blue_cmd)
        times["process_blue_actions"].append(time.perf_counter() - t0)

        # 2. Apply controls
        t0 = time.perf_counter()
        env._apply_controls()
        jax.block_until_ready(env.sim.data.states.pos)
        times["apply_controls"].append(time.perf_counter() - t0)

        # 3. Sim step
        t0 = time.perf_counter()
        env.sim.step(n_steps=env.cfg.sim_steps_per_control)
        jax.block_until_ready(env.sim.data.states.pos)
        times["sim_step"].append(time.perf_counter() - t0)

        # 4. Check collisions
        t0 = time.perf_counter()
        bb_crash, rr_crash, br_crash, out_of_bounds = env._check_collisions()
        jax.block_until_ready(out_of_bounds)
        times["check_collisions"].append(time.perf_counter() - t0)

        # 5. Update alive status
        t0 = time.perf_counter()
        env._update_alive_status(bb_crash, rr_crash, br_crash, out_of_bounds)
        jax.block_until_ready(env.blue_alive)
        times["update_alive"].append(time.perf_counter() - t0)

        # 6. Compute rewards
        t0 = time.perf_counter()
        rewards = env._compute_rewards(bb_crash, rr_crash, br_crash, out_of_bounds)
        times["compute_rewards"].append(time.perf_counter() - t0)

        # 7. Check terminated/truncated
        env.episode_steps += 1

        t0 = time.perf_counter()
        terminated = env._check_terminated()
        times["check_terminated"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        truncated = env._check_truncated()
        times["check_truncated"].append(time.perf_counter() - t0)

        # 8. Reset done worlds
        t0 = time.perf_counter()
        sample_agent = env.possible_agents[0]
        done_mask = terminated[sample_agent] | truncated[sample_agent]
        if done_mask.any():
            env._reset_done_worlds(done_mask)
        times["reset_done_worlds"].append(time.perf_counter() - t0)

        # 9. Get observations
        t0 = time.perf_counter()
        obs_dict = env._get_observations()
        times["get_observations"].append(time.perf_counter() - t0)

        times["total_step"].append(time.perf_counter() - t_total_start)

    return times


def profile_collision_check_detail(env):
    """Profile collision checking in detail."""
    print("\n--- Detailed Collision Check Profiling ---")

    N = env.cfg.n_worlds
    B = env.cfg.n_blue
    R = env.cfg.n_red

    states = env.sim.data.states
    blue_pos = states.pos[:, :B]
    red_pos = states.pos[:, B:]

    # Time the pure JAX distance computation (what it should be)
    t0 = time.perf_counter()
    for _ in range(100):
        # Vectorized distance computation
        blue_pos_expanded = blue_pos[:, :, None, :]  # (N, B, 1, 3)
        blue_pos_tiled = blue_pos[:, None, :, :]     # (N, 1, B, 3)
        bb_dists = jnp.linalg.norm(blue_pos_expanded - blue_pos_tiled, axis=-1)  # (N, B, B)
    jax.block_until_ready(bb_dists)
    t_vectorized = (time.perf_counter() - t0) / 100

    # Time the current loop-based approach
    t0 = time.perf_counter()
    for _ in range(100):
        bb_crash = jnp.zeros((N, B), dtype=jnp.bool_)
        for i in range(B):
            for j in range(i + 1, B):
                dist = jnp.linalg.norm(blue_pos[:, i] - blue_pos[:, j], axis=-1)
                collision = dist < 0.2
                bb_crash = bb_crash.at[:, i].set(bb_crash[:, i] | collision)
                bb_crash = bb_crash.at[:, j].set(bb_crash[:, j] | collision)
    jax.block_until_ready(bb_crash)
    t_loop = (time.perf_counter() - t0) / 100

    print(f"  Vectorized distance computation: {t_vectorized*1000:.3f} ms")
    print(f"  Loop-based collision check:      {t_loop*1000:.3f} ms")
    print(f"  Speedup potential:               {t_loop/t_vectorized:.1f}x")


def profile_quat_to_rpy(env):
    """Profile quaternion to RPY conversion."""
    print("\n--- Quaternion to RPY Profiling ---")

    states = env.sim.data.states
    quat = states.quat

    # Current scipy-based approach
    t0 = time.perf_counter()
    for _ in range(100):
        rpy = env._quat_to_rpy(quat)
    t_scipy = (time.perf_counter() - t0) / 100

    # Pure JAX approach
    def quat_to_rpy_jax(quat):
        """Convert quaternion (xyzw) to roll-pitch-yaw using JAX."""
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = jnp.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = jnp.where(
            jnp.abs(sinp) >= 1,
            jnp.sign(sinp) * jnp.pi / 2,
            jnp.arcsin(sinp)
        )

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = jnp.arctan2(siny_cosp, cosy_cosp)

        return jnp.stack([roll, pitch, yaw], axis=-1)

    # JIT compile it
    quat_to_rpy_jax_jit = jax.jit(quat_to_rpy_jax)

    # Warmup
    _ = quat_to_rpy_jax_jit(quat)

    t0 = time.perf_counter()
    for _ in range(100):
        rpy = quat_to_rpy_jax_jit(quat)
    jax.block_until_ready(rpy)
    t_jax = (time.perf_counter() - t0) / 100

    print(f"  Scipy-based conversion: {t_scipy*1000:.3f} ms")
    print(f"  Pure JAX conversion:    {t_jax*1000:.3f} ms")
    print(f"  Speedup potential:      {t_scipy/t_jax:.1f}x")


def profile_sim_step_only(env, n_steps=100):
    """Profile just the Crazyflow sim.step()."""
    print("\n--- Pure Sim Step Profiling ---")

    # Reset
    env.reset()

    # Apply some controls
    dummy_cmd = jnp.zeros((env.cfg.n_worlds, env.cfg.n_drones, 4))
    env.sim.attitude_control(dummy_cmd)

    # Warmup
    for _ in range(10):
        env.sim.step(n_steps=env.cfg.sim_steps_per_control)

    # Profile
    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.sim.step(n_steps=env.cfg.sim_steps_per_control)
    t_total = time.perf_counter() - t0

    steps_per_sec = n_steps / t_total
    substeps_per_sec = n_steps * env.cfg.sim_steps_per_control / t_total

    print(f"  n_worlds: {env.cfg.n_worlds}")
    print(f"  n_drones: {env.cfg.n_drones}")
    print(f"  sim_steps_per_control: {env.cfg.sim_steps_per_control}")
    print(f"  Time per env step: {t_total/n_steps*1000:.2f} ms")
    print(f"  Env steps per second: {steps_per_sec:.1f}")
    print(f"  Physics substeps per second: {substeps_per_sec:.1f}")
    print(f"  Total sim steps/sec (worlds * drones * substeps): {substeps_per_sec * env.cfg.n_worlds * env.cfg.n_drones:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=2)
    parser.add_argument("--n-worlds", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=50)
    args = parser.parse_args()

    print(f"Configuration: n_pairs={args.n_pairs}, n_worlds={args.n_worlds}")

    env_cfg = RedVsBlueEnvConfig(
        n_pairs=args.n_pairs,
        n_worlds=args.n_worlds,
        pursuer_strategy="ProNav",
        device="cpu",
    )
    env = RedVsBlueEnv(cfg=env_cfg)

    # Profile pure sim step
    profile_sim_step_only(env, n_steps=100)

    # Profile detailed env step
    print(f"\n--- Detailed Env Step Profiling ({args.n_steps} steps) ---")
    times = profile_env_step(env, n_steps=args.n_steps)

    print("\nComponent breakdown (mean time per step):")
    total = np.mean(times["total_step"]) * 1000
    for key, values in times.items():
        if key != "total_step":
            mean_ms = np.mean(values) * 1000
            pct = mean_ms / total * 100
            print(f"  {key:<25}: {mean_ms:8.2f} ms ({pct:5.1f}%)")
    print(f"  {'TOTAL':<25}: {total:8.2f} ms")

    # Profile specific bottlenecks
    profile_collision_check_detail(env)
    profile_quat_to_rpy(env)

    env.close()


if __name__ == "__main__":
    main()
