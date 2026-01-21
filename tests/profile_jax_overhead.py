#!/usr/bin/env python3
"""Profile JAX->NumPy conversion overhead."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import time
import jax
import jax.numpy as jnp
import numpy as np

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig


def main():
    n_worlds = 4096
    n_pairs = 2
    n_iters = 100

    env_cfg = RedVsBlueEnvConfig(
        n_pairs=n_pairs,
        n_worlds=n_worlds,
        pursuer_strategy="ProNav",
        device="cpu",
    )
    env = RedVsBlueEnv(cfg=env_cfg)
    env.reset()

    # Get some JAX arrays
    states = env.sim.data.states
    blue_alive = env.blue_alive

    print(f"Profiling JAX->NumPy conversion overhead with {n_worlds} worlds")
    print("=" * 60)

    # 1. Profile np.array() conversion
    t0 = time.perf_counter()
    for _ in range(n_iters):
        pos_np = np.array(states.pos)
        vel_np = np.array(states.vel)
        quat_np = np.array(states.quat)
        ang_vel_np = np.array(states.ang_vel)
    t_array = (time.perf_counter() - t0) / n_iters * 1000
    print(f"np.array() for 4 state arrays:     {t_array:.3f} ms")

    # 2. Profile np.asarray() conversion (zero-copy when possible)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        pos_np = np.asarray(states.pos)
        vel_np = np.asarray(states.vel)
        quat_np = np.asarray(states.quat)
        ang_vel_np = np.asarray(states.ang_vel)
    t_asarray = (time.perf_counter() - t0) / n_iters * 1000
    print(f"np.asarray() for 4 state arrays:   {t_asarray:.3f} ms")

    # 3. Profile jax.device_get() (explicit transfer)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        pos_np = jax.device_get(states.pos)
        vel_np = jax.device_get(states.vel)
        quat_np = jax.device_get(states.quat)
        ang_vel_np = jax.device_get(states.ang_vel)
    t_device_get = (time.perf_counter() - t0) / n_iters * 1000
    print(f"jax.device_get() for 4 arrays:     {t_device_get:.3f} ms")

    # 4. Profile keeping everything in JAX (no conversion)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        pos_jax = states.pos[:, :n_pairs]
        vel_jax = states.vel[:, :n_pairs]
        # Do some JAX operations
        combined = jnp.concatenate([pos_jax, vel_jax], axis=-1)
    jax.block_until_ready(combined)
    t_jax_only = (time.perf_counter() - t0) / n_iters * 1000
    print(f"Pure JAX (no conversion):          {t_jax_only:.3f} ms")

    print()
    print("=" * 60)
    print("DICT CREATION OVERHEAD")
    print("=" * 60)

    agents = [f"blue_{i}" for i in range(n_pairs)]

    # 5. Profile dict comprehension with np.array
    t0 = time.perf_counter()
    for _ in range(n_iters):
        obs_dict = {agent: np.zeros((n_worlds, 50), dtype=np.float32) for agent in agents}
    t_dict = (time.perf_counter() - t0) / n_iters * 1000
    print(f"Dict comprehension (2 agents):     {t_dict:.3f} ms")

    # 6. Profile returning single array instead
    t0 = time.perf_counter()
    for _ in range(n_iters):
        obs_array = np.zeros((n_worlds, n_pairs, 50), dtype=np.float32)
    t_single = (time.perf_counter() - t0) / n_iters * 1000
    print(f"Single array (N, agents, obs):     {t_single:.3f} ms")

    print()
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print(f"""
1. Use np.asarray() instead of np.array() for zero-copy when possible
   Savings: {t_array - t_asarray:.3f} ms

2. If training loop supports JAX arrays, avoid conversion entirely
   Savings: {t_asarray - t_jax_only:.3f} ms

3. Return stacked array instead of dict for observations
   (requires training loop changes)

4. Consider keeping entire step() in JAX if possible
   - Eliminate Python loop over agents
   - Eliminate all JAX->NumPy conversions
   - Use jax.lax.scan for episode rollouts
""")

    env.close()


if __name__ == "__main__":
    main()
