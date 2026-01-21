#!/usr/bin/env python3
"""Benchmark environment on GPU."""

import time
import jax
import jax.numpy as jnp
import numpy as np

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig


def benchmark(n_worlds, n_steps=100):
    """Benchmark at given world count."""
    env_cfg = RedVsBlueEnvConfig(
        n_pairs=2,
        n_worlds=n_worlds,
        pursuer_strategy="ProNav",
        device="cuda",
    )
    env = RedVsBlueEnv(cfg=env_cfg)

    # Warmup
    obs_dict, _ = env.reset()
    dummy_actions = {
        agent: np.zeros((n_worlds, 4), dtype=np.float32)
        for agent in env.possible_agents
    }
    for _ in range(20):
        obs_dict, _, _, _, _ = env.step(dummy_actions)
    jax.block_until_ready(env.sim.data.states.pos)

    # Benchmark
    obs_dict, _ = env.reset()
    jax.block_until_ready(env.sim.data.states.pos)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        obs_dict, _, _, _, _ = env.step(dummy_actions)
    jax.block_until_ready(env.sim.data.states.pos)
    total_time = time.perf_counter() - t0

    env.close()

    step_time_ms = total_time / n_steps * 1000
    world_steps_per_sec = n_worlds / (total_time / n_steps)
    drone_steps_per_sec = world_steps_per_sec * 4

    return step_time_ms, world_steps_per_sec, drone_steps_per_sec


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    print()

    world_counts = [256, 1024, 4096, 8192, 16384]

    print("=" * 70)
    print("GPU BENCHMARK: Crazyflow Red vs Blue Environment")
    print("=" * 70)
    print(f"{'n_worlds':<12} {'Step(ms)':<12} {'World-steps/s':<15} {'Drone-steps/s':<18} {'Million/s':<10}")
    print("-" * 70)

    for n_worlds in world_counts:
        try:
            step_ms, wps, dps = benchmark(n_worlds, n_steps=100)
            print(f"{n_worlds:<12} {step_ms:<12.2f} {wps:<15,.0f} {dps:<18,.0f} {dps/1e6:<10.2f}")
        except Exception as e:
            print(f"{n_worlds:<12} FAILED: {e}")
            break

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
