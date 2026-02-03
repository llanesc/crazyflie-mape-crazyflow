"""Profile env step components at steady state (after JIT warmup).

Runs 300+ warmup steps to fully compile all JAX paths (including resets),
then profiles each component with proper blocking. Separates reset vs
non-reset steps.

Usage:
    SCIPY_ARRAY_API=1 python tests/profile_env_steady_state.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import time
import numpy as np
import jax.numpy as jnp

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig
from crazyflie_mape_crazyflow.envs.spawn import create_spawn_fn_from_config


def main():
    N_WORLDS = 128
    N_PAIRS = 2
    DRONE_MODEL = "cf2x_T350"
    WARMUP_STEPS = 300
    PROFILE_STEPS = 500

    print("=" * 60)
    print("  STEADY-STATE ENV STEP PROFILING")
    print("=" * 60)
    print(f"  n_worlds: {N_WORLDS}, n_pairs: {N_PAIRS}")
    print(f"  warmup: {WARMUP_STEPS} steps, profile: {PROFILE_STEPS} steps")
    print()

    env_cfg = RedVsBlueEnvConfig(
        n_pairs=N_PAIRS,
        n_worlds=N_WORLDS,
        drone_model=DRONE_MODEL,
        sim_freq=500,
        mellinger_freq=500,
        control_freq=100,
        episode_length_s=20.0,
        device="cpu",
        pursuer_strategy="ProNav",
    )
    spawn_config = {
        "blue": {"method": "deterministic", "x": 2.0, "teammate_spacing": 0.5, "initial_height": 1.0},
        "red": {"method": "deterministic", "x": 0.0, "teammate_spacing": 0.5, "initial_height": 1.0},
    }
    spawn_fn = create_spawn_fn_from_config(spawn_config)
    env = RedVsBlueEnv(cfg=env_cfg, spawn_fn=spawn_fn)
    obs, _ = env.reset()
    agents = list(obs.keys())

    print(f"  sim_steps_per_control: {env_cfg.sim_steps_per_control}")
    print(f"  n_drones: {env_cfg.n_drones}")
    print()

    def make_actions():
        return {name: np.random.uniform(-1, 1, (N_WORLDS, 4)).astype(np.float32) for name in agents}

    # --- Full JIT warmup ---
    print(f"Warming up JIT ({WARMUP_STEPS} steps)...")
    for i in range(WARMUP_STEPS):
        env.step(make_actions())
    print("  JIT warm.")

    # --- Profile with component timing ---
    print(f"Profiling {PROFILE_STEPS} steps...")

    keys = [
        'process_actions', 'apply_controls', 'sim_step',
        'check_collisions', 'update_alive', 'term_events',
        'compute_rewards', 'check_term_trunc',
        'auto_reset', 'get_observations', 'get_info'
    ]
    timings = {k: [] for k in keys}
    n_resets_list = []
    total_times = []

    for step_i in range(PROFILE_STEPS):
        actions = make_actions()
        t_total = time.perf_counter()

        # 1. Process blue actions
        t0 = time.perf_counter()
        env._process_blue_actions(actions)
        env.blue_cmd.block_until_ready()
        timings['process_actions'].append(time.perf_counter() - t0)

        # 2. Apply controls (red pursuit + attitude control)
        t0 = time.perf_counter()
        env._apply_controls()
        # attitude_control modifies sim internals, block on a known output
        env.sim.data.states.pos.block_until_ready()
        timings['apply_controls'].append(time.perf_counter() - t0)

        # 3. Physics sim.step
        t0 = time.perf_counter()
        env.sim.step(n_steps=env_cfg.sim_steps_per_control)
        env.sim.data.states.pos.block_until_ready()
        env.sim.data.states.vel.block_until_ready()
        timings['sim_step'].append(time.perf_counter() - t0)

        # 4. Collision checking
        t0 = time.perf_counter()
        bb_collision, rr_collision, rb_collision, out_of_bounds = env._check_collisions()
        bb_collision.block_until_ready()
        timings['check_collisions'].append(time.perf_counter() - t0)

        # 5. Update alive status
        t0 = time.perf_counter()
        env._update_alive_status(bb_collision, rr_collision, rb_collision, out_of_bounds)
        env.blue_alive.block_until_ready()
        env.red_alive.block_until_ready()
        timings['update_alive'].append(time.perf_counter() - t0)

        # 6. Termination events + rewards
        t0 = time.perf_counter()
        n_worlds = env.cfg.n_worlds
        env.last_termination_events = {
            "bb_collision": float(bb_collision.any(axis=1).sum()) / n_worlds,
            "rr_collision": float(rr_collision.any(axis=1).sum()) / n_worlds,
            "rb_collision": float(rb_collision.any(axis=1).sum()) / n_worlds,
            "out_of_bounds": float(out_of_bounds.any(axis=1).sum()) / n_worlds,
        }
        timings['term_events'].append(time.perf_counter() - t0)

        # 7. Compute rewards
        t0 = time.perf_counter()
        rewards = env._compute_rewards(bb_collision, rr_collision, rb_collision, out_of_bounds)
        timings['compute_rewards'].append(time.perf_counter() - t0)

        # 8. Check termination/truncation
        t0 = time.perf_counter()
        env.episode_steps += 1
        terminated = env._check_terminated()
        truncated = env._check_truncated()
        sample_agent = agents[0]
        all_blue_dead = np.asarray(~env.blue_alive.any(axis=1))
        all_red_dead = np.asarray(~env.red_alive.any(axis=1))
        env.last_termination_events["all_blue_dead"] = float(all_blue_dead.sum()) / n_worlds
        env.last_termination_events["all_red_dead"] = float(all_red_dead.sum()) / n_worlds
        env.last_termination_events["max_steps"] = float(truncated[sample_agent].sum()) / n_worlds
        timings['check_term_trunc'].append(time.perf_counter() - t0)

        # Count resets
        done_mask = terminated[sample_agent] | truncated[sample_agent]
        n_resets = int(done_mask.sum())
        n_resets_list.append(n_resets)

        # 9. Auto-reset
        t0 = time.perf_counter()
        pre_reset_blue_alive = np.array(env.blue_alive)
        pre_reset_red_alive = np.array(env.red_alive)
        if done_mask.any():
            env._reset_done_worlds(done_mask)
            env.sim.data.states.pos.block_until_ready()
        timings['auto_reset'].append(time.perf_counter() - t0)

        # 10. Get observations
        t0 = time.perf_counter()
        obs = env._get_observations()
        timings['get_observations'].append(time.perf_counter() - t0)

        # 11. Get info (shared state for critic)
        t0 = time.perf_counter()
        info = env._get_info(
            bb_collision, rr_collision, rb_collision, out_of_bounds, rewards,
            pre_reset_blue_alive, pre_reset_red_alive
        )
        timings['get_info'].append(time.perf_counter() - t0)

        total_times.append(time.perf_counter() - t_total)

    # Convert to ms
    for k in keys:
        timings[k] = np.array(timings[k]) * 1000
    total_times = np.array(total_times) * 1000
    n_resets_arr = np.array(n_resets_list)

    # --- All steps ---
    print("\n" + "=" * 60)
    print("  ALL STEPS (steady state)")
    print("=" * 60)
    component_sum = sum(np.mean(timings[k]) for k in keys)
    print(f"\n  {'Component':<35} {'Mean(ms)':<10} {'Std':<8} {'%':<6}")
    print(f"  {'─'*60}")
    for k in keys:
        mean = np.mean(timings[k])
        std = np.std(timings[k])
        print(f"  {k:<35} {mean:>7.3f}   {std:>6.3f}   {mean/component_sum*100:>5.1f}%")
    print(f"  {'─'*60}")
    print(f"  {'TOTAL (component sum)':<35} {component_sum:>7.3f} ms")
    print(f"  {'TOTAL (end-to-end)':<35} {np.mean(total_times):>7.3f} ms")
    print(f"  Avg resets/step: {np.mean(n_resets_arr):.2f}")

    # --- Steps WITHOUT resets ---
    no_reset_mask = n_resets_arr == 0
    if no_reset_mask.sum() > 10:
        print(f"\n{'=' * 60}")
        print(f"  STEPS WITHOUT RESETS (n={no_reset_mask.sum()})")
        print(f"{'=' * 60}")
        nr_sum = sum(np.mean(timings[k][no_reset_mask]) for k in keys)
        print(f"\n  {'Component':<35} {'Mean(ms)':<10} {'%':<6}")
        print(f"  {'─'*55}")
        for k in keys:
            mean = np.mean(timings[k][no_reset_mask])
            print(f"  {k:<35} {mean:>7.3f}   {mean/nr_sum*100:>5.1f}%")
        print(f"  {'─'*55}")
        print(f"  {'TOTAL':<35} {nr_sum:>7.3f} ms")

    # --- Steps WITH resets ---
    reset_mask = n_resets_arr > 0
    if reset_mask.sum() > 10:
        print(f"\n{'=' * 60}")
        print(f"  STEPS WITH RESETS (n={reset_mask.sum()}, avg resets={np.mean(n_resets_arr[reset_mask]):.1f})")
        print(f"{'=' * 60}")
        r_sum = sum(np.mean(timings[k][reset_mask]) for k in keys)
        print(f"\n  {'Component':<35} {'Mean(ms)':<10} {'%':<6}")
        print(f"  {'─'*55}")
        for k in keys:
            mean = np.mean(timings[k][reset_mask])
            print(f"  {k:<35} {mean:>7.3f}   {mean/r_sum*100:>5.1f}%")
        print(f"  {'─'*55}")
        print(f"  {'TOTAL':<35} {r_sum:>7.3f} ms")

        # Break down auto_reset by number of resets
        print(f"\n  Auto-reset time by reset count:")
        for n in sorted(set(n_resets_arr)):
            if n == 0:
                continue
            mask = n_resets_arr == n
            if mask.sum() >= 3:
                print(f"    {n} resets: {np.mean(timings['auto_reset'][mask]):>7.3f} ms  (n={mask.sum()})")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    nr_total = sum(np.mean(timings[k][no_reset_mask]) for k in keys) if no_reset_mask.sum() > 0 else 0
    r_total = sum(np.mean(timings[k][reset_mask]) for k in keys) if reset_mask.sum() > 0 else 0
    print(f"  Step without resets: {nr_total:.3f} ms")
    print(f"  Step with resets:    {r_total:.3f} ms  (avg {np.mean(n_resets_arr[reset_mask]):.1f} resets)")
    print(f"  Reset frequency:     {reset_mask.mean()*100:.0f}% of steps")
    print(f"  Weighted average:    {component_sum:.3f} ms")
    print(f"  Rollout (256 steps): {256 * component_sum / 1000:.2f} s")

    print()
    env.close()


if __name__ == "__main__":
    main()
