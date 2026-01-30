"""Profile individual components of the environment step.

Breaks down time spent in:
1. Process blue actions (numpy stacking + clipping)
2. Apply controls (red pursuit + attitude control)
3. Physics simulation (JAX sim.step)
4. Collision checking (JAX JIT)
5. Alive status update + target reassignment
6. Reward computation
7. Termination/truncation checks
8. Observation building (quat→rpy, state extraction, concatenation)
9. Auto-reset

Usage:
    SCIPY_ARRAY_API=1 python tests/profile_env_step.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import time
import numpy as np
import jax

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig, RescaleActionWrapper
from crazyflie_mape_crazyflow.envs.spawn import create_spawn_fn_from_config


def main():
    N_WORLDS = 128
    N_PAIRS = 2
    DRONE_MODEL = "cf2x_T350"
    N_STEPS = 200
    WARMUP = 20

    print("=" * 60)
    print("  ENVIRONMENT STEP PROFILING")
    print("=" * 60)
    print(f"  n_worlds: {N_WORLDS}")
    print(f"  n_pairs: {N_PAIRS} ({N_PAIRS} blue + {N_PAIRS} red = {N_PAIRS*2} drones)")
    print(f"  drone_model: {DRONE_MODEL}")
    print()

    # Create environment
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

    print(f"  sim_freq: {env_cfg.sim_freq} Hz")
    print(f"  control_freq: {env_cfg.control_freq} Hz")
    print(f"  sim_steps_per_control: {env_cfg.sim_steps_per_control}")
    print(f"  total drones: {env_cfg.n_drones}")
    print()

    # Reset
    obs, _ = env.reset()

    # Create random actions
    agent_names = list(obs.keys())
    def make_actions():
        return {name: np.random.uniform(-1, 1, (N_WORLDS, 4)).astype(np.float32) for name in agent_names}

    # Warmup (JIT compile all paths)
    print(f"Warming up ({WARMUP} steps)...")
    for _ in range(WARMUP):
        env.step(make_actions())

    # --- Profile full step for baseline ---
    print(f"Profiling full step ({N_STEPS} steps)...")
    full_times = []
    for _ in range(N_STEPS):
        actions = make_actions()
        t0 = time.perf_counter()
        obs, rewards, terminated, truncated, info = env.step(actions)
        full_times.append(time.perf_counter() - t0)
    full_times = np.array(full_times) * 1000

    # --- Monkey-patch to instrument env.step ---
    print(f"Profiling instrumented env.step ({N_STEPS} steps)...")
    import types
    step_timings = {k: [] for k in [
        'process_actions', 'apply_controls', 'sim_step',
        'check_collisions', 'update_alive', 'term_events_dict',
        'compute_rewards', 'episode_step_update', 'check_term',
        'check_trunc', 'term_events_extra', 'pre_reset_copy',
        'auto_reset', 'get_observations', 'get_info'
    ]}

    def instrumented_step(self, actions):
        t = {}
        t0 = time.perf_counter()
        self._process_blue_actions(actions)
        t['process_actions'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._apply_controls()
        t['apply_controls'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.sim.step(n_steps=self.cfg.sim_steps_per_control)
        t['sim_step'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        bb_crash, rr_crash, br_crash, out_of_bounds = self._check_collisions()
        t['check_collisions'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._update_alive_status(bb_crash, rr_crash, br_crash, out_of_bounds)
        t['update_alive'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        n_worlds = self.cfg.n_worlds
        self.last_termination_events = {
            "bb_crash": float(bb_crash.any(axis=1).sum()) / n_worlds,
            "rr_crash": float(rr_crash.any(axis=1).sum()) / n_worlds,
            "br_crash": float(br_crash.any(axis=1).sum()) / n_worlds,
            "out_of_bounds": float(out_of_bounds.any(axis=1).sum()) / n_worlds,
        }
        t['term_events_dict'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        rewards = self._compute_rewards(bb_crash, rr_crash, br_crash, out_of_bounds)
        t['compute_rewards'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.episode_steps += 1
        t['episode_step_update'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        terminated = self._check_terminated()
        t['check_term'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        truncated = self._check_truncated()
        t['check_trunc'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        sample_agent = self.possible_agents[0]
        all_blue_dead = np.asarray(~self.blue_alive.any(axis=1))
        all_red_dead = np.asarray(~self.red_alive.any(axis=1))
        self.last_termination_events["all_blue_dead"] = float(all_blue_dead.sum()) / n_worlds
        self.last_termination_events["all_red_dead"] = float(all_red_dead.sum()) / n_worlds
        self.last_termination_events["max_steps"] = float(truncated[sample_agent].sum()) / n_worlds
        t['term_events_extra'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        pre_reset_blue_alive = np.array(self.blue_alive)
        pre_reset_red_alive = np.array(self.red_alive)
        t['pre_reset_copy'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        done_mask = terminated[sample_agent] | truncated[sample_agent]
        if done_mask.any():
            self._reset_done_worlds(done_mask)
        t['auto_reset'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        obs = self._get_observations()
        t['get_observations'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        info = self._get_info(
            bb_crash, rr_crash, br_crash, out_of_bounds, rewards,
            pre_reset_blue_alive, pre_reset_red_alive
        )
        t['get_info'] = time.perf_counter() - t0

        for k, v in t.items():
            step_timings[k].append(v)

        return obs, rewards, terminated, truncated, info

    # Bind the instrumented step
    env.step = types.MethodType(instrumented_step, env)

    for _ in range(N_STEPS):
        actions = make_actions()
        env.step(actions)

    # Restore original step
    del env.step

    # --- Profile full step with internal blocking ---
    print(f"Profiling full step with blocking ({N_STEPS} steps)...")
    import jax.numpy as jnp
    full_blocked_times = []
    t_sections = {k: [] for k in [
        'process_actions', 'apply_controls', 'sim_step',
        'collisions', 'alive_update', 'term_events',
        'rewards', 'term_trunc', 'pre_reset_copy',
        'auto_reset', 'observations', 'get_info', 'return_prep'
    ]}
    for _ in range(N_STEPS):
        actions_i = make_actions()
        t_total_start = time.perf_counter()

        # 1. Process blue actions
        t0 = time.perf_counter()
        env._process_blue_actions(actions_i)
        t_sections['process_actions'].append(time.perf_counter() - t0)

        # 2. Apply controls
        t0 = time.perf_counter()
        env._apply_controls()
        t_sections['apply_controls'].append(time.perf_counter() - t0)

        # 3. Sim step
        t0 = time.perf_counter()
        env.sim.step(n_steps=env_cfg.sim_steps_per_control)
        t_sections['sim_step'].append(time.perf_counter() - t0)

        # 4. Collisions
        t0 = time.perf_counter()
        bb_crash, rr_crash, br_crash, out_of_bounds = env._check_collisions()
        t_sections['collisions'].append(time.perf_counter() - t0)

        # 5. Alive update
        t0 = time.perf_counter()
        env._update_alive_status(bb_crash, rr_crash, br_crash, out_of_bounds)
        t_sections['alive_update'].append(time.perf_counter() - t0)

        # 6. Termination events
        t0 = time.perf_counter()
        n_worlds = env.cfg.n_worlds
        env.last_termination_events = {
            "bb_crash": float(bb_crash.any(axis=1).sum()) / n_worlds,
            "rr_crash": float(rr_crash.any(axis=1).sum()) / n_worlds,
            "br_crash": float(br_crash.any(axis=1).sum()) / n_worlds,
            "out_of_bounds": float(out_of_bounds.any(axis=1).sum()) / n_worlds,
        }
        t_sections['term_events'].append(time.perf_counter() - t0)

        # 7. Rewards
        t0 = time.perf_counter()
        rewards = env._compute_rewards(bb_crash, rr_crash, br_crash, out_of_bounds)
        t_sections['rewards'].append(time.perf_counter() - t0)

        # 8. Termination/truncation
        t0 = time.perf_counter()
        env.episode_steps += 1
        terminated = env._check_terminated()
        truncated = env._check_truncated()
        all_blue_dead = np.asarray(~env.blue_alive.any(axis=1))
        all_red_dead = np.asarray(~env.red_alive.any(axis=1))
        env.last_termination_events["all_blue_dead"] = float(all_blue_dead.sum()) / n_worlds
        env.last_termination_events["all_red_dead"] = float(all_red_dead.sum()) / n_worlds
        env.last_termination_events["max_steps"] = float(truncated[env.possible_agents[0]].sum()) / n_worlds
        t_sections['term_trunc'].append(time.perf_counter() - t0)

        # 9. Pre-reset copy
        t0 = time.perf_counter()
        pre_reset_blue_alive = np.array(env.blue_alive)
        pre_reset_red_alive = np.array(env.red_alive)
        t_sections['pre_reset_copy'].append(time.perf_counter() - t0)

        # 10. Auto-reset
        t0 = time.perf_counter()
        sample_agent = env.possible_agents[0]
        done_mask = terminated[sample_agent] | truncated[sample_agent]
        if done_mask.any():
            env._reset_done_worlds(done_mask)
        t_sections['auto_reset'].append(time.perf_counter() - t0)

        # 11. Observations
        t0 = time.perf_counter()
        obs = env._get_observations()
        t_sections['observations'].append(time.perf_counter() - t0)

        # 12. Get info
        t0 = time.perf_counter()
        info = env._get_info(
            bb_crash, rr_crash, br_crash, out_of_bounds, rewards,
            pre_reset_blue_alive, pre_reset_red_alive
        )
        t_sections['get_info'].append(time.perf_counter() - t0)

        full_blocked_times.append(time.perf_counter() - t_total_start)

    full_blocked_times = np.array(full_blocked_times) * 1000

    # --- Print instrumented step results ---
    print("\n" + "=" * 60)
    print("  INSTRUMENTED env.step() (actual method, no blocking)")
    print("=" * 60)
    instr_total = sum(np.mean(np.array(v) * 1000) for v in step_timings.values())
    print(f"\n  {'Section':<30} {'Mean (ms)':<12} {'%':<8}")
    print(f"  {'─'*55}")
    for name, times in step_timings.items():
        ms_arr = np.array(times) * 1000
        mean = np.mean(ms_arr)
        print(f"  {name:<30} {mean:>7.3f} ms   ({mean/instr_total*100:>5.1f}%)")
    print(f"  {'─'*55}")
    print(f"  {'Sum':<30} {instr_total:>7.3f} ms")
    print(f"  {'Full step (original)':<30} {np.mean(full_times):>7.3f} ms")

    # --- Profile individual components ---
    print("Profiling individual components (with JAX blocking)...")

    # We'll manually call each component and time it
    import jax.numpy as jnp
    from crazyflie_mape_crazyflow.envs.red_vs_blue_env import (
        _jit_compute_red_control,
        _jit_check_collisions,
        _jit_update_alive_and_reassign,
        _jit_compute_rewards,
        _jit_quat_to_rpy,
        _jit_ang_vel_to_rpy_rates,
    )

    def block_jax(*arrays):
        """Block until JAX arrays are ready."""
        for a in arrays:
            if hasattr(a, 'block_until_ready'):
                a.block_until_ready()
            elif isinstance(a, (tuple, list)):
                block_jax(*a)

    t_process_actions = []
    t_apply_controls = []
    t_sim_step = []
    t_collisions = []
    t_alive_update = []
    t_rewards = []
    t_termination = []
    t_observations = []
    t_term_events = []
    t_pre_reset_copy = []
    t_auto_reset = []
    t_get_info = []

    for step_i in range(N_STEPS):
        actions = make_actions()

        # 1. Process blue actions
        t0 = time.perf_counter()
        env._process_blue_actions(actions)
        env.blue_cmd.block_until_ready()
        t_process_actions.append(time.perf_counter() - t0)

        # 2. Apply controls (red pursuit + attitude control)
        t0 = time.perf_counter()
        env._apply_controls()
        # Block on sim state to ensure attitude_control is done
        env.sim.data.states.pos.block_until_ready()
        t_apply_controls.append(time.perf_counter() - t0)

        # 3. Physics simulation
        t0 = time.perf_counter()
        env.sim.step(n_steps=env_cfg.sim_steps_per_control)
        # Block on output state
        env.sim.data.states.pos.block_until_ready()
        env.sim.data.states.vel.block_until_ready()
        t_sim_step.append(time.perf_counter() - t0)

        # 4. Collision checking
        t0 = time.perf_counter()
        bb_crash, rr_crash, br_crash, out_of_bounds = env._check_collisions()
        bb_crash.block_until_ready()
        t_collisions.append(time.perf_counter() - t0)

        # 5. Alive status update
        t0 = time.perf_counter()
        env._update_alive_status(bb_crash, rr_crash, br_crash, out_of_bounds)
        env.blue_alive.block_until_ready()
        env.red_alive.block_until_ready()
        t_alive_update.append(time.perf_counter() - t0)

        # 6. Reward computation
        t0 = time.perf_counter()
        rewards = env._compute_rewards(bb_crash, rr_crash, br_crash, out_of_bounds)
        t_rewards.append(time.perf_counter() - t0)

        # 7. Termination/truncation
        t0 = time.perf_counter()
        env.episode_steps += 1
        terminated = env._check_terminated()
        truncated = env._check_truncated()
        t_termination.append(time.perf_counter() - t0)

        # 8. Observations
        t0 = time.perf_counter()
        obs = env._get_observations()
        t_observations.append(time.perf_counter() - t0)

        # 9. Termination event tracking (from step method)
        t0 = time.perf_counter()
        n_worlds = env.cfg.n_worlds
        env.last_termination_events = {
            "bb_crash": float(bb_crash.any(axis=1).sum()) / n_worlds,
            "rr_crash": float(rr_crash.any(axis=1).sum()) / n_worlds,
            "br_crash": float(br_crash.any(axis=1).sum()) / n_worlds,
            "out_of_bounds": float(out_of_bounds.any(axis=1).sum()) / n_worlds,
        }
        all_blue_dead = np.asarray(~env.blue_alive.any(axis=1))
        all_red_dead = np.asarray(~env.red_alive.any(axis=1))
        env.last_termination_events["all_blue_dead"] = float(all_blue_dead.sum()) / n_worlds
        env.last_termination_events["all_red_dead"] = float(all_red_dead.sum()) / n_worlds
        t_term_events.append(time.perf_counter() - t0)

        # 10. Pre-reset alive copy
        t0 = time.perf_counter()
        pre_reset_blue_alive = np.array(env.blue_alive)
        pre_reset_red_alive = np.array(env.red_alive)
        t_pre_reset_copy.append(time.perf_counter() - t0)

        # 11. Auto-reset (only if needed)
        t0 = time.perf_counter()
        sample_agent = env.possible_agents[0]
        done_mask = terminated[sample_agent] | truncated[sample_agent]
        if done_mask.any():
            env._reset_done_worlds(done_mask)
            env.sim.data.states.pos.block_until_ready()
        t_auto_reset.append(time.perf_counter() - t0)

        # 12. Get info (shared state for critic)
        t0 = time.perf_counter()
        info = env._get_info(
            bb_crash, rr_crash, br_crash, out_of_bounds, rewards,
            pre_reset_blue_alive, pre_reset_red_alive
        )
        t_get_info.append(time.perf_counter() - t0)

    # Convert to ms
    t_process_actions = np.array(t_process_actions) * 1000
    t_apply_controls = np.array(t_apply_controls) * 1000
    t_sim_step = np.array(t_sim_step) * 1000
    t_collisions = np.array(t_collisions) * 1000
    t_alive_update = np.array(t_alive_update) * 1000
    t_rewards = np.array(t_rewards) * 1000
    t_termination = np.array(t_termination) * 1000
    t_observations = np.array(t_observations) * 1000
    t_term_events = np.array(t_term_events) * 1000
    t_pre_reset_copy = np.array(t_pre_reset_copy) * 1000
    t_auto_reset = np.array(t_auto_reset) * 1000
    t_get_info = np.array(t_get_info) * 1000

    # --- Results ---
    print("\n" + "=" * 60)
    print("  RESULTS (per step, {} worlds × {} drones)".format(N_WORLDS, env_cfg.n_drones))
    print("=" * 60)

    component_sum = (
        np.mean(t_process_actions) + np.mean(t_apply_controls) + np.mean(t_sim_step) +
        np.mean(t_collisions) + np.mean(t_alive_update) + np.mean(t_rewards) +
        np.mean(t_termination) + np.mean(t_observations) + np.mean(t_term_events) +
        np.mean(t_pre_reset_copy) + np.mean(t_auto_reset) + np.mean(t_get_info)
    )

    def print_row(label, times, total=component_sum):
        mean = np.mean(times)
        pct = mean / total * 100
        print(f"  {label:<40} {mean:>7.3f} ms  ({pct:>5.1f}%)  std={np.std(times):.3f}")

    print()
    print_row("1. Process blue actions", t_process_actions)
    print_row("2. Apply controls (red + attitude)", t_apply_controls)
    print_row("3. Physics sim.step()", t_sim_step)
    print_row("4. Collision checking", t_collisions)
    print_row("5. Alive status update", t_alive_update)
    print_row("6. Reward computation", t_rewards)
    print_row("7. Termination/truncation checks", t_termination)
    print_row("8. Get observations", t_observations)
    print_row("9. Termination event tracking", t_term_events)
    print_row("10. Pre-reset alive copy", t_pre_reset_copy)
    print_row("11. Auto-reset", t_auto_reset)
    print_row("12. Get info (shared state)", t_get_info)
    print(f"  {'─'*65}")
    print(f"  {'Component sum':<40} {component_sum:>7.3f} ms")
    print(f"  {'Full step (measured)':<40} {np.mean(full_times):>7.3f} ms")
    print(f"  {'Overhead':<40} {np.mean(full_times) - component_sum:>7.3f} ms")

    # --- Inline step profiling (no forced blocking between ops) ---
    print("\n" + "=" * 60)
    print("  INLINE STEP (no forced blocking, lazy JAX)")
    print("=" * 60)
    inline_sum = sum(np.mean(np.array(v) * 1000) for v in t_sections.values())
    print(f"\n  {'Section':<40} {'Mean (ms)':<12} {'%':<8}")
    print(f"  {'─'*60}")
    for name, times in t_sections.items():
        ms = np.array(times) * 1000
        mean = np.mean(ms)
        print(f"  {name:<40} {mean:>7.3f} ms   ({mean/inline_sum*100:>5.1f}%)")
    print(f"  {'─'*60}")
    print(f"  {'Sum of sections':<40} {inline_sum:>7.3f} ms")
    print(f"  {'Total step (timed end-to-end)':<40} {np.mean(full_blocked_times):>7.3f} ms")
    print(f"  {'Original full step':<40} {np.mean(full_times):>7.3f} ms")

    # Group into categories
    print("\n" + "-" * 60)
    print("  GROUPED BREAKDOWN")
    print("-" * 60)

    physics = np.mean(t_sim_step)
    control = np.mean(t_apply_controls) + np.mean(t_process_actions)
    game_logic = np.mean(t_collisions) + np.mean(t_alive_update) + np.mean(t_rewards) + np.mean(t_termination) + np.mean(t_term_events)
    obs_build = np.mean(t_observations)
    info_build = np.mean(t_get_info) + np.mean(t_pre_reset_copy)
    reset = np.mean(t_auto_reset)

    total = physics + control + game_logic + obs_build + info_build + reset
    print(f"  {'Physics (sim.step)':<35} {physics:>7.3f} ms  ({physics/total*100:>5.1f}%)")
    print(f"  {'Control (red pursuit + att)':<35} {control:>7.3f} ms  ({control/total*100:>5.1f}%)")
    print(f"  {'Game logic (coll+alive+rew+term)':<35} {game_logic:>7.3f} ms  ({game_logic/total*100:>5.1f}%)")
    print(f"  {'Observations (per-agent)':<35} {obs_build:>7.3f} ms  ({obs_build/total*100:>5.1f}%)")
    print(f"  {'Info (shared state for critic)':<35} {info_build:>7.3f} ms  ({info_build/total*100:>5.1f}%)")
    print(f"  {'Auto-reset':<35} {reset:>7.3f} ms  ({reset/total*100:>5.1f}%)")

    # Per-drone and per-world costs
    print("\n" + "-" * 60)
    print("  PER-UNIT COSTS")
    print("-" * 60)
    print(f"  Per world:  {np.mean(full_times)/N_WORLDS*1000:.1f} us/world")
    print(f"  Per drone:  {np.mean(full_times)/env_cfg.n_drones*1000:.1f} us/drone")
    print(f"  Per substep: {np.mean(t_sim_step)/env_cfg.sim_steps_per_control:.3f} ms ({N_WORLDS} worlds × {env_cfg.n_drones} drones)")

    # Rollout phase estimate
    print("\n" + "-" * 60)
    print("  ROLLOUT PHASE ESTIMATE (256 steps)")
    print("-" * 60)
    rollout_env = 256 * np.mean(full_times) / 1000
    print(f"  Env steps total: {rollout_env:.1f} s")
    print(f"  Implied env it/s: {1000/np.mean(full_times):.1f}")

    print()
    env.close()


if __name__ == "__main__":
    main()
