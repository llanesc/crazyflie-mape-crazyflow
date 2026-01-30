"""Profile individual components of a training step.

Measures time spent in:
1. Policy forward (neural net + MPC solve)
2. Environment step (JAX physics + collision + rewards)
3. Observation building
4. Action rescaling/processing

Usage:
    SCIPY_ARRAY_API=1 python tests/profile_training_step.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import time
from pathlib import Path

import numpy as np
import torch
import jax

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig, RescaleActionWrapper
from crazyflie_mape_crazyflow.envs.spawn import create_spawn_fn_from_config
from crazyflie_mape_crazyflow.policies import LeapCSharedGaussianPolicy


def profile_env_step(env, actions, n_steps=100, warmup=10):
    """Profile environment step (physics + collision + rewards + obs)."""
    # Warmup (JIT compilation)
    for _ in range(warmup):
        obs, rewards, terminated, truncated, info = env.step(actions)

    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        obs, rewards, terminated, truncated, info = env.step(actions)
        times.append(time.perf_counter() - t0)

    return np.array(times)


def profile_policy_forward(policy, obs_tensor, state_tensor, n_steps=100, warmup=10):
    """Profile policy forward pass (neural net + MPC)."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            action = policy.mpc_layer(obs_tensor, state_tensor)

    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        with torch.no_grad():
            action = policy.mpc_layer(obs_tensor, state_tensor)
        times.append(time.perf_counter() - t0)

    return np.array(times)


def profile_policy_forward_backward(policy, obs_tensor, state_tensor, n_steps=50, warmup=5):
    """Profile policy forward + backward pass."""
    # Warmup
    for _ in range(warmup):
        obs_t = obs_tensor.clone().requires_grad_(True)
        action = policy.mpc_layer(obs_t, state_tensor)
        loss = action.sum()
        loss.backward()

    times_fwd = []
    times_bwd = []
    for _ in range(n_steps):
        obs_t = obs_tensor.clone().requires_grad_(True)

        t0 = time.perf_counter()
        action = policy.mpc_layer(obs_t, state_tensor)
        t1 = time.perf_counter()
        loss = action.sum()
        loss.backward()
        t2 = time.perf_counter()

        times_fwd.append(t1 - t0)
        times_bwd.append(t2 - t1)

    return np.array(times_fwd), np.array(times_bwd)


def profile_cost_net(policy, obs_tensor, n_steps=200, warmup=20):
    """Profile just the cost network forward."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            out = policy.mpc_layer.cost_net(obs_tensor)

    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = policy.mpc_layer.cost_net(obs_tensor)
        times.append(time.perf_counter() - t0)

    return np.array(times)


def profile_mpc_only(policy, obs_tensor, state_tensor, n_steps=100, warmup=10):
    """Profile just the MPC solve (excluding neural net)."""
    # Get cost params once
    with torch.no_grad():
        cost_net_out = policy.mpc_layer.cost_net(obs_tensor)
        mpc_params = policy.mpc_layer._scale_parameters(cost_net_out, obs_tensor.shape[0])

    # Extract state
    state_indices = policy.state_indices
    batch_size = obs_tensor.shape[0]
    pos = obs_tensor[:, state_indices['position']]
    vel = obs_tensor[:, state_indices['velocity']]
    att = obs_tensor[:, state_indices['attitude']]
    rpy_rates = obs_tensor[:, state_indices['rpy_rates']]
    mpc_state = torch.cat([pos, att, vel, rpy_rates], dim=-1)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _, u0, x_traj, u_traj, value = policy.mpc_layer.planner(
                obs=mpc_state, param=mpc_params
            )

    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        with torch.no_grad():
            _, u0, x_traj, u_traj, value = policy.mpc_layer.planner(
                obs=mpc_state, param=mpc_params
            )
        times.append(time.perf_counter() - t0)

    return np.array(times)


def main():
    N_WORLDS = 128
    N_PAIRS = 2
    DRONE_MODEL = "cf2x_T350"
    MPC_HORIZON = 2
    MPC_DT = 0.01
    ROLL_PITCH_MAX = 0.5
    YAW_MAX = 0.1
    N_PROFILE_STEPS = 100

    print("="*60)
    print("  TRAINING STEP PROFILING")
    print("="*60)
    print(f"  n_worlds: {N_WORLDS}")
    print(f"  n_pairs: {N_PAIRS} (= {N_PAIRS} blue + {N_PAIRS} red)")
    print(f"  MPC solves per step: {N_WORLDS * N_PAIRS}")
    print(f"  mpc_horizon: {MPC_HORIZON}")
    print(f"  mpc_dt: {MPC_DT}")
    print()

    # --- Create environment ---
    print("Creating environment...")
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
    raw_env = RedVsBlueEnv(cfg=env_cfg, spawn_fn=spawn_fn)
    env = RescaleActionWrapper(raw_env)

    # Get spaces
    obs_dim = raw_env.obs_dim
    print(f"  obs_dim per agent: {obs_dim}")
    print(f"  total drones: {env_cfg.n_drones}")
    print(f"  sim_steps_per_control: {env_cfg.sim_steps_per_control}")

    # --- Create policy ---
    print("\nCreating policy (builds MPC solver)...")
    import gymnasium
    obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    n_batch_max = max(N_WORLDS, (N_WORLDS * 256) // 4)  # Same as training script
    print(f"  n_batch_max: {n_batch_max}")

    policy = LeapCSharedGaussianPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device="cpu",
        mpc_horizon=MPC_HORIZON,
        mpc_dt=MPC_DT,
        hidden_dim=256,
        roll_pitch_max=ROLL_PITCH_MAX,
        yaw_max=YAW_MAX,
        drone_model=DRONE_MODEL,
        n_batch_max=n_batch_max,
        velocity_max=None,
        verbose=False,
    )

    # --- Get initial observations ---
    print("\nResetting environment...")
    obs_dict, _ = env.reset()

    # Stack observations for all blue agents
    agent_names = list(obs_dict.keys())
    obs_list = [obs_dict[name] for name in agent_names]
    obs_stacked = np.concatenate(obs_list, axis=0)  # (n_worlds * n_blue, obs_dim)
    obs_tensor = torch.tensor(obs_stacked, dtype=torch.float32)

    # Extract state for MPC
    state_indices = policy.state_indices
    pos = obs_tensor[:, state_indices['position']]
    vel = obs_tensor[:, state_indices['velocity']]
    att = obs_tensor[:, state_indices['attitude']]
    rpy_rates = obs_tensor[:, state_indices['rpy_rates']]
    state_tensor = torch.cat([pos, att, vel, rpy_rates], dim=-1)

    print(f"  obs_stacked shape: {obs_stacked.shape}")
    print(f"  state_tensor shape: {state_tensor.shape}")

    # Create random actions for env stepping
    random_actions = {}
    for name in agent_names:
        random_actions[name] = np.random.uniform(-1, 1, (N_WORLDS, 4)).astype(np.float32)

    # --- Profile each component ---
    print(f"\nProfiling ({N_PROFILE_STEPS} iterations each, with warmup)...")
    print()

    # 1. Full environment step
    print("  [1/6] Environment step (physics + collision + obs)...")
    env_times = profile_env_step(env, random_actions, n_steps=N_PROFILE_STEPS)

    # 2. Cost network only
    print("  [2/6] Cost network forward...")
    cost_net_times = profile_cost_net(policy, obs_tensor, n_steps=N_PROFILE_STEPS)

    # 3. MPC solve only
    print("  [3/6] MPC solve only (no neural net)...")
    mpc_times = profile_mpc_only(policy, obs_tensor, state_tensor, n_steps=N_PROFILE_STEPS)

    # 4. Full policy forward (net + MPC)
    print("  [4/6] Full policy forward (net + MPC)...")
    policy_fwd_times = profile_policy_forward(policy, obs_tensor, state_tensor, n_steps=N_PROFILE_STEPS)

    # 5. Policy forward + backward
    print("  [5/6] Policy forward + backward...")
    fwd_times, bwd_times = profile_policy_forward_backward(
        policy, obs_tensor, state_tensor, n_steps=min(N_PROFILE_STEPS, 50)
    )

    # 6. State extraction overhead
    print("  [6/6] State extraction + action rescaling...")
    def profile_overhead(n_steps=500):
        times = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            pos = obs_tensor[:, state_indices['position']]
            vel = obs_tensor[:, state_indices['velocity']]
            att = obs_tensor[:, state_indices['attitude']]
            rpy_rates = obs_tensor[:, state_indices['rpy_rates']]
            _ = torch.cat([pos, att, vel, rpy_rates], dim=-1)
            times.append(time.perf_counter() - t0)
        return np.array(times)
    overhead_times = profile_overhead()

    # --- Results ---
    print("\n" + "="*60)
    print("  RESULTS (per step, batch of {} MPC solves)".format(N_WORLDS * N_PAIRS))
    print("="*60)

    def print_timing(label, times_ms):
        print(f"  {label:<40} {np.mean(times_ms):.2f} ms  (std: {np.std(times_ms):.2f}, p95: {np.percentile(times_ms, 95):.2f})")

    env_ms = env_times * 1000
    cost_net_ms = cost_net_times * 1000
    mpc_ms = mpc_times * 1000
    policy_fwd_ms = policy_fwd_times * 1000
    fwd_ms = fwd_times * 1000
    bwd_ms = bwd_times * 1000
    overhead_ms = overhead_times * 1000

    print()
    print_timing("Environment step (full)", env_ms)
    print_timing("Cost network forward", cost_net_ms)
    print_timing("MPC solve only", mpc_ms)
    print_timing("Policy forward (net + MPC)", policy_fwd_ms)
    print_timing("Policy forward (in fwd+bwd)", fwd_ms)
    print_timing("Policy backward", bwd_ms)
    print_timing("State extraction overhead", overhead_ms)

    # Breakdown
    print("\n" + "-"*60)
    print("  BREAKDOWN: Single Rollout Step")
    print("-"*60)
    env_mean = np.mean(env_ms)
    policy_mean = np.mean(policy_fwd_ms)
    total_step = env_mean + policy_mean
    print(f"  Environment step:    {env_mean:.2f} ms  ({env_mean/total_step*100:.1f}%)")
    print(f"  Policy forward:      {policy_mean:.2f} ms  ({policy_mean/total_step*100:.1f}%)")
    print(f"  ----- Total:         {total_step:.2f} ms")
    print(f"  Implied it/s:        {1000/total_step:.1f}")

    print("\n" + "-"*60)
    print("  BREAKDOWN: Policy Forward")
    print("-"*60)
    cost_mean = np.mean(cost_net_ms)
    mpc_mean = np.mean(mpc_ms)
    other = policy_mean - cost_mean - mpc_mean
    print(f"  Cost network:        {cost_mean:.2f} ms  ({cost_mean/policy_mean*100:.1f}%)")
    print(f"  MPC solve:           {mpc_mean:.2f} ms  ({mpc_mean/policy_mean*100:.1f}%)")
    print(f"  Other (scaling etc): {max(0, other):.2f} ms")

    print("\n" + "-"*60)
    print("  BREAKDOWN: Training Step (fwd + bwd)")
    print("-"*60)
    fwd_mean = np.mean(fwd_ms)
    bwd_mean = np.mean(bwd_ms)
    total_train = env_mean + fwd_mean + bwd_mean
    print(f"  Environment step:    {env_mean:.2f} ms  ({env_mean/total_train*100:.1f}%)")
    print(f"  Policy forward:      {fwd_mean:.2f} ms  ({fwd_mean/total_train*100:.1f}%)")
    print(f"  Policy backward:     {bwd_mean:.2f} ms  ({bwd_mean/total_train*100:.1f}%)")
    print(f"  ----- Total:         {total_train:.2f} ms")
    print(f"  Implied it/s:        {1000/total_train:.1f}")

    print("\n" + "-"*60)
    print("  PER-SOLVE COSTS")
    print("-"*60)
    n_solves = N_WORLDS * N_PAIRS
    print(f"  MPC forward:         {np.mean(mpc_ms)/n_solves*1000:.1f} us/solve  ({n_solves} solves)")
    print(f"  MPC backward:        {bwd_mean/n_solves*1000:.1f} us/solve  ({n_solves} solves)")
    print(f"  Env step/drone:      {env_mean/env_cfg.n_drones*1000:.1f} us/drone  ({env_cfg.n_drones} drones × {env_cfg.sim_steps_per_control} substeps)")

    print()
    env.close()


if __name__ == "__main__":
    main()
