"""Profile MPC batch scaling: mini_batches=4 (8192) vs mini_batches=1 (32768).

Tests whether reducing mini_batches from 4 to 1 speeds up the PPO update phase.
With mini_batches=1, each call processes 4x more samples but there are 4x fewer calls.

Usage:
    SCIPY_ARRAY_API=1 python tests/profile_minibatch_size.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import time
import numpy as np
import torch
import gymnasium

from crazyflie_mape_crazyflow.policies import LeapCSharedGaussianPolicy


def profile_forward_backward(policy, obs_tensor, state_tensor, n_steps=10, warmup=3):
    """Profile forward + backward for a given batch size."""
    # Warmup
    for _ in range(warmup):
        obs_t = obs_tensor.clone().requires_grad_(True)
        action = policy.mpc_layer(obs_t, state_tensor)
        loss = action.sum()
        loss.backward()

    fwd_times = []
    bwd_times = []
    for _ in range(n_steps):
        obs_t = obs_tensor.clone().requires_grad_(True)

        t0 = time.perf_counter()
        action = policy.mpc_layer(obs_t, state_tensor)
        t1 = time.perf_counter()
        loss = action.sum()
        loss.backward()
        t2 = time.perf_counter()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)

    return np.array(fwd_times), np.array(bwd_times)


def profile_forward_only(policy, obs_tensor, state_tensor, n_steps=20, warmup=5):
    """Profile forward pass only (rollout phase)."""
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


def main():
    N_WORLDS = 128
    N_PAIRS = 2
    ROLLOUTS = 256
    LEARNING_EPOCHS = 8
    DRONE_MODEL = "cf2x_T350"
    MPC_HORIZON = 2
    MPC_DT = 0.01
    OBS_DIM = 30  # Approximate obs dim

    # Batch sizes to test
    BATCH_4MB = (N_WORLDS * ROLLOUTS) // 4   # 8192 (mini_batches=4)
    BATCH_1MB = N_WORLDS * ROLLOUTS           # 32768 (mini_batches=1)
    BATCH_ROLLOUT = N_WORLDS * N_PAIRS        # 256 (rollout phase)

    print("=" * 60)
    print("  MINI-BATCH SIZE PROFILING")
    print("=" * 60)
    print(f"  n_worlds: {N_WORLDS}, rollouts: {ROLLOUTS}, epochs: {LEARNING_EPOCHS}")
    print(f"  mini_batches=4 → batch_size={BATCH_4MB}, calls/update={LEARNING_EPOCHS * 4}")
    print(f"  mini_batches=1 → batch_size={BATCH_1MB}, calls/update={LEARNING_EPOCHS * 1}")
    print(f"  Total solves/update: {LEARNING_EPOCHS * 4 * BATCH_4MB} (same both ways)")
    print()

    # Need n_batch_max to accommodate the largest batch
    n_batch_max = BATCH_1MB
    print(f"  Building solver with n_batch_max={n_batch_max}...")

    # Create policy
    obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    policy = LeapCSharedGaussianPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device="cpu",
        mpc_horizon=MPC_HORIZON,
        mpc_dt=MPC_DT,
        hidden_dim=256,
        roll_pitch_max=0.5,
        yaw_max=0.1,
        drone_model=DRONE_MODEL,
        n_batch_max=n_batch_max,
        velocity_max=None,
        verbose=False,
    )

    # Get state indices for building state tensor
    state_indices = policy.state_indices

    # --- Profile batch_size=8192 (mini_batches=4) ---
    print(f"\n--- Profiling batch_size={BATCH_4MB} (mini_batches=4) ---")
    obs_4 = torch.randn(BATCH_4MB, OBS_DIM, dtype=torch.float32)
    pos = obs_4[:, state_indices['position']]
    vel = obs_4[:, state_indices['velocity']]
    att = obs_4[:, state_indices['attitude']]
    rpy_rates = obs_4[:, state_indices['rpy_rates']]
    state_4 = torch.cat([pos, att, vel, rpy_rates], dim=-1)

    print("  Forward+backward...")
    fwd_4, bwd_4 = profile_forward_backward(policy, obs_4, state_4, n_steps=10, warmup=3)

    # --- Profile batch_size=32768 (mini_batches=1) ---
    print(f"\n--- Profiling batch_size={BATCH_1MB} (mini_batches=1) ---")
    obs_1 = torch.randn(BATCH_1MB, OBS_DIM, dtype=torch.float32)
    pos = obs_1[:, state_indices['position']]
    vel = obs_1[:, state_indices['velocity']]
    att = obs_1[:, state_indices['attitude']]
    rpy_rates = obs_1[:, state_indices['rpy_rates']]
    state_1 = torch.cat([pos, att, vel, rpy_rates], dim=-1)

    print("  Forward+backward...")
    fwd_1, bwd_1 = profile_forward_backward(policy, obs_1, state_1, n_steps=10, warmup=3)

    # --- Profile batch_size=256 (rollout phase, forward only) ---
    print(f"\n--- Profiling batch_size={BATCH_ROLLOUT} (rollout, forward only) ---")
    obs_r = torch.randn(BATCH_ROLLOUT, OBS_DIM, dtype=torch.float32)
    pos = obs_r[:, state_indices['position']]
    vel = obs_r[:, state_indices['velocity']]
    att = obs_r[:, state_indices['attitude']]
    rpy_rates = obs_r[:, state_indices['rpy_rates']]
    state_r = torch.cat([pos, att, vel, rpy_rates], dim=-1)

    print("  Forward only...")
    fwd_r = profile_forward_only(policy, obs_r, state_r, n_steps=20, warmup=5)

    # --- Results ---
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    def ms(t): return t * 1000

    print(f"\n  {'Metric':<40} {'batch=8192':<15} {'batch=32768':<15} {'Ratio':<10}")
    print(f"  {'-'*80}")

    fwd_4_mean = np.mean(ms(fwd_4))
    fwd_1_mean = np.mean(ms(fwd_1))
    bwd_4_mean = np.mean(ms(bwd_4))
    bwd_1_mean = np.mean(ms(bwd_1))

    print(f"  {'Forward (ms/call)':<40} {fwd_4_mean:<15.1f} {fwd_1_mean:<15.1f} {fwd_1_mean/fwd_4_mean:<10.2f}x")
    print(f"  {'Backward (ms/call)':<40} {bwd_4_mean:<15.1f} {bwd_1_mean:<15.1f} {bwd_1_mean/bwd_4_mean:<10.2f}x")
    print(f"  {'Fwd+Bwd (ms/call)':<40} {fwd_4_mean+bwd_4_mean:<15.1f} {fwd_1_mean+bwd_1_mean:<15.1f} {(fwd_1_mean+bwd_1_mean)/(fwd_4_mean+bwd_4_mean):<10.2f}x")

    # Per-solve cost
    fwd_4_per = fwd_4_mean / BATCH_4MB * 1000  # us/solve
    fwd_1_per = fwd_1_mean / BATCH_1MB * 1000
    bwd_4_per = bwd_4_mean / BATCH_4MB * 1000
    bwd_1_per = bwd_1_mean / BATCH_1MB * 1000

    print(f"\n  {'Forward (us/solve)':<40} {fwd_4_per:<15.1f} {fwd_1_per:<15.1f} {fwd_1_per/fwd_4_per:<10.2f}x")
    print(f"  {'Backward (us/solve)':<40} {bwd_4_per:<15.1f} {bwd_1_per:<15.1f} {bwd_1_per/bwd_4_per:<10.2f}x")

    # Total PPO update time
    print(f"\n  {'='*60}")
    print(f"  PPO UPDATE TOTAL TIME (8 epochs)")
    print(f"  {'='*60}")

    calls_4 = LEARNING_EPOCHS * 4  # 32 calls
    calls_1 = LEARNING_EPOCHS * 1  # 8 calls

    total_4 = calls_4 * (fwd_4_mean + bwd_4_mean) / 1000  # seconds
    total_1 = calls_1 * (fwd_1_mean + bwd_1_mean) / 1000

    print(f"  mini_batches=4: {calls_4} calls × {fwd_4_mean+bwd_4_mean:.0f} ms = {total_4:.1f} s")
    print(f"  mini_batches=1: {calls_1} calls × {fwd_1_mean+bwd_1_mean:.0f} ms = {total_1:.1f} s")
    print(f"  Speedup: {total_4/total_1:.2f}x")
    print(f"  Time saved per update: {total_4-total_1:.1f} s")

    # Rollout phase
    print(f"\n  {'='*60}")
    print(f"  ROLLOUT PHASE (256 steps, forward only)")
    print(f"  {'='*60}")
    fwd_r_mean = np.mean(ms(fwd_r))
    rollout_total = ROLLOUTS * fwd_r_mean / 1000
    print(f"  Forward per step: {fwd_r_mean:.1f} ms (batch={BATCH_ROLLOUT})")
    print(f"  Rollout total: {rollout_total:.1f} s")

    # Full iteration
    print(f"\n  {'='*60}")
    print(f"  FULL ITERATION (rollout + update)")
    print(f"  {'='*60}")
    iter_4 = rollout_total + total_4
    iter_1 = rollout_total + total_1
    print(f"  mini_batches=4: {rollout_total:.1f}s rollout + {total_4:.1f}s update = {iter_4:.1f} s/iter")
    print(f"  mini_batches=1: {rollout_total:.1f}s rollout + {total_1:.1f}s update = {iter_1:.1f} s/iter")
    print(f"  Speedup: {iter_4/iter_1:.2f}x")

    print()


if __name__ == "__main__":
    main()
