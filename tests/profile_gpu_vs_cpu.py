"""Profile GPU vs CPU for MPC policy at different batch sizes.

The MPC solver (acados) always runs on CPU. With device="cuda":
- Cost network runs on GPU
- Data transfers: GPU→CPU before MPC, CPU→GPU after MPC
- Question: does GPU help the neural net enough to offset transfer cost?

Usage:
    SCIPY_ARRAY_API=1 python tests/profile_gpu_vs_cpu.py
"""

import os
os.environ["SCIPY_ARRAY_API"] = "1"

import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import time
import numpy as np
import torch
import gymnasium

from crazyflie_mape_crazyflow.policies import LeapCSharedGaussianPolicy


def profile_forward_backward(policy, obs_tensor, state_tensor, n_steps=10, warmup=3):
    """Profile forward + backward."""
    for _ in range(warmup):
        obs_t = obs_tensor.clone().requires_grad_(True)
        action = policy.mpc_layer(obs_t, state_tensor)
        loss = action.sum()
        loss.backward()
        if obs_tensor.is_cuda:
            torch.cuda.synchronize()

    fwd_times = []
    bwd_times = []
    for _ in range(n_steps):
        obs_t = obs_tensor.clone().requires_grad_(True)
        if obs_tensor.is_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        action = policy.mpc_layer(obs_t, state_tensor)
        if obs_tensor.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss = action.sum()
        loss.backward()
        if obs_tensor.is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)

    return np.array(fwd_times), np.array(bwd_times)


def profile_cost_net_only(policy, obs_tensor, n_steps=50, warmup=10):
    """Profile just the cost network (no MPC)."""
    for _ in range(warmup):
        with torch.no_grad():
            out = policy.mpc_layer.cost_net(obs_tensor)
        if obs_tensor.is_cuda:
            torch.cuda.synchronize()

    times = []
    for _ in range(n_steps):
        if obs_tensor.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = policy.mpc_layer.cost_net(obs_tensor)
        if obs_tensor.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return np.array(times)


def main():
    DRONE_MODEL = "cf2x_T350"
    MPC_HORIZON = 2
    MPC_DT = 0.01
    OBS_DIM = 30

    BATCH_SIZES = [256, 8192, 32768]
    N_BATCH_MAX = 32768

    print("=" * 70)
    print("  GPU vs CPU PROFILING FOR MPC POLICY")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  n_batch_max: {N_BATCH_MAX}")
    print(f"  Batch sizes to test: {BATCH_SIZES}")
    print()

    # --- Build CPU policy ---
    print("Building CPU policy...")
    obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    policy_cpu = LeapCSharedGaussianPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device="cpu",
        mpc_horizon=MPC_HORIZON,
        mpc_dt=MPC_DT,
        hidden_dim=256,
        roll_pitch_max=0.5,
        yaw_max=0.1,
        drone_model=DRONE_MODEL,
        n_batch_max=N_BATCH_MAX,
        velocity_max=None,
        verbose=False,
    )
    state_indices = policy_cpu.state_indices

    # --- Build GPU policy (same weights, MPC solver shared) ---
    print("Building GPU policy...")
    policy_gpu = LeapCSharedGaussianPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device="cuda",
        mpc_horizon=MPC_HORIZON,
        mpc_dt=MPC_DT,
        hidden_dim=256,
        roll_pitch_max=0.5,
        yaw_max=0.1,
        drone_model=DRONE_MODEL,
        n_batch_max=N_BATCH_MAX,
        velocity_max=None,
        verbose=False,
    )
    policy_gpu.mpc_layer = policy_gpu.mpc_layer.to("cuda")

    # --- Profile each batch size ---
    results = {}

    for batch_size in BATCH_SIZES:
        print(f"\n{'─'*70}")
        print(f"  Batch size: {batch_size}")
        print(f"{'─'*70}")

        # Create data
        obs_np = np.random.randn(batch_size, OBS_DIM).astype(np.float32)

        # CPU tensors
        obs_cpu = torch.tensor(obs_np, dtype=torch.float32, device="cpu")
        pos = obs_cpu[:, state_indices['position']]
        vel = obs_cpu[:, state_indices['velocity']]
        att = obs_cpu[:, state_indices['attitude']]
        rpy_rates = obs_cpu[:, state_indices['rpy_rates']]
        state_cpu = torch.cat([pos, att, vel, rpy_rates], dim=-1)

        # GPU tensors
        obs_gpu = obs_cpu.to("cuda")
        state_gpu = state_cpu.to("cuda")

        # Profile cost net only (to see GPU benefit for neural net)
        print("  Cost net only...")
        cost_cpu = profile_cost_net_only(policy_cpu, obs_cpu)
        cost_gpu = profile_cost_net_only(policy_gpu, obs_gpu)

        # Profile full forward+backward (CPU)
        n_steps = max(3, 20 // (batch_size // 256))
        warmup = max(1, n_steps // 3)

        print(f"  Full fwd+bwd CPU ({n_steps} steps)...")
        fwd_cpu, bwd_cpu = profile_forward_backward(
            policy_cpu, obs_cpu, state_cpu, n_steps=n_steps, warmup=warmup
        )

        # Profile full forward+backward (GPU)
        print(f"  Full fwd+bwd GPU ({n_steps} steps)...")
        fwd_gpu, bwd_gpu = profile_forward_backward(
            policy_gpu, obs_gpu, state_gpu, n_steps=n_steps, warmup=warmup
        )

        results[batch_size] = {
            'cost_cpu': cost_cpu, 'cost_gpu': cost_gpu,
            'fwd_cpu': fwd_cpu, 'fwd_gpu': fwd_gpu,
            'bwd_cpu': bwd_cpu, 'bwd_gpu': bwd_gpu,
        }

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Batch':<8} {'Component':<20} {'CPU (ms)':<12} {'GPU (ms)':<12} {'GPU/CPU':<10}")
    print(f"  {'─'*62}")

    for batch_size in BATCH_SIZES:
        r = results[batch_size]
        cost_cpu_ms = np.mean(r['cost_cpu']) * 1000
        cost_gpu_ms = np.mean(r['cost_gpu']) * 1000
        fwd_cpu_ms = np.mean(r['fwd_cpu']) * 1000
        fwd_gpu_ms = np.mean(r['fwd_gpu']) * 1000
        bwd_cpu_ms = np.mean(r['bwd_cpu']) * 1000
        bwd_gpu_ms = np.mean(r['bwd_gpu']) * 1000

        print(f"  {batch_size:<8} {'Cost net':<20} {cost_cpu_ms:<12.2f} {cost_gpu_ms:<12.2f} {cost_gpu_ms/cost_cpu_ms:<10.2f}x")
        print(f"  {'':<8} {'MPC forward':<20} {fwd_cpu_ms:<12.1f} {fwd_gpu_ms:<12.1f} {fwd_gpu_ms/fwd_cpu_ms:<10.2f}x")
        print(f"  {'':<8} {'MPC backward':<20} {bwd_cpu_ms:<12.1f} {bwd_gpu_ms:<12.1f} {bwd_gpu_ms/bwd_cpu_ms:<10.2f}x")
        print(f"  {'':<8} {'Total fwd+bwd':<20} {fwd_cpu_ms+bwd_cpu_ms:<12.1f} {fwd_gpu_ms+bwd_gpu_ms:<12.1f} {(fwd_gpu_ms+bwd_gpu_ms)/(fwd_cpu_ms+bwd_cpu_ms):<10.2f}x")
        print()

    # PPO update comparison
    print(f"\n  {'='*70}")
    print(f"  PPO UPDATE COMPARISON (8 epochs)")
    print(f"  {'='*70}")

    EPOCHS = 8
    for mini_batches, batch_size in [(4, 8192), (1, 32768)]:
        r = results[batch_size]
        calls = EPOCHS * mini_batches
        fwd_cpu_ms = np.mean(r['fwd_cpu']) * 1000
        bwd_cpu_ms = np.mean(r['bwd_cpu']) * 1000
        fwd_gpu_ms = np.mean(r['fwd_gpu']) * 1000
        bwd_gpu_ms = np.mean(r['bwd_gpu']) * 1000

        total_cpu = calls * (fwd_cpu_ms + bwd_cpu_ms) / 1000
        total_gpu = calls * (fwd_gpu_ms + bwd_gpu_ms) / 1000

        print(f"\n  mini_batches={mini_batches} (batch={batch_size}, {calls} calls):")
        print(f"    CPU: {total_cpu:.1f} s")
        print(f"    GPU: {total_gpu:.1f} s")
        print(f"    Ratio: {total_gpu/total_cpu:.2f}x {'(GPU faster)' if total_gpu < total_cpu else '(CPU faster)'}")

    print()


if __name__ == "__main__":
    main()
