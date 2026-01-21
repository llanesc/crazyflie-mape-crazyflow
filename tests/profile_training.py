#!/usr/bin/env python3
"""Profiling script to identify training bottlenecks.

Measures time spent in:
- Environment reset
- Policy forward pass (FFN vs MPC)
- Environment step
- Backward pass / gradient computation
"""

import argparse
import os
import time

# Parse device argument early to set JAX platform before imports
def _get_device_from_args():
    """Parse just the device argument before other imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, default="cpu")
    args, _ = parser.parse_known_args()
    return args.device

_device = _get_device_from_args()
if _device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import numpy as np
import torch

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig
from crazyflie_mape_crazyflow.policies import (
    FFNSharedGaussianPolicy,
    LeapCSharedGaussianPolicy,
    SharedCritic,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile training components")
    parser.add_argument("--n-pairs", type=int, default=2)
    parser.add_argument("--n-worlds", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-steps", type=int, default=100, help="Number of steps to profile")
    parser.add_argument("--policy", type=str, default="both", choices=["ffn", "mpc", "both"])
    return parser.parse_args()


def profile_policy(policy, policy_name, env, n_steps, device):
    """Profile a single policy."""
    print(f"\n{'='*60}")
    print(f"Profiling {policy_name} Policy")
    print(f"{'='*60}")

    # Timing accumulators
    reset_times = []
    policy_forward_times = []
    env_step_times = []
    backward_times = []

    # Reset environment
    t0 = time.perf_counter()
    obs_dict, _ = env.reset()
    reset_times.append(time.perf_counter() - t0)

    # Create a simple loss function for backward pass timing
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    for step in range(n_steps):
        actions = {}

        # Policy forward pass
        t0 = time.perf_counter()
        for agent_name in env.possible_agents:
            obs = obs_dict[agent_name]
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            action, log_std, _ = policy.compute({"states": obs_tensor}, role="")
            actions[agent_name] = action.detach().cpu().numpy()
        policy_forward_times.append(time.perf_counter() - t0)

        # Environment step
        t0 = time.perf_counter()
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        env_step_times.append(time.perf_counter() - t0)

        # Simulate backward pass (like in PPO update)
        t0 = time.perf_counter()
        optimizer.zero_grad()
        # Get fresh forward pass for gradient computation
        sample_obs = torch.tensor(
            obs_dict[env.possible_agents[0]], dtype=torch.float32, device=device
        )
        action_out, log_std, _ = policy.compute({"states": sample_obs}, role="")
        # Simple loss to trigger backward
        loss = action_out.mean() + log_std.mean()
        loss.backward()
        optimizer.step()
        backward_times.append(time.perf_counter() - t0)

        # Reset if any world is done
        any_terminated = any(v.any() if hasattr(v, 'any') else v for v in terminated.values())
        any_truncated = any(v.any() if hasattr(v, 'any') else v for v in truncated.values())
        if any_terminated or any_truncated:
            t0 = time.perf_counter()
            obs_dict, _ = env.reset()
            reset_times.append(time.perf_counter() - t0)

    # Print results
    print(f"\nResults over {n_steps} steps:")
    print(f"  Reset:          {np.mean(reset_times)*1000:8.2f} ms (n={len(reset_times)})")
    print(f"  Policy Forward: {np.mean(policy_forward_times)*1000:8.2f} ms")
    print(f"  Env Step:       {np.mean(env_step_times)*1000:8.2f} ms")
    print(f"  Backward:       {np.mean(backward_times)*1000:8.2f} ms")

    total_per_step = (
        np.mean(policy_forward_times) +
        np.mean(env_step_times) +
        np.mean(backward_times)
    )
    print(f"\n  Total per step: {total_per_step*1000:8.2f} ms")
    print(f"  Steps per sec:  {1/total_per_step:8.1f}")

    # Breakdown percentages
    print(f"\n  Breakdown:")
    print(f"    Policy Forward: {np.mean(policy_forward_times)/total_per_step*100:5.1f}%")
    print(f"    Env Step:       {np.mean(env_step_times)/total_per_step*100:5.1f}%")
    print(f"    Backward:       {np.mean(backward_times)/total_per_step*100:5.1f}%")

    return {
        "reset": np.mean(reset_times),
        "policy_forward": np.mean(policy_forward_times),
        "env_step": np.mean(env_step_times),
        "backward": np.mean(backward_times),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Policy settings (normally from YAML config)
    MAX_ROLL_PITCH = 0.5  # rad
    MAX_YAW = 0.5  # rad
    MPC_HORIZON = 2
    MPC_DT = 0.01

    print(f"Profiling Configuration:")
    print(f"  n_pairs: {args.n_pairs}")
    print(f"  n_worlds: {args.n_worlds}")
    print(f"  device: {args.device}")
    print(f"  n_steps: {args.n_steps}")

    # Create environment
    env_cfg = RedVsBlueEnvConfig(
        n_pairs=args.n_pairs,
        n_worlds=args.n_worlds,
        pursuer_strategy="ProNav",
        device=args.device,
    )
    env = RedVsBlueEnv(cfg=env_cfg)

    sample_obs_space = env.observation_space[env.possible_agents[0]]
    sample_action_space = env.action_space[env.possible_agents[0]]

    print(f"\nObservation dim: {sample_obs_space.shape[0]}")
    print(f"Action dim: {sample_action_space.shape[0]}")

    results = {}

    # Profile FFN
    if args.policy in ["ffn", "both"]:
        ffn_policy = FFNSharedGaussianPolicy(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            device=device,
            hidden_sizes=(256, 256),
        )
        print(f"\nFFN Policy parameters: {sum(p.numel() for p in ffn_policy.parameters())}")
        results["ffn"] = profile_policy(ffn_policy, "FFN", env, args.n_steps, device)

    # Profile MPC
    if args.policy in ["mpc", "both"]:
        mpc_policy = LeapCSharedGaussianPolicy(
            observation_space=sample_obs_space,
            action_space=sample_action_space,
            device=device,
            mpc_horizon=MPC_HORIZON,
            mpc_dt=MPC_DT,
            hidden_dim=256,
            max_roll_pitch=MAX_ROLL_PITCH,
            max_yaw=MAX_YAW,
            drone_model=env_cfg.drone_model,
            n_batch_max=args.n_worlds * 2,
        )
        print(f"\nMPC Policy parameters: {sum(p.numel() for p in mpc_policy.parameters())}")
        results["mpc"] = profile_policy(mpc_policy, "MPC", env, args.n_steps, device)

    # Compare if both were profiled
    if args.policy == "both":
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Component':<20} {'FFN (ms)':<12} {'MPC (ms)':<12} {'Ratio':<10}")
        print("-" * 54)
        for key in ["policy_forward", "env_step", "backward"]:
            ffn_t = results["ffn"][key] * 1000
            mpc_t = results["mpc"][key] * 1000
            ratio = mpc_t / ffn_t if ffn_t > 0 else float('inf')
            print(f"{key:<20} {ffn_t:<12.2f} {mpc_t:<12.2f} {ratio:<10.2f}x")

    env.close()


if __name__ == "__main__":
    main()
