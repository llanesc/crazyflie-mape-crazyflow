#!/usr/bin/env python3
"""Debug script to check policy outputs."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import torch
import numpy as np
from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig
from crazyflie_mape_crazyflow.policies import LeapCSharedGaussianPolicy

# Policy settings (normally from YAML config)
MAX_ROLL_PITCH = 0.5  # rad
MAX_YAW = 0.5  # rad
MPC_HORIZON = 2
MPC_DT = 0.01

# Create environment
env_cfg = RedVsBlueEnvConfig(n_pairs=2, n_worlds=1, device='cpu')
env = RedVsBlueEnv(cfg=env_cfg)

print(f'Mass: {env_cfg.mass}')
print(f'Gravity: {env_cfg.gravity}')
print(f'Hover thrust: {env_cfg.mass * env_cfg.gravity}')
print(f'Min thrust: {env_cfg.thrust_min}')
print(f'Max thrust: {env_cfg.thrust_max}')
print(f'Thrust mean: {(env_cfg.thrust_min + env_cfg.thrust_max) / 2.0}')
print(f'Thrust scale: {(env_cfg.thrust_max - env_cfg.thrust_min) / 2.0}')

# Create policy
sample_obs_space = env.observation_space[env.possible_agents[0]]
sample_action_space = env.action_space[env.possible_agents[0]]

policy = LeapCSharedGaussianPolicy(
    observation_space=sample_obs_space,
    action_space=sample_action_space,
    device='cpu',
    mpc_horizon=MPC_HORIZON,
    hidden_dim=256,
    max_roll_pitch=MAX_ROLL_PITCH,
    max_yaw=MAX_YAW,
    drone_model=env_cfg.drone_model,
)

# Load checkpoint
checkpoint = torch.load('results/red_vs_blue_mappo_20260116_032420/red_vs_blue_mappo/checkpoints/best_agent.pt', map_location='cpu')
policy.load_state_dict(checkpoint['blue_0']['policy'])
policy.eval()

# Reset and get observation
obs_dict, info = env.reset()
obs = obs_dict[env.possible_agents[0]]

print(f'\nObservation shape: {obs.shape}')
print(f'Own pos: {obs[0, :3]}')
print(f'Own vel: {obs[0, 3:6]}')
print(f'Own rotmat: {obs[0, 6:15]}')
print(f'Own body_rates: {obs[0, 15:18]}')

# Get action from policy
obs_tensor = torch.tensor(obs, dtype=torch.float32)
with torch.no_grad():
    action, log_std, _ = policy.compute({'states': obs_tensor}, role='')

print(f'\nNormalized action: {action.numpy()[0]}')

# Denormalize to see physical values
thrust_mean = (env_cfg.thrust_min + env_cfg.thrust_max) / 2.0
thrust_scale = (env_cfg.thrust_max - env_cfg.thrust_min) / 2.0
action_mean = np.array([0.0, 0.0, 0.0, thrust_mean])
action_scale = np.array([MAX_ROLL_PITCH, MAX_ROLL_PITCH, MAX_YAW, thrust_scale])

physical_action = action.numpy()[0] * action_scale + action_mean
print(f'Physical action (roll, pitch, yaw, thrust): {physical_action}')
print(f'Physical thrust: {physical_action[3]:.6f} N')
print(f'Hover thrust: {env_cfg.mass * env_cfg.gravity:.6f} N')
print(f'Thrust ratio to hover: {physical_action[3] / (env_cfg.mass * env_cfg.gravity):.2f}')

# Also check what the MPC layer is outputting before normalization
print('\n--- MPC Layer Details ---')
print(f'MPC action_mean: {policy.mpc_layer.action_mean}')
print(f'MPC action_scale: {policy.mpc_layer.action_scale}')

env.close()
