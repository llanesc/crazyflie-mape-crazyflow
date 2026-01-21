#!/usr/bin/env python3
"""Debug script to trace the control pipeline."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig

# Create environment
env_cfg = RedVsBlueEnvConfig(n_pairs=2, n_worlds=1, device='cpu')
env = RedVsBlueEnv(cfg=env_cfg)

print("=== Environment Config ===")
print(f"Mass: {env_cfg.mass}")
print(f"Gravity: {env_cfg.gravity}")
print(f"Hover thrust: {env_cfg.mass * env_cfg.gravity}")
print(f"Min thrust: {env_cfg.min_thrust}")
print(f"Max thrust: {env_cfg.max_thrust}")

# Reset environment
obs_dict, info = env.reset()

# Check initial state
states = env.sim.data.states
print("\n=== Initial State ===")
print(f"Position: {np.array(states.pos[0, 0])}")
print(f"Velocity: {np.array(states.vel[0, 0])}")
print(f"Quat: {np.array(states.quat[0, 0])}")
print(f"Ang vel: {np.array(states.ang_vel[0, 0])}")
print(f"Rotor vel: {np.array(states.rotor_vel[0, 0]) if states.rotor_vel is not None else 'None'}")

# Check control data
controls = env.sim.data.controls
print("\n=== Control Data ===")
print(f"Rotor vel cmd: {np.array(controls.rotor_vel[0, 0]) if controls.rotor_vel is not None else 'None'}")

if controls.attitude is not None:
    print(f"Attitude cmd: {np.array(controls.attitude.cmd[0, 0])}")
    print(f"Attitude staged_cmd: {np.array(controls.attitude.staged_cmd[0, 0])}")
    print(f"Attitude params keys: {controls.attitude.params._asdict().keys() if hasattr(controls.attitude.params, '_asdict') else controls.attitude.params}")
    # Check thrust_max
    if hasattr(controls.attitude.params, 'thrust_max'):
        print(f"Attitude thrust_max: {controls.attitude.params.thrust_max}")

if controls.force_torque is not None:
    print(f"Force torque cmd: {np.array(controls.force_torque.cmd[0, 0])}")
    print(f"Force torque params: {controls.force_torque.params._asdict().keys() if hasattr(controls.force_torque.params, '_asdict') else 'N/A'}")

# Check physics params
params = env.sim.data.params
print("\n=== Physics Params ===")
print(f"Params type: {type(params)}")
if hasattr(params, 'mass'):
    print(f"Mass: {np.array(params.mass[0, 0])}")
if hasattr(params, 'rpm2thrust'):
    print(f"rpm2thrust: {np.array(params.rpm2thrust)}")
if hasattr(params, 'gravity_vec'):
    print(f"Gravity vec: {np.array(params.gravity_vec)}")

# Now let's apply a hover thrust and see what happens
print("\n=== Testing hover thrust ===")
hover_thrust = env_cfg.mass * env_cfg.gravity
print(f"Hover thrust to apply: {hover_thrust}")

# Create action dict with hover thrust (normalized)
thrust_mean = (env_cfg.min_thrust + env_cfg.max_thrust) / 2.0
thrust_scale = (env_cfg.max_thrust - env_cfg.min_thrust) / 2.0
normalized_thrust = (hover_thrust - thrust_mean) / thrust_scale
print(f"Normalized thrust: {normalized_thrust}")

actions = {}
for agent in env.possible_agents:
    # [roll, pitch, yaw, thrust_normalized]
    actions[agent] = np.array([[0.0, 0.0, 0.0, normalized_thrust]], dtype=np.float32)

# Step the environment
obs_dict, rewards, terminated, truncated, info = env.step(actions)

# Check state after step
states = env.sim.data.states
print("\n=== State After Step ===")
print(f"Position: {np.array(states.pos[0, 0])}")
print(f"Velocity: {np.array(states.vel[0, 0])}")
print(f"Rotor vel: {np.array(states.rotor_vel[0, 0]) if states.rotor_vel is not None else 'None'}")

# Check derivatives
states_deriv = env.sim.data.states_deriv
print("\n=== State Derivatives ===")
print(f"Vel (pos_dot): {np.array(states_deriv.vel[0, 0])}")
print(f"Acc (vel_dot): {np.array(states_deriv.acc[0, 0])}")
print(f"Ang acc: {np.array(states_deriv.ang_acc[0, 0])}")
if states_deriv.rotor_acc is not None:
    print(f"Rotor acc: {np.array(states_deriv.rotor_acc[0, 0])}")

# Check rotor velocity after step
controls = env.sim.data.controls
print("\n=== Controls After Step ===")
print(f"Rotor vel cmd: {np.array(controls.rotor_vel[0, 0]) if controls.rotor_vel is not None else 'None'}")
if controls.force_torque is not None:
    print(f"Force torque cmd: {np.array(controls.force_torque.cmd[0, 0])}")

env.close()
