#!/usr/bin/env python3
"""Detailed profiling of _apply_controls."""

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
from crazyflie_mape_crazyflow.pursuit import proportional_nav, pure_pursuit
from crazyflie_mape_crazyflow.utils import accel_to_attitude


def main():
    env_cfg = RedVsBlueEnvConfig(
        n_pairs=2,
        n_worlds=256,
        pursuer_strategy="ProNav",
        device="cpu",
    )
    env = RedVsBlueEnv(cfg=env_cfg)
    env.reset()

    B = env.cfg.n_blue
    R = env.cfg.n_red
    N = env.cfg.n_worlds

    # Process dummy actions
    dummy_actions = {agent: np.zeros((N, 4), dtype=np.float32) for agent in env.possible_agents}
    env._process_blue_actions(dummy_actions)

    # Get states
    states = env.sim.data.states
    blue_pos = states.pos[:, :B]
    blue_vel = states.vel[:, :B]
    red_pos = states.pos[:, B:]
    red_vel = states.vel[:, B:]
    red_quat = states.quat[:, B:]

    # Timing functions
    n_iter = 100

    # 1. Profile target gathering
    target_idx = env.red_target
    t0 = time.perf_counter()
    for _ in range(n_iter):
        target_pos = jnp.take_along_axis(blue_pos, target_idx[:, :, None].astype(jnp.int32), axis=1)
        target_vel = jnp.take_along_axis(blue_vel, target_idx[:, :, None].astype(jnp.int32), axis=1)
    jax.block_until_ready(target_vel)
    t_gather = (time.perf_counter() - t0) / n_iter * 1000
    print(f"Target gathering: {t_gather:.3f} ms")

    # Relative state
    pos_rb = target_pos - red_pos
    vel_rb = target_vel - red_vel

    # 2. Profile pure pursuit
    t0 = time.perf_counter()
    for _ in range(n_iter):
        accel_pp = pure_pursuit(pos_rb, vel_rb, gravity=env.cfg.gravity)
    jax.block_until_ready(accel_pp)
    t_pp = (time.perf_counter() - t0) / n_iter * 1000
    print(f"Pure pursuit: {t_pp:.3f} ms")

    # 3. Profile proportional nav
    target_accel = env.prev_blue_accel
    accel_target = jnp.take_along_axis(target_accel, target_idx[:, :, None].astype(jnp.int32), axis=1)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        accel_pronav = proportional_nav(
            pos_rb, vel_rb, accel_target,
            N_fb=env.cfg.N_pronav_fb,
            N_ff=env.cfg.N_pronav_ff,
            velocity_closure_threshold=env.cfg.velocity_closure_threshold,
            gravity=env.cfg.gravity,
        )
    jax.block_until_ready(accel_pronav)
    t_pronav = (time.perf_counter() - t0) / n_iter * 1000
    print(f"Proportional nav: {t_pronav:.3f} ms")

    # 4. Profile accel_to_attitude
    accel = accel_pronav
    t0 = time.perf_counter()
    for _ in range(n_iter):
        rpy_des, thrust_des = accel_to_attitude(accel, red_quat, mass=env.cfg.mass)
    jax.block_until_ready(thrust_des)
    t_accel_to_att = (time.perf_counter() - t0) / n_iter * 1000
    print(f"accel_to_attitude: {t_accel_to_att:.3f} ms")

    # 5. Profile clipping and concatenation
    t0 = time.perf_counter()
    for _ in range(n_iter):
        rpy_clipped = jnp.clip(rpy_des, -0.5, 0.5)
        thrust_clipped = jnp.clip(thrust_des, env.cfg.min_thrust, env.cfg.max_thrust)
        red_cmd = jnp.concatenate([rpy_clipped, thrust_clipped[..., None]], axis=-1)
        red_cmd = red_cmd * env.red_alive[:, :, None]
        all_cmd = jnp.concatenate([env.blue_cmd, red_cmd], axis=1)
    jax.block_until_ready(all_cmd)
    t_clip = (time.perf_counter() - t0) / n_iter * 1000
    print(f"Clip and concat: {t_clip:.3f} ms")

    # 6. Profile attitude_control
    t0 = time.perf_counter()
    for _ in range(n_iter):
        env.sim.attitude_control(all_cmd)
    t_att_ctrl = (time.perf_counter() - t0) / n_iter * 1000
    print(f"attitude_control: {t_att_ctrl:.3f} ms")

    # Total
    total = t_gather + t_pronav + t_accel_to_att + t_clip + t_att_ctrl
    print(f"\nTotal estimated: {total:.3f} ms")

    # Now profile the full _apply_controls
    print("\n--- Full _apply_controls ---")
    t0 = time.perf_counter()
    for _ in range(n_iter):
        env._apply_controls()
    t_full = (time.perf_counter() - t0) / n_iter * 1000
    print(f"Full _apply_controls: {t_full:.3f} ms")

    env.close()


if __name__ == "__main__":
    main()
