"""Test script for pursuit-evasion scenario.

This script tests:
- Red pursuer using ProNav guidance + accel_to_attitude
- Blue evader using state2attitude for position control (hovering at goal)
- Crazyflow sim with attitude control and first_principles physics
- sim_freq = 1000Hz, ctrl_freq = 100Hz
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax.numpy as jnp
import numpy as np
from drone_controllers import parametrize
from drone_controllers.mellinger import state2attitude
from drone_models.core import load_params

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim

from crazyflie_mape_crazyflow.pursuit import proportional_nav
from crazyflie_mape_crazyflow.utils.accel_to_attitude import accel_to_attitude


def main():
    parser = argparse.ArgumentParser(description="Test pursuit-evasion scenario")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration [s]")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Enable visualization")
    parser.add_argument("--render-fps", type=int, default=30, help="Render FPS")
    parser.add_argument("--drone-model", type=str, default="cf2x_T350", help="Drone model")
    args = parser.parse_args()

    # Simulation parameters
    sim_freq = 1000  # Hz
    ctrl_freq = 100  # Hz
    steps_per_ctrl = sim_freq // ctrl_freq

    # Load drone parameters
    drone_params = load_params("so_rpy", args.drone_model)
    mass = float(drone_params["mass"])
    gravity = float(np.abs(drone_params["gravity_vec"][2]))
    min_thrust = float(drone_params["thrust_min"]) * 4
    max_thrust = float(drone_params["thrust_max"]) * 4

    # Load first_principles params to get rpm2thrust coefficients for hover RPM calculation
    fp_params = load_params("first_principles", args.drone_model)
    rpm2thrust = fp_params["rpm2thrust"]  # [c0, c1, c2] where thrust = c0 + c1*rpm + c2*rpm^2

    # Solve for hover RPM: rpm2thrust[2]*rpm^2 + rpm2thrust[1]*rpm + (rpm2thrust[0] - thrust_per_motor) = 0
    thrust_per_motor_hover = mass * gravity / 4
    a = rpm2thrust[2]
    b = rpm2thrust[1]
    c = rpm2thrust[0] - thrust_per_motor_hover
    # Quadratic formula (take positive root)
    hover_rpm = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

    print(f"Drone parameters:")
    print(f"  mass: {mass:.4f} kg")
    print(f"  gravity: {gravity:.2f} m/s^2")
    print(f"  thrust range: [{min_thrust:.3f}, {max_thrust:.3f}] N")
    print(f"  rpm2thrust: {rpm2thrust}")
    print(f"  hover RPM: {hover_rpm:.1f}")

    # Create sim with 2 drones: drone 0 = blue (evader), drone 1 = red (pursuer)
    sim = Sim(
        n_worlds=1,
        n_drones=2,
        drone_model=args.drone_model,
        physics=Physics.first_principles,
        control=Control.attitude,
        freq=sim_freq,
        state_freq=ctrl_freq,
        attitude_freq=ctrl_freq,
    )

    # Parametrize state2attitude controller for the evader
    evader_controller = parametrize(state2attitude, drone_model=args.drone_model)

    # Simulation loop parameters
    n_steps = int(args.duration * ctrl_freq)
    render_interval = max(1, ctrl_freq // args.render_fps) if args.render else None
    render_dt = 1.0 / args.render_fps  # Time between frames
    capture_distance = 0.3  # m

    # Track statistics across episodes
    capture_times = []
    episode_lengths = []

    print(f"\nRunning {args.episodes} episodes, {args.duration}s each ({n_steps} control steps)")

    for episode in range(args.episodes):
        # Reset sim
        sim.reset()

        # Initial positions
        # Blue evader starts at origin, red pursuer starts offset
        evader_start = np.array([0.0, 0.0, 1.0])
        pursuer_start = np.array([2.0, 2.0, 1.0])

        # Set initial positions and set initial rotor velocities for hover
        hover_rotor_vel = hover_rpm * np.ones(4)
        sim.data = sim.data.replace(
            states=sim.data.states.replace(
                pos=sim.data.states.pos.at[0, 0].set(evader_start).at[0, 1].set(pursuer_start),
                rotor_vel=sim.data.states.rotor_vel.at[0, 0].set(hover_rotor_vel).at[0, 1].set(hover_rotor_vel),
            )
        )

        # Evader goal position (randomized within [-2, 2] for x and y)
        evader_goal = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            1.5
        ])

        # Initialize integral error for evader controller
        int_pos_err = np.zeros(3)

        # Track capture
        captured = False
        distance = 0.0
        last_render_time = time.time()

        print(f"\n=== Episode {episode + 1}/{args.episodes} ===")
        print(f"  Evader goal: {evader_goal}")
        print(f"  Pursuer start: {pursuer_start}")

        for step in range(n_steps):
            t = step / ctrl_freq

            # Get current states (shape: n_worlds=1, n_drones=2)
            pos = np.array(sim.data.states.pos[0])  # (2, 3)
            vel = np.array(sim.data.states.vel[0])  # (2, 3)
            quat = np.array(sim.data.states.quat[0])  # (2, 4)
            ang_vel = np.array(sim.data.states.ang_vel[0])  # (2, 3)

            evader_pos = pos[0]
            evader_vel = vel[0]
            evader_quat = quat[0]
            evader_ang_vel = ang_vel[0]

            pursuer_pos = pos[1]
            pursuer_vel = vel[1]
            pursuer_quat = quat[1]

            # Check capture
            distance = np.linalg.norm(pursuer_pos - evader_pos)
            if distance < capture_distance and not captured:
                print(f"  CAPTURED at t={t:.2f}s, distance={distance:.3f}m")
                captured = True
                capture_times.append(t)
                episode_lengths.append(step)
                break  # End simulation on capture

            # === Blue Evader Controller (state2attitude) ===
            # Build command: [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]
            evader_cmd = np.zeros(13)
            evader_cmd[0:3] = evader_goal  # Target position
            evader_cmd[3:6] = 0.0  # Target velocity
            evader_cmd[6:9] = 0.0  # Target acceleration
            evader_cmd[9] = 0.0  # Target yaw

            evader_rpyt, int_pos_err = evader_controller(
                pos=evader_pos,
                quat=evader_quat,
                vel=evader_vel,
                ang_vel=evader_ang_vel,
                cmd=evader_cmd,
                ctrl_errors=(int_pos_err,),
                ctrl_freq=ctrl_freq,
            )
            evader_action = np.array(evader_rpyt)

            # === Red Pursuer Controller (ProNav + accel_to_attitude) ===
            # Relative position and velocity (target - pursuer)
            pos_rb = jnp.array(evader_pos - pursuer_pos)
            vel_rb = jnp.array(evader_vel - pursuer_vel)

            # Estimate target acceleration (assume zero for stationary target)
            accel_target = jnp.zeros(3)

            # Compute ProNav acceleration command
            accel_cmd = proportional_nav(
                pos_rb, vel_rb, accel_target,
                N_fb=5.0,
                N_ff=1.0,
                velocity_closure_threshold=0.5,
                gravity=gravity,
            )

            # Convert acceleration to attitude command
            rpy_des, thrust_des = accel_to_attitude(accel_cmd, pursuer_quat, mass=mass)

            # Clip thrust to valid range
            thrust_des = jnp.clip(thrust_des, min_thrust, max_thrust)

            pursuer_action = np.array([
                float(rpy_des[0]),
                float(rpy_des[1]),
                float(rpy_des[2]),
                float(thrust_des),
            ])

            # Clip roll/pitch
            max_roll_pitch = 0.3
            pursuer_action[0] = np.clip(pursuer_action[0], -max_roll_pitch, max_roll_pitch)
            pursuer_action[1] = np.clip(pursuer_action[1], -max_roll_pitch, max_roll_pitch)
            pursuer_action[2] = np.clip(pursuer_action[2], -0.3, 0.3)

            # Apply control commands
            # Shape: (n_worlds=1, n_drones=2, 4)
            cmd = np.zeros((1, 2, 4), dtype=np.float32)
            cmd[0, 0] = evader_action
            cmd[0, 1] = pursuer_action

            sim.attitude_control(cmd)
            sim.step(steps_per_ctrl)

            # Render with proper frame timing
            if args.render and render_interval and (step % render_interval) == 0:
                sim.render()
                # Sleep to maintain target FPS
                current_time = time.time()
                elapsed = current_time - last_render_time
                sleep_time = render_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_render_time = time.time()

            # Print progress
            if step % (ctrl_freq * 2) == 0:  # Every 2 seconds
                print(f"  t={t:.1f}s: dist={distance:.3f}m, evader_pos={evader_pos}, pursuer_pos={pursuer_pos}")

        # Episode ended without capture
        if not captured:
            print(f"  Episode ended without capture. Final distance: {distance:.3f}m")
            episode_lengths.append(n_steps)

    # Final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Captures: {len(capture_times)}/{args.episodes}")
    if capture_times:
        print(f"  Capture times: mean={np.mean(capture_times):.2f}s, min={np.min(capture_times):.2f}s, max={np.max(capture_times):.2f}s")
    print(f"  Episode lengths: mean={np.mean(episode_lengths):.1f} steps")

    sim.close()


if __name__ == "__main__":
    main()
