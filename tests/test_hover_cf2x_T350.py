"""Test script to hover the cf2x_T350 drone and report rotor velocity and PWM signals.

This script:
1. Initializes crazyflow with the cf2x_T350 drone model
2. Makes the drone hover at 1m using state control (position hold)
3. Reports rotor velocity (RPM) and estimated PWM signals at steady-state hover
"""

import numpy as np
import jax.numpy as jnp
from crazyflow.sim import Sim
from crazyflow.sim.physics import Physics
from crazyflow.control.control import Control
from drone_models.core import load_params


def thrust_to_pwm(thrust: np.ndarray, thrust_max: float, pwm_max: int = 65535) -> np.ndarray:
    """Convert thrust to PWM signal using linear mapping (same as Mellinger controller).

    pwm = (thrust / thrust_max) * pwm_max
    """
    pwm = (thrust / thrust_max) * pwm_max
    return np.clip(pwm, 0, pwm_max)


def main():
    # Configuration
    drone_model = "cf2x_T350"
    n_worlds = 1
    n_drones = 1
    sim_freq = 500  # Hz
    state_freq = 100  # Hz for state controller
    attitude_freq = 500  # Hz for attitude controller
    hover_duration = 3.0  # seconds
    target_altitude = 1.0  # meters

    # Load drone parameters
    so_rpy_params = load_params("so_rpy", drone_model)
    fp_params = load_params("first_principles", drone_model)

    mass = float(so_rpy_params["mass"])
    gravity = float(np.abs(so_rpy_params["gravity_vec"][2]))

    # Parameters for RPM <-> thrust conversion
    rpm2thrust = np.array(fp_params["rpm2thrust"])  # [c0, c1, c2] polynomial coefficients
    thrust_max = float(so_rpy_params["thrust_max"])  # per motor max thrust (N)

    print("=" * 60)
    print(f"Hovering {drone_model} Drone Test")
    print("=" * 60)
    print(f"\nDrone Parameters:")
    print(f"  Mass: {mass * 1000:.2f} g")
    print(f"  Gravity: {gravity:.2f} m/s²")
    print(f"\nSimulation Settings:")
    print(f"  Physics freq: {sim_freq} Hz")
    print(f"  State control freq: {state_freq} Hz")
    print(f"  Attitude control freq: {attitude_freq} Hz")

    # Calculate theoretical hover values
    hover_thrust_total = mass * gravity  # Total thrust needed (N)
    thrust_per_motor = hover_thrust_total / 4  # Equal thrust on all 4 motors

    # Solve RPM from thrust using quadratic: c2*rpm^2 + c1*rpm + (c0 - thrust) = 0
    c0, c1, c2 = rpm2thrust
    a = c2
    b = c1
    c = c0 - thrust_per_motor
    hover_rpm = float((-b + np.sqrt(b**2 - 4*a*c)) / (2*a))

    # Calculate theoretical PWM (linear: thrust/thrust_max * pwm_max)
    hover_pwm = thrust_to_pwm(np.array([thrust_per_motor]), thrust_max)[0]

    print(f"\nTheoretical Hover Values (per motor):")
    print(f"  Thrust: {thrust_per_motor * 1000:.3f} mN")
    print(f"  RPM: {hover_rpm:.1f}")
    print(f"  PWM: {hover_pwm:.0f}")

    # Create simulator with state control (position hold)
    print(f"\nInitializing simulator...")
    sim = Sim(
        n_worlds=n_worlds,
        n_drones=n_drones,
        drone_model=drone_model,
        physics=Physics.first_principles,
        control=Control.state,  # Position/state control mode
        freq=sim_freq,
        state_freq=state_freq,
        attitude_freq=attitude_freq,
        device="cpu",
    )

    # Reset simulator
    sim.reset()

    # Set initial position at target altitude with hover rotor velocity
    initial_pos = jnp.array([[[0.0, 0.0, target_altitude]]])
    hover_rotor_vel = jnp.array([[[hover_rpm, hover_rpm, hover_rpm, hover_rpm]]])

    sim.data = sim.data.replace(
        states=sim.data.states.replace(
            pos=initial_pos,
            rotor_vel=hover_rotor_vel
        )
    )

    # State control command: [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]
    # For hover at 1m: position=(0,0,1), velocity=(0,0,0), accel=(0,0,0), yaw=0, rates=0
    hover_state_cmd = jnp.array([[[
        0.0, 0.0, target_altitude,  # x, y, z
        0.0, 0.0, 0.0,              # vx, vy, vz
        0.0, 0.0, 0.0,              # ax, ay, az
        0.0,                         # yaw
        0.0, 0.0, 0.0               # roll_rate, pitch_rate, yaw_rate
    ]]])

    initial_z = float(sim.data.states.pos[0, 0, 2])
    print(f"\nInitial drone position z: {initial_z:.4f} m")
    print(f"Target hover altitude: {target_altitude:.2f} m")

    print(f"\nRunning hover simulation for {hover_duration}s...")
    print("-" * 60)

    n_steps = int(hover_duration * sim_freq)

    # Storage for steady-state analysis
    rotor_vel_history = []
    pos_history = []

    print_interval = int(0.5 * sim_freq)  # Print every 0.5 seconds

    for step in range(n_steps):
        # Apply state control command (hover at target altitude)
        sim.state_control(hover_state_cmd)

        # Step physics
        sim.step(n_steps=1)

        # Get current state (rotor_vel is in RPM for first_principles)
        pos = np.array(sim.data.states.pos[0, 0])
        vel = np.array(sim.data.states.vel[0, 0])
        rotor_vel_rpm = np.array(sim.data.states.rotor_vel[0, 0])

        # Store for analysis (every 10th step to reduce memory)
        if step % 10 == 0:
            rotor_vel_history.append(rotor_vel_rpm.copy())
            pos_history.append(pos.copy())

        # Print progress every 0.5 seconds
        t = (step + 1) / sim_freq
        if step % print_interval == 0 or step == n_steps - 1:
            # Calculate thrust from RPM, then PWM from thrust
            thrust_from_vel = c0 + c1 * rotor_vel_rpm + c2 * rotor_vel_rpm**2
            pwm = thrust_to_pwm(thrust_from_vel, thrust_max)
            print(f"  t={t:.2f}s: z={pos[2]:.4f}m, vel_z={vel[2]:.4f}m/s")
            print(f"           rotor_vel: {rotor_vel_rpm.mean():.1f} RPM")
            print(f"           PWM: {pwm.mean():.0f} (per motor)")

    # Analyze steady-state (last 1 second)
    print("\n" + "=" * 60)
    print("STEADY-STATE HOVER RESULTS (last 1 second average)")
    print("=" * 60)

    steady_state_samples = int(1.0 * sim_freq / 10)  # Samples in last 1 second (every 10th stored)
    rotor_vel_steady = np.array(rotor_vel_history[-steady_state_samples:])
    pos_steady = np.array(pos_history[-steady_state_samples:])

    # Average rotor velocity (in RPM)
    mean_rotor_vel_rpm = rotor_vel_steady.mean(axis=0)
    std_rotor_vel_rpm = rotor_vel_steady.std(axis=0)

    # Calculate thrust from RPM
    thrust_from_rpm = c0 + c1 * mean_rotor_vel_rpm + c2 * mean_rotor_vel_rpm**2

    # Calculate PWM from thrust (linear mapping)
    mean_pwm = thrust_to_pwm(thrust_from_rpm, thrust_max)

    print(f"\nPosition (z):")
    print(f"  Mean: {pos_steady[:, 2].mean():.4f} m")
    print(f"  Std:  {pos_steady[:, 2].std():.6f} m")

    print(f"\nRotor Velocity (all 4 motors):")
    print(f"  Motor 0: {mean_rotor_vel_rpm[0]:.1f} ± {std_rotor_vel_rpm[0]:.2f} RPM")
    print(f"  Motor 1: {mean_rotor_vel_rpm[1]:.1f} ± {std_rotor_vel_rpm[1]:.2f} RPM")
    print(f"  Motor 2: {mean_rotor_vel_rpm[2]:.1f} ± {std_rotor_vel_rpm[2]:.2f} RPM")
    print(f"  Motor 3: {mean_rotor_vel_rpm[3]:.1f} ± {std_rotor_vel_rpm[3]:.2f} RPM")
    print(f"  Average: {mean_rotor_vel_rpm.mean():.1f} RPM")

    print(f"\nPWM Signals (all 4 motors):")
    print(f"  Motor 0: {mean_pwm[0]:.0f}")
    print(f"  Motor 1: {mean_pwm[1]:.0f}")
    print(f"  Motor 2: {mean_pwm[2]:.0f}")
    print(f"  Motor 3: {mean_pwm[3]:.0f}")
    print(f"  Average: {mean_pwm.mean():.0f}")

    print(f"\nThrust (calculated from RPM):")
    print(f"  Per motor: {thrust_from_rpm.mean() * 1000:.3f} mN")
    print(f"  Total:     {thrust_from_rpm.sum() * 1000:.3f} mN")
    print(f"  Weight:    {mass * gravity * 1000:.3f} mN")

    print(f"\nComparison with Theoretical Values:")
    rpm_error = mean_rotor_vel_rpm.mean() - hover_rpm
    pwm_error = mean_pwm.mean() - hover_pwm
    print(f"  RPM error:   {rpm_error:.1f} RPM ({(rpm_error / hover_rpm) * 100:.2f}%)")
    print(f"  PWM error:   {pwm_error:.0f} ({(pwm_error / hover_pwm) * 100:.2f}%)")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    return {
        "rotor_vel_rpm": mean_rotor_vel_rpm,
        "pwm": mean_pwm,
        "thrust_per_motor": thrust_from_rpm,
        "hover_rpm": hover_rpm,
        "hover_pwm": hover_pwm,
    }


if __name__ == "__main__":
    results = main()
