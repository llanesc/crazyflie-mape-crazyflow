#!/usr/bin/env python3
"""Collect thrust pipeline data from a CrazySim drone via cflib logging.

Records all pipeline layers during a hover:
  1. State estimate (pos, vel, quat)
  2. Controller output (thrust, yaw)
  3. Motor PWMs (m1-m4 uncapped)

Omega and forces are computed offline from PWM since PWM2OMEGA is deterministic.

Flight sequence:
  1. Send zero setpoint to activate low-level commander
  2. Stream hover thrust via send_setpoint while logging all pipeline data
  3. Stop motors

Usage:
    # Start CrazySim first, then:
    python3 scripts/crazysim_thrust_pipeline_logger.py --uri udp://localhost:19850 --duration 10
"""

import argparse
import logging
import time
from pathlib import Path
from collections import defaultdict
from threading import Thread

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig


# CrazySim thrust upgrade parameters (from CrtpUtils.h and SDF)
THRUST_MAX_PER_MOTOR = 0.18  # N
MOTOR_CONSTANT = 2.3375e-8   # F = kf * omega^2
MAX_ROT_VELOCITY = 2797.0    # rad/s cap
PWM_MIN = 7000
PWM_MAX = 65535

# Flight parameters
HOVER_HEIGHT = 0.5    # m
TAKEOFF_DURATION = 2.0  # s
MASS = 0.0404          # kg (SDF mass for thrust upgrade)
GRAVITY = 9.81
HOVER_THRUST_N = MASS * GRAVITY  # total collective thrust for hover


def pwm2omega(pwm: int) -> float:
    """Replicate CrazySim's PWM2OMEGA from CrtpUtils.h."""
    if pwm < PWM_MIN:
        return 0.0
    thrust_desired = (pwm / PWM_MAX) * THRUST_MAX_PER_MOTOR
    omega = np.sqrt(thrust_desired / MOTOR_CONSTANT)
    return min(omega, MAX_ROT_VELOCITY)


def omega2force(omega: float) -> float:
    """Gazebo MulticopterMotorModel: F = motorConstant * omega^2."""
    return MOTOR_CONSTANT * omega ** 2


class PipelineLogger:
    def __init__(self, link_uri: str, duration: float, output: str, hover_height: float):
        self.duration = duration
        self.output = output
        self.hover_height = hover_height
        self.data = defaultdict(list)
        self.start_time = None
        self.is_connected = False
        self.logging_started = False
        self.flight_done = False

        self._cf = Crazyflie(rw_cache="./cache")
        self._cf.connected.add_callback(self._connected)
        self._cf.fully_connected.add_callback(self._fully_connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)

        print(f"Connecting to {link_uri}")
        cflib.crtp.init_drivers()
        self._cf.open_link(link_uri)

    def _connected(self, link_uri):
        print(f"Connected to {link_uri}")

    def _fully_connected(self, link_uri):
        print("Fully connected, setting up LOG blocks and starting flight...")
        self.is_connected = True
        self._setup_logging()
        Thread(target=self._flight_sequence, daemon=True).start()

    def _setup_logging(self):
        """Configure and start all LOG blocks."""
        # Block 1: Position + velocity (6 floats = 24 bytes)
        lg_state = LogConfig(name="State", period_in_ms=10)
        lg_state.add_variable("stateEstimate.x", "float")
        lg_state.add_variable("stateEstimate.y", "float")
        lg_state.add_variable("stateEstimate.z", "float")
        lg_state.add_variable("stateEstimate.vx", "float")
        lg_state.add_variable("stateEstimate.vy", "float")
        lg_state.add_variable("stateEstimate.vz", "float")

        # Block 2: Quaternion (4 floats = 16 bytes)
        lg_quat = LogConfig(name="Quat", period_in_ms=10)
        lg_quat.add_variable("stateEstimate.qx", "float")
        lg_quat.add_variable("stateEstimate.qy", "float")
        lg_quat.add_variable("stateEstimate.qz", "float")
        lg_quat.add_variable("stateEstimate.qw", "float")

        # Block 3: Controller output + motor PWMs (capped, from power_distribution_sitl)
        lg_ctrl_motor = LogConfig(name="CtrlMotor", period_in_ms=10)
        lg_ctrl_motor.add_variable("stabilizer.thrust", "float")
        lg_ctrl_motor.add_variable("controller.ctr_yaw", "int16_t")
        lg_ctrl_motor.add_variable("motor.m1", "uint16_t")
        lg_ctrl_motor.add_variable("motor.m2", "uint16_t")
        lg_ctrl_motor.add_variable("motor.m3", "uint16_t")
        lg_ctrl_motor.add_variable("motor.m4", "uint16_t")

        for lg in [lg_state, lg_quat, lg_ctrl_motor]:
            try:
                self._cf.log.add_config(lg)
                lg.data_received_cb.add_callback(self._log_data)
                lg.error_cb.add_callback(self._log_error)
                lg.start()
                print(f"  Started LOG block: {lg.name}")
            except KeyError as e:
                print(f"  Could not start {lg.name}: {e} not in TOC")
            except AttributeError:
                print(f"  Bad config for {lg.name}")

        self.logging_started = True

    def _flight_sequence(self):
        """Use send_hover_setpoint (Mellinger altitude control) for stable hover.

        send_hover_setpoint(vx, vy, yawrate, zdistance):
          - Sets mode.z = modeAbs → Mellinger position controller handles altitude
          - The controller computes thrust internally using its mass parameter
        """
        cf = self._cf

        # Fix massThrust (force-to-PWM scaling) for CrazySim's linear force model
        # CrazySim: F_motor = (pwm/65535) * 0.18, so for hover equilibrium:
        #   4 * (massThrust * mass * g / 65535) * 0.18 = mass * g
        #   massThrust = 65535 / (4 * 0.18) = 90910.4
        # Firmware default 132000 is ~45% too high for this force model
        mass_thrust_crazysim = PWM_MAX / (4 * THRUST_MAX_PER_MOTOR)
        print(f"[Flight] Setting ctrlMel.massThrust = {mass_thrust_crazysim:.1f}...")
        cf.param.set_value('ctrlMel.massThrust', f'{mass_thrust_crazysim:.1f}')
        time.sleep(0.5)
        # Verify parameter was set
        readback = cf.param.get_value('ctrlMel.massThrust')
        print(f"[Flight] ctrlMel.massThrust readback = {readback}")

        # Also fix mass to match SDF
        print(f"[Flight] Setting ctrlMel.mass = {MASS}...")
        cf.param.set_value('ctrlMel.mass', f'{MASS}')
        time.sleep(0.5)
        readback_mass = cf.param.get_value('ctrlMel.mass')
        print(f"[Flight] ctrlMel.mass readback = {readback_mass}")

        # Step 1: Send zero setpoint to unlock commander
        print("[Flight] Activating commander...")
        cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(0.1)

        # Step 2: Ramp height using position setpoint (x=0, y=0, z=ramp, yaw=0)
        ramp_steps = int(TAKEOFF_DURATION / 0.02)  # 50 Hz
        print(f"[Flight] Ramping to {self.hover_height}m over {TAKEOFF_DURATION}s...")
        for i in range(1, ramp_steps + 1):
            frac = i / ramp_steps
            z = self.hover_height * frac
            cf.commander.send_position_setpoint(0, 0, z, 0)
            time.sleep(0.02)

        # Step 3: Hold hover for data collection
        print(f"[Flight] Hovering at {self.hover_height}m for {self.duration}s...")
        t_start = time.time()
        while time.time() - t_start < self.duration:
            cf.commander.send_position_setpoint(0, 0, self.hover_height, 0)
            time.sleep(0.02)  # 50 Hz

        # Step 4: Ramp height down to land
        print(f"[Flight] Landing over {TAKEOFF_DURATION}s...")
        for i in range(ramp_steps, -1, -1):
            frac = i / ramp_steps
            z = max(self.hover_height * frac, 0.02)
            cf.commander.send_position_setpoint(0, 0, z, 0)
            time.sleep(0.02)

        # Step 5: Stop
        cf.commander.send_stop_setpoint()
        print("[Flight] Flight complete.")
        self.flight_done = True

    def _log_data(self, timestamp, data, logconf):
        t = timestamp / 1000.0  # ms -> s
        if self.start_time is None:
            self.start_time = t
        t -= self.start_time
        for key, value in data.items():
            self.data[key].append((t, value))

    def _log_error(self, logconf, msg):
        print(f"LOG error [{logconf.name}]: {msg}")

    def _connection_failed(self, link_uri, msg):
        print(f"Connection to {link_uri} failed: {msg}")
        self.is_connected = False

    def _disconnected(self, link_uri):
        print(f"Disconnected from {link_uri}")
        self.is_connected = False

    def save(self):
        """Save collected data to NPZ with offline-computed omega and forces."""
        if not self.data:
            print("No data collected!")
            return

        save_dict = {}
        for key, values in self.data.items():
            times = np.array([v[0] for v in values])
            vals = np.array([v[1] for v in values])
            safe_key = key.replace(".", "_")
            save_dict[f"{safe_key}_t"] = times
            save_dict[safe_key] = vals

        # Compute offline omega and forces from motor PWMs
        for m in ["m1", "m2", "m3", "m4"]:
            key = f"motor_{m}"
            if key in save_dict:
                pwms = save_dict[key].astype(int)
                omegas = np.array([pwm2omega(int(p)) for p in pwms])
                forces = np.array([omega2force(o) for o in omegas])
                save_dict[f"omega_{m}"] = omegas
                save_dict[f"force_{m}"] = forces

        out_path = Path(self.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **save_dict)
        print(f"Saved {len(save_dict)} arrays to {out_path}")

    def run(self):
        """Wait for connection, run flight, collect data, then save."""
        try:
            # Wait for flight to complete
            while not self.flight_done:
                if not self.is_connected and self.logging_started:
                    print("Lost connection!")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            # Emergency: send zero thrust
            try:
                self._cf.commander.send_setpoint(0, 0, 0, 0)
            except Exception:
                pass

        print("Stopping...")
        self._cf.close_link()
        time.sleep(0.5)
        self.save()


def main():
    parser = argparse.ArgumentParser(
        description="Collect CrazySim thrust pipeline data via cflib logging"
    )
    parser.add_argument("--uri", default="udp://localhost:19850",
                        help="Crazyflie URI (port=19850+cf_id)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Hover duration for data collection (s)")
    parser.add_argument("--height", type=float, default=HOVER_HEIGHT,
                        help="Hover height (m)")
    parser.add_argument("--output", default="hardware/logs/crazysim_pipeline.npz",
                        help="Output NPZ path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    logger = PipelineLogger(args.uri, args.duration, args.output, args.height)
    logger.run()


if __name__ == "__main__":
    main()
