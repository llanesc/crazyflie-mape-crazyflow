#!/usr/bin/env python3
"""System identification for the so_rpy drone model using rosbag data.

Extracts commanded RPY + thrust and measured states from a rosbag, then runs
the drone-models identification pipeline for each drone (blue1, blue2, red1, red2).

Only uses data during MAPE_STATUS_RUNNING and before each drone crashes/deactivates.

Requires: ROS2 workspace to be sourced.

Usage:
    source ros2_ws/install/setup.bash
    python sysid_from_rosbag.py <bag_path>
"""

import os
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from rosbags.rosbag2 import Reader
from rclpy.serialization import deserialize_message

from multiagent_pursuit_evasion_interfaces.msg import Status
from crazyflie_interfaces.msg import AttitudeSetpoint
from nav_msgs.msg import Odometry

# Drone names in order
BLUE_NAMES = ["blue_1", "blue_2"]
RED_NAMES = ["red_1", "red_2"]
ALL_NAMES = BLUE_NAMES + RED_NAMES

STATUS_TOPIC = "/multiagent_pursuit_evasion/status"

# cf2x_T350 parameters for PWM -> thrust conversion
THRUST_MAX_HW = 0.18 * 4  # per motor * 4
PWM_MAX = 65535
MASS = 0.0406


def pwm_to_thrust(pwm: int) -> float:
    """Convert PWM value back to collective thrust in Newtons."""
    return (pwm / PWM_MAX) * THRUST_MAX_HW


def extract_data_from_bag(bag_path: Path) -> dict:
    """Extract time-aligned command and state data for each drone.

    Returns dict mapping drone name -> {
        'time': array, 'pos': array, 'vel': array, 'quat': array,
        'cmd_rpy': array, 'cmd_f': array, 'rpy': array
    }
    """
    # Collect raw timestamped data per topic
    cmd_data = {name: [] for name in ALL_NAMES}  # (timestamp_ns, roll, pitch, yaw_rate, thrust_pwm)
    odom_data = {name: [] for name in ALL_NAMES}  # (timestamp_ns, pos, vel, quat)

    # Status timestamps for RUNNING window
    running_start = None
    running_end = None

    with Reader(bag_path) as reader:
        # Build topic -> connection map
        topic_map = {}
        for c in reader.connections:
            topic_map.setdefault(c.topic, []).append(c)

        # First pass: find RUNNING time window from status
        status_conns = topic_map.get(STATUS_TOPIC, [])
        if not status_conns:
            raise ValueError(f"No {STATUS_TOPIC} topic in bag")

        for _, timestamp, rawdata in reader.messages(connections=status_conns):
            msg = deserialize_message(rawdata, Status)
            if msg.status == Status.MAPE_STATUS_RUNNING:
                if running_start is None:
                    running_start = timestamp
                running_end = timestamp
            elif msg.status in (Status.MAPE_STATUS_BLUE_WON, Status.MAPE_STATUS_RED_WON):
                running_end = timestamp
                break

    if running_start is None:
        raise ValueError("No RUNNING status found in bag")

    print(f"RUNNING window: {(running_end - running_start) / 1e9:.2f}s")

    with Reader(bag_path) as reader:
        topic_map = {}
        for c in reader.connections:
            topic_map.setdefault(c.topic, []).append(c)

        # Extract cmd_attitude for all drones
        for name in ALL_NAMES:
            topic = f"/{name}/cmd_attitude"
            conns = topic_map.get(topic, [])
            for _, timestamp, rawdata in reader.messages(connections=conns):
                if timestamp < running_start or timestamp > running_end:
                    continue
                msg = deserialize_message(rawdata, AttitudeSetpoint)
                cmd_data[name].append((
                    timestamp,
                    float(msg.roll),    # radians
                    float(msg.pitch),   # radians
                    float(msg.yaw_rate),  # radians (treated as yaw command)
                    int(msg.thrust),    # PWM
                ))

        # Extract odom for all drones
        for name in ALL_NAMES:
            topic = f"/{name}/odom"
            conns = topic_map.get(topic, [])
            for _, timestamp, rawdata in reader.messages(connections=conns):
                if timestamp < running_start or timestamp > running_end:
                    continue
                msg = deserialize_message(rawdata, Odometry)
                pos = np.array([
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ])
                vel = np.array([
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z,
                ])
                quat = np.array([
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ])
                odom_data[name].append((timestamp, pos, vel, quat))

    # Also get per-drone active status to trim crashed drones
    active_status = {name: [] for name in ALL_NAMES}
    with Reader(bag_path) as reader:
        topic_map = {}
        for c in reader.connections:
            topic_map.setdefault(c.topic, []).append(c)
        status_conns = topic_map.get(STATUS_TOPIC, [])
        for _, timestamp, rawdata in reader.messages(connections=status_conns):
            if timestamp < running_start or timestamp > running_end:
                continue
            msg = deserialize_message(rawdata, Status)
            if msg.status != Status.MAPE_STATUS_RUNNING:
                continue
            # Blue drones
            for i, name in enumerate(BLUE_NAMES):
                if i < len(msg.cf_evader_states):
                    active_status[name].append((timestamp, msg.cf_evader_states[i].active))
            # Red drones
            for i, name in enumerate(RED_NAMES):
                if i < len(msg.cf_pursuer_states):
                    active_status[name].append((timestamp, msg.cf_pursuer_states[i].active))

    # Build aligned data for each drone
    result = {}
    for name in ALL_NAMES:
        if not cmd_data[name] or not odom_data[name]:
            print(f"  {name}: No data, skipping")
            continue

        # Find when this drone deactivates (crashed)
        crash_time = running_end
        for ts, active in active_status[name]:
            if not active:
                crash_time = ts
                break

        # Filter to before crash
        cmds = [(t, r, p, y, th) for t, r, p, y, th in cmd_data[name] if t <= crash_time]
        odoms = [(t, pos, vel, q) for t, pos, vel, q in odom_data[name] if t <= crash_time]

        if len(cmds) < 10 or len(odoms) < 10:
            print(f"  {name}: Too few samples (cmd={len(cmds)}, odom={len(odoms)}), skipping")
            continue

        # Convert to arrays
        cmd_times = np.array([t for t, *_ in cmds])
        cmd_roll = np.array([r for _, r, *_ in cmds])
        cmd_pitch = np.array([p for _, _, p, *_ in cmds])
        cmd_yaw = np.array([y for _, _, _, y, _ in cmds])
        cmd_thrust_pwm = np.array([th for _, _, _, _, th in cmds])
        cmd_thrust_N = np.array([pwm_to_thrust(th) for _, _, _, _, th in cmds])

        odom_times = np.array([t for t, *_ in odoms])
        positions = np.array([pos for _, pos, _, _ in odoms])
        velocities = np.array([vel for _, _, vel, _ in odoms])
        quats = np.array([q for _, _, _, q in odoms])

        # Interpolate commands onto odom timestamps (odom is the primary timeline)
        from scipy.interpolate import interp1d
        t0 = odom_times[0]
        odom_t_sec = (odom_times - t0) / 1e9
        cmd_t_sec = (cmd_times - t0) / 1e9

        # Only use overlapping time range
        t_start = max(odom_t_sec[0], cmd_t_sec[0])
        t_end = min(odom_t_sec[-1], cmd_t_sec[-1])

        odom_mask = (odom_t_sec >= t_start) & (odom_t_sec <= t_end)
        odom_t_sec = odom_t_sec[odom_mask]
        positions = positions[odom_mask]
        velocities = velocities[odom_mask]
        quats = quats[odom_mask]

        if len(odom_t_sec) < 10:
            print(f"  {name}: Too few overlapping samples, skipping")
            continue

        # Interpolate commands to odom times
        interp_roll = interp1d(cmd_t_sec, cmd_roll, fill_value="extrapolate")(odom_t_sec)
        interp_pitch = interp1d(cmd_t_sec, cmd_pitch, fill_value="extrapolate")(odom_t_sec)
        interp_yaw = interp1d(cmd_t_sec, cmd_yaw, fill_value="extrapolate")(odom_t_sec)
        interp_thrust = interp1d(cmd_t_sec, cmd_thrust_N, fill_value="extrapolate")(odom_t_sec)

        cmd_rpy = np.stack([interp_roll, interp_pitch, interp_yaw], axis=-1)
        rpy = R.from_quat(quats).as_euler('xyz')

        duration = odom_t_sec[-1] - odom_t_sec[0]
        print(f"  {name}: {len(odom_t_sec)} samples, {duration:.2f}s (before crash/end)")

        result[name] = {
            'time': odom_t_sec,
            'pos': positions,
            'vel': velocities,
            'quat': quats,
            'rpy': rpy,
            'cmd_rpy': cmd_rpy,
            'cmd_f': interp_thrust,
        }

    return result


def svf_filter(y, t, f_c=6.0, N_deriv=2):
    """State Variable Filter for smooth derivatives (from drone_models)."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d
    from scipy.signal import bilinear, butter, lfilter, lfiltic

    if y.ndim == 1:
        y = y[None, :]
    batch_size, signal_length = y.shape

    N_ord = N_deriv + 2
    omega_c = 2 * np.pi * f_c
    f_s = 1 / np.mean(np.diff(t))

    b, a = butter(N=N_ord, Wn=omega_c, analog=True)
    b_dig, a_dig = bilinear(b, a, fs=f_s)
    a_flipped = np.flip(a)

    def _f(t_val, x, u):
        x_dot = []
        x_dot_last = 0
        for i in np.arange(1, N_ord):
            x_dot.append(x[i])
        for i in np.arange(0, N_ord):
            x_dot_last -= a_flipped[i] * x[i]
        x_dot_last += b[0] * u(t_val)
        x_dot.append(x_dot_last)
        return x_dot

    results = np.zeros((batch_size, N_deriv + 1, signal_length))

    for i in range(batch_size):
        pad = 100
        y_backwards = np.flip(y[i], axis=-1)
        y_backwards_padded = np.concatenate([np.ones(pad) * y_backwards[0], y_backwards])
        zi = lfiltic(b_dig, a_dig, y_backwards_padded, x=y_backwards_padded)
        y_backwards_filt, _ = lfilter(b_dig, a_dig, y_backwards_padded, axis=-1, zi=zi)
        u = interp1d(t, np.flip(y_backwards_filt[pad:], axis=-1), kind="linear",
                      fill_value="extrapolate")
        x0 = np.zeros(N_ord)
        x0[0] = y[i, 0]
        sol = solve_ivp(_f, [t[0], t[-1]], x0, t_eval=t, args=(u,))
        results[i] = sol.y[:-1]

    return results.squeeze()


def compute_svf_derivatives(data: dict) -> dict:
    """Compute SVF-filtered states and their derivatives."""
    t = data['time']

    # Position -> velocity -> acceleration
    svf_lin = svf_filter(data['pos'].T, t, f_c=6, N_deriv=3)
    data['SVF_pos'] = svf_lin[:, 0].T
    data['SVF_vel'] = svf_lin[:, 1].T
    data['SVF_acc'] = svf_lin[:, 2].T

    # RPY -> rpy_rates -> rpy_acc
    svf_rot = svf_filter(data['rpy'].T, t, f_c=8, N_deriv=3)
    data['SVF_rpy'] = svf_rot[:, 0].T
    data['SVF_drpy'] = svf_rot[:, 1].T
    data['SVF_ddrpy'] = svf_rot[:, 2].T

    # SVF quaternion from filtered RPY
    data['SVF_quat'] = R.from_euler('xyz', data['SVF_rpy']).as_quat()

    # Command thrust (SVF filtered)
    svf_cmd_f = svf_filter(data['cmd_f'], t, f_c=6, N_deriv=1)
    data['SVF_cmd_f'] = svf_cmd_f[0]

    # Command RPY (SVF filtered)
    svf_cmd_rpy = svf_filter(data['cmd_rpy'].T, t, f_c=8, N_deriv=1)
    data['SVF_cmd_rpy'] = svf_cmd_rpy[:, 0].T

    return data


def sysid_translation_lsq(data: dict, mass: float, gravity: np.ndarray) -> dict:
    """Identify cmd_f_coef using linear least squares.

    Model: vel_dot = (1/mass) * (acc_coef + cmd_f_coef * cmd_f) * z_axis + gravity
    For so_rpy, acc_coef = 0, so: vel_dot - gravity = (cmd_f_coef * cmd_f / mass) * z_axis

    Project onto z_axis: (vel_dot - gravity) . z_axis = cmd_f_coef * cmd_f / mass
    => acc_z_body = cmd_f_coef * cmd_f / mass
    => cmd_f_coef = mass * acc_z_body / cmd_f  (least squares)
    """
    acc = data['SVF_acc']  # (N, 3)
    cmd_f = data['SVF_cmd_f']  # (N,)
    quat = data['SVF_quat']  # (N, 4)

    # Body z-axis in world frame
    z_axis = R.from_quat(quat).as_matrix()[..., :, 2]  # (N, 3)

    # Acceleration minus gravity, projected onto body z
    acc_minus_g = acc - gravity[None, :]
    acc_z_body = np.sum(acc_minus_g * z_axis, axis=-1)  # (N,)

    # Linear model: acc_z_body = (cmd_f_coef / mass) * cmd_f
    # Least squares: A * x = b where A = cmd_f, x = cmd_f_coef/mass, b = acc_z_body
    A = cmd_f[:, None]
    b = acc_z_body
    x, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cmd_f_coef = x[0] * mass

    # Also try with acc_coef: acc_z_body = (acc_coef + cmd_f_coef * cmd_f) / mass
    A2 = np.column_stack([np.ones_like(cmd_f), cmd_f])
    x2, _, _, _ = np.linalg.lstsq(A2, acc_z_body, rcond=None)
    acc_coef_full = x2[0] * mass
    cmd_f_coef_full = x2[1] * mass

    # Compute fit quality
    acc_z_pred = cmd_f_coef / mass * cmd_f
    rmse = np.sqrt(np.mean((acc_z_body - acc_z_pred)**2))
    ss_res = np.sum((acc_z_body - acc_z_pred)**2)
    ss_tot = np.sum((acc_z_body - np.mean(acc_z_body))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Full 3D RMSE
    acc_pred_3d = (cmd_f_coef / mass * cmd_f)[:, None] * z_axis + gravity[None, :]
    rmse_3d = np.sqrt(np.mean((acc - acc_pred_3d)**2))

    print(f"  cmd_f_coef = {cmd_f_coef:.6f}")
    print(f"  (with acc_coef: acc_coef={acc_coef_full:.6f}, cmd_f_coef={cmd_f_coef_full:.6f})")
    print(f"  Z-axis RMSE = {rmse:.4f} m/s², R² = {r2:.4f}")
    print(f"  3D acc RMSE = {rmse_3d:.4f} m/s²")

    return {'cmd_f_coef': cmd_f_coef, 'acc_coef': acc_coef_full, 'cmd_f_coef_full': cmd_f_coef_full}


def sysid_rotation_lsq(data: dict) -> dict:
    """Identify rpy_coef, rpy_rates_coef, cmd_rpy_coef using linear least squares.

    Model: rpy_ddot = rpy_coef * rpy + rpy_rates_coef * rpy_dot + cmd_rpy_coef * cmd_rpy
    This is linear in the 3 coefficients per axis (roll/pitch share, yaw separate).
    """
    rpy = data['SVF_rpy']        # (N, 3)
    drpy = data['SVF_drpy']      # (N, 3)
    ddrpy = data['SVF_ddrpy']    # (N, 3)
    cmd_rpy = data['SVF_cmd_rpy']  # (N, 3)

    # For each axis: ddrpy_i = a_i * rpy_i + b_i * drpy_i + c_i * cmd_rpy_i
    results = {}
    axis_names = ['roll', 'pitch', 'yaw']
    coefs = np.zeros((3, 3))  # [axis, (rpy_coef, rpy_rates_coef, cmd_rpy_coef)]

    for i, axis in enumerate(axis_names):
        A = np.column_stack([rpy[:, i], drpy[:, i], cmd_rpy[:, i]])
        b = ddrpy[:, i]
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        coefs[i] = x

        # Fit quality
        pred = A @ x
        rmse = np.sqrt(np.mean((b - pred)**2))
        ss_res = np.sum((b - pred)**2)
        ss_tot = np.sum((b - np.mean(b))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  {axis}: rpy_coef={x[0]:.2f}, rpy_rates_coef={x[1]:.2f}, "
              f"cmd_rpy_coef={x[2]:.2f}  (RMSE={rmse:.4f}, R²={r2:.4f})")

    rpy_coef = coefs[:, 0]
    rpy_rates_coef = coefs[:, 1]
    cmd_rpy_coef = coefs[:, 2]

    return {
        'rpy_coef': rpy_coef,
        'rpy_rates_coef': rpy_rates_coef,
        'cmd_rpy_coef': cmd_rpy_coef,
    }


def run_sysid(drone_data: dict, mass: float = MASS):
    """Run system identification for each drone."""
    gravity = np.array([0.0, 0.0, -9.81])

    results = {}
    for name, data in drone_data.items():
        print(f"\n{'='*60}")
        print(f"  System Identification: {name}")
        print(f"{'='*60}")

        # Compute SVF derivatives
        data = compute_svf_derivatives(data)

        # Translation sysid (cmd_f_coef)
        print("\n--- Translation ---")
        try:
            trans_params = sysid_translation_lsq(data, mass, gravity)
        except Exception as e:
            print(f"  Translation sysid failed: {e}")
            trans_params = {}

        # Rotation sysid
        print("\n--- Rotation ---")
        try:
            rot_params = sysid_rotation_lsq(data)
        except Exception as e:
            print(f"  Rotation sysid failed: {e}")
            rot_params = {}

        results[name] = {**trans_params, **rot_params}

    return results


def main():
    parser = argparse.ArgumentParser(description='SysID from rosbag for so_rpy model')
    parser.add_argument('bag_path', type=str, help='Path to the ROS2 bag directory')
    parser.add_argument('--mass', type=float, default=MASS, help=f'Drone mass in kg (default: {MASS})')
    args = parser.parse_args()

    bag_path = Path(args.bag_path)
    if not bag_path.exists():
        print(f"Error: Bag path not found: {bag_path}")
        return 1

    print(f"Bag: {bag_path}")
    print(f"Mass: {args.mass} kg")
    print(f"\nExtracting data...")
    drone_data = extract_data_from_bag(bag_path)

    if not drone_data:
        print("No valid drone data extracted")
        return 1

    print(f"\nRunning system identification...")
    results = run_sysid(drone_data, mass=args.mass)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, params in results.items():
        print(f"\n{name}:")
        for key, val in params.items():
            if isinstance(val, np.ndarray):
                print(f"  {key}: {val}")
            else:
                print(f"  {key}: {val:.6f}")

    # Compare with current cf2x_T350 params
    print(f"\n{'='*60}")
    print("  Current cf2x_T350 reference values:")
    print(f"{'='*60}")
    from crazyflow.sim.physics import load_params
    ref = load_params("so_rpy", "cf2x_T350")
    for key in ['cmd_f_coef', 'rpy_coef', 'rpy_rates_coef', 'cmd_rpy_coef']:
        if key in ref:
            print(f"  {key}: {ref[key]}")

    return 0


if __name__ == '__main__':
    exit(main())
