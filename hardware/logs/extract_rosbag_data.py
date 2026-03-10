#!/usr/bin/env python3
"""Extract HW data from rosbag and save as .npz for sim replay."""

import argparse
from pathlib import Path

import numpy as np

from rosbags.rosbag2 import Reader
from rclpy.serialization import deserialize_message
from multiagent_pursuit_evasion_interfaces.msg import Status
from crazyflie_interfaces.msg import AttitudeSetpoint
from nav_msgs.msg import Odometry

BLUE_NAMES = ["blue_1", "blue_2"]
RED_NAMES = ["red_1", "red_2"]
ALL_NAMES = BLUE_NAMES + RED_NAMES
STATUS_TOPIC = "/multiagent_pursuit_evasion/status"
THRUST_MAX_HW = 0.18 * 4
PWM_MAX = 65535


def pwm_to_thrust(pwm):
    return (pwm / PWM_MAX) * THRUST_MAX_HW


def extract_hw_data(bag_path: Path) -> dict:
    """Extract per-drone state + command data from rosbag."""
    cmd_data = {name: [] for name in ALL_NAMES}
    odom_data = {name: [] for name in ALL_NAMES}
    running_start = running_end = None

    with Reader(bag_path) as reader:
        topic_map = {}
        for c in reader.connections:
            topic_map.setdefault(c.topic, []).append(c)
        for _, timestamp, rawdata in reader.messages(connections=topic_map.get(STATUS_TOPIC, [])):
            msg = deserialize_message(rawdata, Status)
            if msg.status == Status.MAPE_STATUS_RUNNING:
                if running_start is None:
                    running_start = timestamp
                running_end = timestamp
            elif msg.status in (Status.MAPE_STATUS_BLUE_WON, Status.MAPE_STATUS_RED_WON):
                running_end = timestamp
                break

    if running_start is None:
        raise ValueError("No RUNNING status")

    active_status = {name: [] for name in ALL_NAMES}
    with Reader(bag_path) as reader:
        topic_map = {}
        for c in reader.connections:
            topic_map.setdefault(c.topic, []).append(c)
        for _, timestamp, rawdata in reader.messages(connections=topic_map.get(STATUS_TOPIC, [])):
            if timestamp < running_start or timestamp > running_end:
                continue
            msg = deserialize_message(rawdata, Status)
            if msg.status != Status.MAPE_STATUS_RUNNING:
                continue
            for i, name in enumerate(BLUE_NAMES):
                if i < len(msg.cf_evader_states):
                    active_status[name].append((timestamp, msg.cf_evader_states[i].active))
            for i, name in enumerate(RED_NAMES):
                if i < len(msg.cf_pursuer_states):
                    active_status[name].append((timestamp, msg.cf_pursuer_states[i].active))

    with Reader(bag_path) as reader:
        topic_map = {}
        for c in reader.connections:
            topic_map.setdefault(c.topic, []).append(c)

        for name in ALL_NAMES:
            for _, timestamp, rawdata in reader.messages(connections=topic_map.get(f"/{name}/cmd_attitude", [])):
                if running_start <= timestamp <= running_end:
                    msg = deserialize_message(rawdata, AttitudeSetpoint)
                    cmd_data[name].append((timestamp, float(msg.roll), float(msg.pitch),
                                           float(msg.yaw_rate), int(msg.thrust)))
            for _, timestamp, rawdata in reader.messages(connections=topic_map.get(f"/{name}/odom", [])):
                if running_start <= timestamp <= running_end:
                    msg = deserialize_message(rawdata, Odometry)
                    p = msg.pose.pose.position
                    v = msg.twist.twist.linear
                    o = msg.pose.pose.orientation
                    av = msg.twist.twist.angular
                    odom_data[name].append((timestamp,
                        np.array([p.x, p.y, p.z]),
                        np.array([v.x, v.y, v.z]),
                        np.array([o.x, o.y, o.z, o.w]),
                        np.array([av.x, av.y, av.z])))

    result = {}
    for name in ALL_NAMES:
        if not cmd_data[name] or not odom_data[name]:
            continue

        crash_time = running_end
        for ts, active in active_status[name]:
            if not active:
                crash_time = ts
                break

        cmds = [c for c in cmd_data[name] if c[0] <= crash_time]
        odoms = [o for o in odom_data[name] if o[0] <= crash_time]
        if len(cmds) < 10 or len(odoms) < 10:
            continue

        t0 = odoms[0][0]
        odom_t = np.array([(o[0] - t0) / 1e9 for o in odoms])
        cmd_t = np.array([(c[0] - t0) / 1e9 for c in cmds])

        result[name] = {
            'odom_t': odom_t,
            'pos': np.array([o[1] for o in odoms]),
            'vel': np.array([o[2] for o in odoms]),
            'quat': np.array([o[3] for o in odoms]),
            'ang_vel': np.array([o[4] for o in odoms]),
            'cmd_t': cmd_t,
            'cmd_roll': np.array([c[1] for c in cmds]),
            'cmd_pitch': np.array([c[2] for c in cmds]),
            'cmd_yaw': np.array([c[3] for c in cmds]),
            'cmd_thrust_pwm': np.array([c[4] for c in cmds]),
            'cmd_thrust_N': np.array([pwm_to_thrust(c[4]) for c in cmds]),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_path', type=str)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    bag_path = Path(args.bag_path)
    hw_data = extract_hw_data(bag_path)
    print(f"Extracted {len(hw_data)} drones: {list(hw_data.keys())}")

    out_path = args.output or str(bag_path) + "_hw_data.npz"
    save_dict = {}
    save_dict['drone_names'] = np.array(list(hw_data.keys()))
    for name, data in hw_data.items():
        for key, val in data.items():
            save_dict[f"{name}/{key}"] = val

    np.savez(out_path, **save_dict)
    print(f"Saved to {out_path}")

    for name, data in hw_data.items():
        print(f"  {name}: {len(data['odom_t'])} odom, {len(data['cmd_t'])} cmd, {data['odom_t'][-1]:.2f}s")


if __name__ == '__main__':
    main()
