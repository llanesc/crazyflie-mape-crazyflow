#!/usr/bin/env python3
"""Plot commanded vs actual roll, pitch, yaw, and thrust from rosbag data.

4 subplots per drone:
  - Roll:   cmd_roll (angle) vs actual roll
  - Pitch:  cmd_pitch (angle) vs actual pitch
  - Yaw:    actual yaw (from odom quaternion)
  - Thrust: cmd_thrust (PWM)

Usage:
    # Extract data first
    python extract_rosbag_data.py <rosbag_path>

    # Plot from .npz
    python plot_commands.py <hw_data.npz>

    # Or directly from rosbag
    python plot_commands.py <rosbag_path>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Convert quaternions (xyzw) to roll, pitch, yaw.

    Args:
        quat: Array of shape (N, 4) with [x, y, z, w] quaternions.

    Returns:
        Array of shape (N, 3) with [roll, pitch, yaw] in radians.
    """
    rot = Rotation.from_quat(quat)  # scipy expects xyzw
    return rot.as_euler('xyz')


def load_data(path: Path) -> dict:
    """Load hardware data from .npz or extract from rosbag."""
    if path.suffix == '.npz':
        raw = np.load(path, allow_pickle=True)
        drone_names = list(raw['drone_names'])
        data = {}
        for name in drone_names:
            d = {
                'odom_t': raw[f'{name}/odom_t'],
                'pos': raw[f'{name}/pos'],
                'vel': raw[f'{name}/vel'],
                'quat': raw[f'{name}/quat'],
                'ang_vel': raw[f'{name}/ang_vel'],
                'cmd_t': raw[f'{name}/cmd_t'],
                'cmd_roll': raw[f'{name}/cmd_roll'],
                'cmd_pitch': raw[f'{name}/cmd_pitch'],
                'cmd_yaw': raw[f'{name}/cmd_yaw'],
                'cmd_thrust_pwm': raw[f'{name}/cmd_thrust_pwm'],
            }
            if f'{name}/body_rate_t' in raw:
                d['body_rate_t'] = raw[f'{name}/body_rate_t']
                d['body_rates'] = raw[f'{name}/body_rates']
            data[name] = d
        return data
    else:
        from extract_rosbag_data import extract_hw_data
        return extract_hw_data(path)


def plot_drone(name: str, d: dict, output_dir: Path = None, show: bool = True):
    """Plot 4 subplots for a single drone."""
    odom_t = d['odom_t']
    cmd_t = d['cmd_t']
    rpy = quat_to_rpy(d['quat'])
    actual_roll = rpy[:, 0]
    actual_pitch = rpy[:, 1]
    actual_yaw = rpy[:, 2]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'{name} — Commands vs Actual', fontsize=13, fontweight='bold')

    # Roll
    axes[0].plot(cmd_t, np.degrees(d['cmd_roll']), 'r-', label='Cmd Roll', linewidth=1.0, alpha=0.8)
    axes[0].plot(odom_t, np.degrees(actual_roll), 'b-', label='Actual Roll', linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel('Roll [deg]')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Pitch
    axes[1].plot(cmd_t, np.degrees(d['cmd_pitch']), 'r-', label='Cmd Pitch', linewidth=1.0, alpha=0.8)
    axes[1].plot(odom_t, np.degrees(actual_pitch), 'b-', label='Actual Pitch', linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel('Pitch [deg]')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Yaw rate: cmd yaw_rate (body rate r) vs actual body rate r from Status msg
    axes[2].plot(cmd_t, np.degrees(d['cmd_yaw']), 'r-', label='Cmd Yaw Rate', linewidth=1.0, alpha=0.8)
    if 'body_rate_t' in d and 'body_rates' in d:
        br_t = d['body_rate_t']
        actual_yaw_rate = d['body_rates'][:, 2]  # r (body yaw rate)
        axes[2].plot(br_t, np.degrees(actual_yaw_rate), 'b-', label='Actual Yaw Rate', linewidth=1.0, alpha=0.8)
    axes[2].set_ylabel('Yaw Rate [deg/s]')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Thrust
    axes[3].plot(cmd_t, d['cmd_thrust_pwm'], 'r-', label='Cmd Thrust', linewidth=1.0, alpha=0.8)
    axes[3].set_ylabel('Thrust [PWM]')
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()

    if output_dir:
        fig.savefig(output_dir / f'{name}_commands.png', dpi=150)

    if not show:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot commanded vs actual roll, pitch, yaw, thrust from rosbag data'
    )
    parser.add_argument('path', type=str, help='Path to .npz file or rosbag directory')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output directory for plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    parser.add_argument('--drones', nargs='*', default=None,
                        help='Drone names to plot (default: all)')
    args = parser.parse_args()

    path = Path(args.path)
    data = load_data(path)
    print(f"Loaded {len(data)} drones: {list(data.keys())}")

    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    drones = args.drones if args.drones else list(data.keys())
    show = not args.no_show

    for name in drones:
        if name not in data:
            print(f"Warning: {name} not found in data, skipping")
            continue
        print(f"Plotting {name}...")
        plot_drone(name, data[name], output_dir, show)

    if show:
        plt.show()

    if output_dir:
        print(f"Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
