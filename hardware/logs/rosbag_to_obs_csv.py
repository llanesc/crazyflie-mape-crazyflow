#!/usr/bin/env python3
"""Extract observations from a ROS2 bag file to CSV format.

Reads Status messages from a bag file and extracts drone states only during
MAPE_STATUS_RUNNING. Stops when MAPE_STATUS_BLUE_WON or MAPE_STATUS_RED_WON
is detected. Computes observations in the same format as red_vs_blue_env.py.

Observation format per blue agent (n_pairs=2, obs_dim=46):
  - Own state [12]: pos(3), vel(3), rpy(3), rpy_rates(3)
  - Ally one-hot [n_blue]: one-hot encoding for which agent
  - Shared state:
    - Blue agents: pos(3), vel(3), alive(1) for each = 7*n_blue
    - Red agents: pos(3), vel(3), alive(1) for each = 7*n_red
    - Target one-hot: n_red * n_blue

Requires: ROS2 workspace to be sourced (for message type imports)

Usage:
    source ros2_ws/install/setup.bash
    python rosbag_to_obs_csv.py <bag_path> [--output <output_csv>]

Example:
    python rosbag_to_obs_csv.py rosbag2_2026_02_05-12_30_00/
    python rosbag_to_obs_csv.py rosbag2_2026_02_05-12_30_00/ --output obs_data.csv
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# ROS2 bag reading
from rosbags.rosbag2 import Reader
from rclpy.serialization import deserialize_message

# Import actual message types (requires ROS2 workspace to be sourced)
from multiagent_pursuit_evasion_interfaces.msg import Status

# Topic name for status messages
STATUS_TOPIC = '/multiagent_pursuit_evasion/status'


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to roll-pitch-yaw angles.

    Uses the same convention as red_vs_blue_env.py: 'xyz' extrinsic Euler angles.

    Args:
        quat: Quaternion array [x, y, z, w] shape (..., 4)

    Returns:
        RPY angles [roll, pitch, yaw] shape (..., 3)
    """
    return Rotation.from_quat(quat).as_euler('xyz')


def ang_vel_to_rpy_rates(ang_vel: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Convert body-frame angular velocity to Euler angle rates.

    Uses the same transformation as red_vs_blue_env.py.
    omega_body = [p, q, r] (body angular velocity)
    rpy_rates = [roll_dot, pitch_dot, yaw_dot]

    The transformation matrix is:
    [roll_dot ]   [1  sin(r)*tan(p)   cos(r)*tan(p) ] [p]
    [pitch_dot] = [0  cos(r)          -sin(r)       ] [q]
    [yaw_dot  ]   [0  sin(r)/cos(p)   cos(r)/cos(p) ] [r]

    Args:
        ang_vel: Body angular velocity [p, q, r] shape (..., 3)
        rpy: Euler angles [roll, pitch, yaw] shape (..., 3)

    Returns:
        RPY rates [roll_dot, pitch_dot, yaw_dot] shape (..., 3)
    """
    roll = rpy[..., 0]
    pitch = rpy[..., 1]

    sin_r = np.sin(roll)
    cos_r = np.cos(roll)
    tan_p = np.tan(pitch)
    cos_p = np.cos(pitch)

    # Avoid division by zero near gimbal lock (pitch = ±90°)
    cos_p = np.where(np.abs(cos_p) < 1e-6, 1e-6, cos_p)

    p = ang_vel[..., 0]
    q = ang_vel[..., 1]
    r = ang_vel[..., 2]

    roll_dot = p + sin_r * tan_p * q + cos_r * tan_p * r
    pitch_dot = cos_r * q - sin_r * r
    yaw_dot = sin_r / cos_p * q + cos_r / cos_p * r

    return np.stack([roll_dot, pitch_dot, yaw_dot], axis=-1)


def compute_observation(
    evader_states: list,
    pursuer_states: list,
    pursuer_targets: list,
    agent_idx: int,
) -> np.ndarray:
    """Compute the observation vector for a single blue agent (evader).

    Matches the format used in red_vs_blue_env.py _get_observations().

    Args:
        evader_states: List of evader state messages.
        pursuer_states: List of pursuer state messages.
        pursuer_targets: List of target indices (which evader each pursuer targets).
        agent_idx: Index of the agent for which to compute observation.

    Returns:
        Observation vector for this agent.
    """
    n_blue = len(evader_states)
    n_red = len(pursuer_states)

    # =========================================================================
    # Part 1: Own state [12 dims]
    # =========================================================================
    evader = evader_states[agent_idx]
    pos = np.array(evader.position)
    vel = np.array(evader.velocity)
    quat = np.array(evader.attitude)  # [x, y, z, w]
    ang_vel = np.array(evader.angular_velocity)

    rpy = quat_to_rpy(quat)
    rpy_rates = ang_vel_to_rpy_rates(ang_vel, rpy)

    own_state = np.concatenate([pos, vel, rpy, rpy_rates])  # 12 dims

    # =========================================================================
    # Part 2: Ally one-hot [n_blue dims]
    # =========================================================================
    ally_one_hot = np.zeros(n_blue)
    ally_one_hot[agent_idx] = float(evader.active)  # Masked by alive status

    # =========================================================================
    # Part 3: Shared state
    # =========================================================================
    # Blue agents: pos(3), vel(3), alive(1) for each = 7*n_blue
    blue_shared = []
    for e in evader_states:
        blue_shared.extend([
            *e.position,
            *e.velocity,
            float(e.active),
        ])
    blue_shared = np.array(blue_shared)

    # Red agents: pos(3), vel(3), alive(1) for each = 7*n_red
    red_shared = []
    for p in pursuer_states:
        red_shared.extend([
            *p.position,
            *p.velocity,
            float(p.active),
        ])
    red_shared = np.array(red_shared)

    # Target one-hot: n_red * n_blue
    # pursuer_targets[i] = index of evader that pursuer i is targeting
    target_one_hot = np.zeros((n_red, n_blue))
    for pursuer_idx, target_evader_idx in enumerate(pursuer_targets):
        if target_evader_idx < n_blue:
            target_one_hot[pursuer_idx, target_evader_idx] = 1.0
    target_one_hot = target_one_hot.flatten()

    shared_state = np.concatenate([blue_shared, red_shared, target_one_hot])

    # =========================================================================
    # Concatenate all parts
    # =========================================================================
    observation = np.concatenate([own_state, ally_one_hot, shared_state])

    return observation


def extract_pursuer_state(pursuer_state) -> np.ndarray:
    """Extract raw state vector from a pursuer state message.

    Returns a 10D vector: pos(3), vel(3), rpy(3), active(1).

    Args:
        pursuer_state: A CrazyflieState message for a pursuer.

    Returns:
        State vector of shape (10,).
    """
    pos = np.array(pursuer_state.position)
    vel = np.array(pursuer_state.velocity)
    quat = np.array(pursuer_state.attitude)  # [x, y, z, w]
    rpy = quat_to_rpy(quat)
    active = float(pursuer_state.active)
    return np.concatenate([pos, vel, rpy, [active]])


def build_header(n_agents: int, obs_dim: int, n_pursuers: int = 0) -> list:
    """Build CSV header with column names matching eval script format.

    Uses the same naming convention as eval_mappo_ffn.py --save-obs:
    time, agent0_obs0, agent0_obs1, ..., agentN_obsM, red0_pos_x, ...

    Args:
        n_agents: Number of blue agents (evaders).
        obs_dim: Dimension of observation per agent.
        n_pursuers: Number of red agents (pursuers).

    Returns:
        List of column names.
    """
    header = ['time']
    for agent_idx in range(n_agents):
        for feat_idx in range(obs_dim):
            header.append(f'agent{agent_idx}_obs{feat_idx}')

    # Pursuer (red) state columns: pos(3), vel(3), rpy(3), active(1) = 10D
    red_state_names = [
        'pos_x', 'pos_y', 'pos_z',
        'vel_x', 'vel_y', 'vel_z',
        'roll', 'pitch', 'yaw',
        'active',
    ]
    for red_idx in range(n_pursuers):
        for name in red_state_names:
            header.append(f'red{red_idx}_{name}')

    return header


def extract_observations_from_bag(bag_path: Path) -> tuple[np.ndarray, list, str | None]:
    """Extract observations from a ROS2 bag file.

    Extracts data only during RUNNING status. Stops when BLUE_WON or RED_WON
    status is detected.

    Args:
        bag_path: Path to the bag directory.

    Returns:
        Tuple of (data array, header list, game_outcome).
        Data array shape: (n_samples, 1 + n_agents * obs_dim)
        First column is time from start in seconds.
        game_outcome is 'blue_won', 'red_won', or None if game didn't finish.
    """
    observations = []
    timestamps = []
    start_time = None
    n_evaders = None
    n_pursuers = None
    obs_dim = None
    game_outcome = None

    with Reader(bag_path) as reader:
        # Find connections for our topic
        connections = [c for c in reader.connections if c.topic == STATUS_TOPIC]

        if not connections:
            raise ValueError(f"Topic '{STATUS_TOPIC}' not found in bag. "
                           f"Available topics: {[c.topic for c in reader.connections]}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_message(rawdata, Status)

            # Check for game end first (before extracting data)
            if msg.status == Status.MAPE_STATUS_BLUE_WON:
                game_outcome = 'blue_won'
                break
            elif msg.status == Status.MAPE_STATUS_RED_WON:
                game_outcome = 'red_won'
                break

            # Only extract data during RUNNING
            if msg.status != Status.MAPE_STATUS_RUNNING:
                continue

            # Initialize dimensions on first RUNNING message
            if n_evaders is None:
                n_evaders = msg.n_evaders
                n_pursuers = msg.n_pursuers
                # Compute obs_dim: 12 + n_blue + 7*n_blue + 7*n_red + n_red*n_blue
                obs_dim = 12 + n_evaders + 7 * n_evaders + 7 * n_pursuers + n_pursuers * n_evaders
                print(f"Detected {n_evaders} evaders and {n_pursuers} pursuers")
                print(f"Observation dimension per agent: {obs_dim}")

            # Record start time
            if start_time is None:
                start_time = timestamp

            # Extract time from start (nanoseconds -> seconds)
            time_from_start = (timestamp - start_time) / 1e9
            timestamps.append(time_from_start)

            # Compute observations for all blue agents (evaders)
            row = []
            for agent_idx in range(n_evaders):
                obs = compute_observation(
                    evader_states=msg.cf_evader_states,
                    pursuer_states=msg.cf_pursuer_states,
                    pursuer_targets=msg.pursuer_targets,
                    agent_idx=agent_idx,
                )
                row.extend(obs)

            # Extract raw state for all pursuers (red agents)
            for pursuer in msg.cf_pursuer_states:
                row.extend(extract_pursuer_state(pursuer))

            observations.append(row)

    if not observations:
        raise ValueError("No RUNNING/WIN status messages found in bag")

    if game_outcome is None:
        print("Warning: Game did not reach a win state (bag may be incomplete)")

    # Build data array with time column
    data = np.column_stack([timestamps, observations])
    header = build_header(n_evaders, obs_dim, n_pursuers)

    return data, header, game_outcome


def main():
    parser = argparse.ArgumentParser(
        description='Extract observations from a ROS2 bag file to CSV format'
    )
    parser.add_argument(
        'bag_path',
        type=str,
        help='Path to the ROS2 bag directory'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: <bag_name>_obs.csv)'
    )

    args = parser.parse_args()

    bag_path = Path(args.bag_path)
    if not bag_path.exists():
        print(f"Error: Bag path not found: {bag_path}")
        return 1

    # Default output path
    if args.output is None:
        output_path = bag_path.parent / f"{bag_path.name}_obs.csv"
    else:
        output_path = Path(args.output)

    print(f"Reading bag: {bag_path}")
    print(f"Output: {output_path}")

    try:
        data, header, game_outcome = extract_observations_from_bag(bag_path)
    except Exception as e:
        print(f"Error reading bag: {e}")
        return 1

    # Save to CSV
    np.savetxt(
        output_path,
        data,
        delimiter=',',
        header=','.join(header),
        comments=''
    )

    print(f"Saved {len(data)} samples to {output_path}")
    print(f"Duration: {data[-1, 0]:.2f} seconds")
    if game_outcome:
        print(f"Game outcome: {game_outcome.upper().replace('_', ' ')}")

    return 0


if __name__ == '__main__':
    exit(main())
