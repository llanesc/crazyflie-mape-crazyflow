"""Multi-agent pursuit-evasion server node for hardware experiments.

Coordinates blue (evader) and red (pursuer) teams, handles collision detection,
boundary enforcement, and target reassignment matching the simulation environment.
"""

import json
import time
import threading
import numpy as np
from numpy.linalg import norm
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PoseStamped, AccelStamped, TwistStamped
from nav_msgs.msg import Odometry
from multiagent_pursuit_evasion_interfaces.msg import Status, State
from multiagent_pursuit_evasion_interfaces.srv import Command

from functools import partial


STATUS_PUBLISHER_FREQUENCY = 100  # Hz
GRAVITY = 9.81


def quat_to_euler(x: float, y: float, z: float, w: float) -> tuple:
    """Fast quaternion to euler (roll, pitch, yaw) conversion.

    Inline implementation avoiding tf_transformations overhead.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class MultiAgentPursuitEvasionServer(Node):
    """Server node for multi-agent pursuit-evasion experiments.

    Subscribes to pose, velocity, and angular_velocity for all agents.
    Publishes Status messages containing full game state.
    Handles collision detection, boundary violations, and target reassignment.
    """

    def __init__(self, n_blue: int, n_red: int, config: dict):
        """Initialize the server.

        Args:
            n_blue: Number of blue (evader) agents.
            n_red: Number of red (pursuer) agents.
            config: Environment configuration dictionary.
        """
        super().__init__('multiagent_pursuit_evasion_server')

        self.status = Status.MAPE_STATUS_OFF
        self.n_blue = n_blue
        self.n_red = n_red
        self.n_total_agents = n_blue + n_red
        self.config = config

        # Extract config parameters
        self.bb_crash_tolerance = config.get('bb_crash_tolerance', 0.2)
        self.rr_crash_tolerance = config.get('rr_crash_tolerance', 0.2)
        self.br_crash_tolerance = config.get('br_crash_tolerance', 0.2)
        self.boundary_size = config.get('boundary_size', 3.0)
        self.min_altitude = config.get('min_altitude', 0.1)
        self.max_altitude = config.get('max_altitude', 2.0)

        # CF names: blue = blue_1, blue_2, ...; red = red_1, red_2, ...
        self.blue_cf_names = [f'blue_{i}' for i in range(1, self.n_blue + 1)]
        self.red_cf_names = [f'red_{i}' for i in range(1, self.n_red + 1)]

        # Readiness tracking: [odom, body_rates] per agent
        self.blue_ready = np.full((self.n_blue, 2), False, dtype=bool)
        self.red_ready = np.full((self.n_red, 2), False, dtype=bool)

        # State tracking
        self.cf_blue_states = [State() for _ in range(n_blue)]
        self.cf_red_states = [State() for _ in range(n_red)]

        # Initialize all agents as active
        for state in self.cf_blue_states:
            state.active = True
        for state in self.cf_red_states:
            state.active = True

        # Target assignments: red agent i pursues blue agent targets[i]
        self.red_targets = list(range(min(n_blue, n_red)))
        # Extend with modulo assignment if more reds than blues
        if n_red > n_blue:
            self.red_targets = [i % n_blue for i in range(n_red)]

        # Pre-compute meshgrids and permutations for fast collision checking
        self.blue_meshgrid = np.arange(n_blue)[None, :].repeat(n_blue, axis=0)
        self.blue_permutations = np.array([np.roll(np.arange(n_blue), -k) for k in np.arange(n_blue)])
        self.red_meshgrid = np.arange(n_red)[None, :].repeat(n_red, axis=0)
        self.red_permutations = np.array([np.roll(np.arange(n_red), -k) for k in np.arange(n_red)])

        # Collision check frequency (every N status updates)
        self.collision_check_interval = 5
        self.status_update_counter = 0


        # QoS: best effort, keep last 1 to drop old messages
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscriptions for blue agents
        for n in range(n_blue):
            cf_name = self.blue_cf_names[n]

            self.create_subscription(
                Odometry,
                f'/{cf_name}/odom',
                partial(self._odom_callback, index=n, cf_states=self.cf_blue_states, ready=self.blue_ready),
                sensor_qos)

            self.create_subscription(
                TwistStamped,
                f'/{cf_name}/body_rates',
                partial(self._body_rates_callback, index=n, cf_states=self.cf_blue_states, ready=self.blue_ready),
                sensor_qos)

            self.create_subscription(
                AccelStamped,
                f'/{cf_name}/accel',
                partial(self._acceleration_callback, index=n, cf_states=self.cf_blue_states),
                sensor_qos)

        # Create subscriptions for red agents
        for n in range(n_red):
            cf_name = self.red_cf_names[n]

            self.create_subscription(
                Odometry,
                f'/{cf_name}/odom',
                partial(self._odom_callback, index=n, cf_states=self.cf_red_states, ready=self.red_ready),
                sensor_qos)

            self.create_subscription(
                TwistStamped,
                f'/{cf_name}/body_rates',
                partial(self._body_rates_callback, index=n, cf_states=self.cf_red_states, ready=self.red_ready),
                sensor_qos)

        # Publisher for status (BEST_EFFORT to match subscribers)
        status_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.status_publisher = self.create_publisher(
            Status,
            '/multiagent_pursuit_evasion/status',
            status_qos
        )

        # Command service
        self.command_srv = self.create_service(Command, 'command', self._command_callback)

        # Status publishing thread (bypasses ROS2 executor scheduling)
        self.publish_period = 1.0 / STATUS_PUBLISHER_FREQUENCY
        self.running = True
        self.publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
        self.publisher_thread.start()

        self.get_logger().info(
            f'MAPE Server initialized: {n_blue} blue vs {n_red} red agents'
        )
        self.get_logger().info(
            f'Collision tolerances: BB={self.bb_crash_tolerance}, RR={self.rr_crash_tolerance}, BR={self.br_crash_tolerance}'
        )
        self.get_logger().info(
            f'Boundary: size={self.boundary_size}, alt=[{self.min_altitude}, {self.max_altitude}]'
        )

    def _command_callback(self, request: Command.Request, response: Command.Response):
        """Handle command service requests."""
        if request.command == Command.Request.MAPE_CMD_OFF:
            self.get_logger().info('Command: OFF')
            self.status = request.command
            # Reset all agents to active when turning off
            for state in self.cf_blue_states:
                state.active = True
            for state in self.cf_red_states:
                state.active = True
        elif request.command == Command.Request.MAPE_CMD_INITIALIZE:
            if self._all_ready():
                self.get_logger().info('Command: INITIALIZE accepted')
                self.status = request.command
            else:
                self.get_logger().info('Command: INITIALIZE denied - agents not ready')
                self._log_ready_status()
        elif request.command == Command.Request.MAPE_CMD_RUN:
            if self._all_ready():
                self.get_logger().info('Command: RUN accepted')
                self.status = request.command
            else:
                self.get_logger().info('Command: RUN denied - agents not ready')
                self._log_ready_status()
        else:
            self.get_logger().warn(f'Invalid command: {request.command}')

        return response

    def _all_ready(self) -> bool:
        """Check if all agents have received pose, velocity, and angular_velocity."""
        return np.all(self.blue_ready) and np.all(self.red_ready)

    def _log_ready_status(self):
        """Log which agents are not ready."""
        for i in range(self.n_blue):
            if not np.all(self.blue_ready[i]):
                missing = []
                if not self.blue_ready[i, 0]:
                    missing.append('odom')
                if not self.blue_ready[i, 1]:
                    missing.append('body_rates')
                self.get_logger().info(f'Blue {i} ({self.blue_cf_names[i]}) missing: {missing}')
        for i in range(self.n_red):
            if not np.all(self.red_ready[i]):
                missing = []
                if not self.red_ready[i, 0]:
                    missing.append('odom')
                if not self.red_ready[i, 1]:
                    missing.append('body_rates')
                self.get_logger().info(f'Red {i} ({self.red_cf_names[i]}) missing: {missing}')

    def _odom_callback(self, msg: Odometry, index: int, cf_states: list, ready: np.ndarray):
        """Handle odometry message (contains pose and linear velocity)."""
        cf_states[index].position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
        cf_states[index].attitude = list(quat_to_euler(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ))
        cf_states[index].velocity = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ]
        ready[index, 0] = True

    def _body_rates_callback(self, msg: TwistStamped, index: int, cf_states: list, ready: np.ndarray):
        """Handle body rates message (angular velocity)."""
        # Skip zero body rates messages (estimator not converged or packet corruption)
        # if (msg.twist.angular.x == 0.0 and
        #     msg.twist.angular.y == 0.0 and
        #     msg.twist.angular.z == 0.0):
        #     self.get_logger().debug(f'Skipping zero body_rates for agent {index}')
        #     return

        cf_states[index].angular_velocity = [
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z
        ]
        ready[index, 1] = True

    def _acceleration_callback(self, msg: AccelStamped, index: int, cf_states: list):
        """Handle acceleration message.

        Values are global acceleration in m/s^2 with gravity already removed.
        """
        cf_states[index].acceleration = [
            msg.accel.linear.x,
            msg.accel.linear.y,
            msg.accel.linear.z
        ]

    def _states_to_np(self, states: list) -> np.ndarray:
        """Convert State list to numpy array.

        Returns:
            Array of shape (n_agents, 13): [pos(3), vel(3), rpy(3), ang_vel(3), active(1)]
        """
        np_states = np.zeros((len(states), 13))
        for idx, state in enumerate(states):
            np_states[idx, :] = np.array([
                *state.position,      # 0:3
                *state.velocity,      # 3:6
                *state.attitude,      # 6:9
                *state.angular_velocity,  # 9:12
                float(state.active)   # 12
            ])
        return np_states

    def _check_collisions(self):
        """Check for collisions and boundary violations matching env logic.

        Updates agent alive status based on:
        - BB (blue-blue) collisions: both blues die
        - RR (red-red) collisions: both reds die
        - BR (blue-red) captures: blue dies, red that captured dies too
        - Boundary violations (blue only): blue dies if out of bounds

        Uses vectorized meshgrid indexing to exclude self-collisions.
        """
        blue_states = self._states_to_np(self.cf_blue_states)
        red_states = self._states_to_np(self.cf_red_states)

        blue_pos = blue_states[:, 0:3]
        red_pos = red_states[:, 0:3]

        # Blue-blue collisions (vectorized, excluding self via meshgrid indexing)
        # Computes pairwise distances and uses permutation indexing to exclude diagonal
        bb_dist = norm(blue_pos[self.blue_meshgrid, :] - blue_pos[:, None, :], axis=2, keepdims=True)
        bb_collision = bb_dist < self.bb_crash_tolerance
        # Index with [meshgrid[:-1,:].T, permutations[:,1:]] to get off-diagonal elements
        bb_crash = np.any(bb_collision[self.blue_meshgrid[:-1, :].T, self.blue_permutations[:, 1:]], axis=1).flatten()

        # Red-red collisions (vectorized, excluding self via meshgrid indexing)
        rr_dist = norm(red_pos[self.red_meshgrid, :] - red_pos[:, None, :], axis=2, keepdims=True)
        rr_collision = rr_dist < self.rr_crash_tolerance
        rr_crash = np.any(rr_collision[self.red_meshgrid[:-1, :].T, self.red_permutations[:, 1:]], axis=1).flatten()

        # Blue-red captures: blue dies if caught by active red (vectorized)
        # Use red_meshgrid truncated to n_blue rows for broadcasting
        br_dist = norm(red_pos[self.red_meshgrid[0:self.n_blue, :], :] - blue_pos[:, None, :], axis=2, keepdims=True)
        br_crash = np.any((br_dist < self.br_crash_tolerance) * red_states[:, [-1]], axis=1).flatten()

        # Red-blue captures: red dies after capturing active blue (vectorized)
        # Use blue_meshgrid truncated to n_red rows for broadcasting
        rb_dist = norm(blue_pos[self.blue_meshgrid[0:self.n_red, :], :] - red_pos[:, None, :], axis=2, keepdims=True)
        rb_crash = np.any((rb_dist < self.br_crash_tolerance) * blue_states[:, [-1]], axis=1).flatten()

        # Boundary violations (blue only)
        out_of_bounds = (
            (blue_pos[:, 2] < self.min_altitude) |
            (blue_pos[:, 2] > self.max_altitude) |
            (np.abs(blue_pos[:, 0]) > self.boundary_size) |
            (np.abs(blue_pos[:, 1]) > self.boundary_size)
        )

        blue_deactivated = np.where(bb_crash | br_crash | out_of_bounds)[0]
        red_deactivated = np.where(rr_crash | rb_crash)[0]

        # Update states
        for idx in blue_deactivated:
            self.cf_blue_states[idx].active = False
        for idx in red_deactivated:
            self.cf_red_states[idx].active = False

        # Target reassignment: when a blue dies, reassign pursuing reds to closest alive blue
        if len(blue_deactivated) > 0:
            blue_alive = np.array([s.active for s in self.cf_blue_states])
            red_alive = np.array([s.active for s in self.cf_red_states])
            self._reassign_targets(blue_pos, red_pos, blue_alive, red_alive)

    def _reassign_targets(self, blue_pos: np.ndarray, red_pos: np.ndarray,
                          blue_alive: np.ndarray, red_alive: np.ndarray):
        """Reassign red targets when their current target dies.

        Each red pursuing a dead blue is reassigned to the closest alive blue.
        Vectorized implementation using distance matrix.
        """
        # Check if any blues are alive
        if not np.any(blue_alive):
            return

        # Find reds that need reassignment (alive and current target is dead)
        current_targets = np.array(self.red_targets)
        needs_reassign = red_alive & ~blue_alive[current_targets]

        if not np.any(needs_reassign):
            return

        # Distance matrix: (n_red, n_blue)
        dist_matrix = norm(red_pos[:, None, :] - blue_pos[None, :, :], axis=2)

        # Mask dead blues with infinity
        dist_matrix[:, ~blue_alive] = np.inf

        # Find closest alive blue for each red
        new_targets = np.argmin(dist_matrix, axis=1)

        # Update only reds that need reassignment
        for r in np.where(needs_reassign)[0]:
            self.red_targets[r] = new_targets[r]


    def _publisher_loop(self):
        """Dedicated thread for status publishing at fixed rate."""
        next_time = time.perf_counter()
        while self.running:
            self._status_publisher_callback()
            next_time += self.publish_period
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Missed deadline, reset timing
                next_time = time.perf_counter()

    def _status_publisher_callback(self):
        """Publish status message at configured frequency."""
        if not self._all_ready():
            return

        # Check collisions every N updates when running
        if self.status == Status.MAPE_STATUS_RUNNING:
            self.status_update_counter += 1
            if self.status_update_counter >= self.collision_check_interval:
                self.status_update_counter = 0
                self._check_collisions()

        # Build and publish status message
        msg = Status()
        msg.status = self.status
        msg.evader_cf_names = self.blue_cf_names
        msg.pursuer_cf_names = self.red_cf_names
        msg.pursuer_targets = self.red_targets
        msg.n_evaders = self.n_blue
        msg.n_pursuers = self.n_red
        msg.cf_evader_states = self.cf_blue_states
        msg.cf_pursuer_states = self.cf_red_states
        self.status_publisher.publish(msg)


def main():
    """Standalone test entry point."""
    rclpy.init()

    # Default config for testing
    config = {
        'bb_crash_tolerance': 0.2,
        'rr_crash_tolerance': 0.2,
        'br_crash_tolerance': 0.2,
        'boundary_size': 3.0,
        'min_altitude': 0.1,
        'max_altitude': 2.0,
    }

    server = MultiAgentPursuitEvasionServer(n_blue=2, n_red=2, config=config)

    rclpy.spin(server)

    server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
