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

from nav_msgs.msg import Odometry
from crazyflie_interfaces.msg import LogDataGeneric
from multiagent_pursuit_evasion_interfaces.msg import Status, EvaderState, PursuerState
from multiagent_pursuit_evasion_interfaces.srv import Command, ReadyForLowLevel

from functools import partial


STATUS_PUBLISHER_FREQUENCY = 50  # Hz
GRAVITY = 9.81


class MultiAgentPursuitEvasionServer(Node):
    """Server node for multi-agent pursuit-evasion experiments.

    Subscribes to pose, velocity, and angular_velocity for all agents.
    Publishes Status messages containing full game state.
    Handles collision detection, boundary violations, and target reassignment.
    """

    def __init__(self, n_blue: int, n_red: int, config: dict, require_accel: bool = True):
        """Initialize the server.

        Args:
            n_blue: Number of blue (evader) agents.
            n_red: Number of red (pursuer) agents.
            config: Environment configuration dictionary.
            require_accel: Whether acceleration data is required for blue agents.
        """
        super().__init__('multiagent_pursuit_evasion_server')

        self.status = Status.MAPE_STATUS_OFF
        self.n_blue = n_blue
        self.n_red = n_red
        self.n_total_agents = n_blue + n_red
        self.config = config
        self.require_accel = require_accel

        # Extract config parameters
        self.bb_collision_tolerance = config.get('bb_collision_tolerance', 0.2)
        self.rr_collision_tolerance = config.get('rr_collision_tolerance', 0.2)
        self.rb_collision_tolerance = config.get('rb_collision_tolerance', 0.2)
        self.boundary_size = config.get('boundary_size', 5.0)
        self.min_altitude = config.get('min_altitude', 0.1)
        self.max_altitude = config.get('max_altitude', 2.0)

        # CF names: blue = blue_1, blue_2, ...; red = red_1, red_2, ...
        self.blue_cf_names = [f'blue_{i}' for i in range(1, self.n_blue + 1)]
        self.red_cf_names = [f'red_{i}' for i in range(1, self.n_red + 1)]

        # Readiness tracking: [odom, body_rates, (accel)] for blue, [odom] for red
        n_blue_ready_fields = 3 if require_accel else 2
        self.blue_ready = np.full((self.n_blue, n_blue_ready_fields), False, dtype=bool)
        self.red_ready = np.full((self.n_red, 1), False, dtype=bool)

        # State tracking
        self.cf_blue_states = [EvaderState() for _ in range(n_blue)]
        self.cf_red_states = [PursuerState() for _ in range(n_red)]

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

        # Track which teams are ready for low-level control
        self.teams_ready_for_low_level = {'evader': False, 'pursuer': False}

        # Track if game over has been reported (to avoid spam)
        self.game_over_reported = False

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
                partial(self._odom_callback, index=n, cf_states=self.cf_blue_states, ready=self.blue_ready, ready_index=0),
                sensor_qos)

            self.create_subscription(
                LogDataGeneric,
                f'/{cf_name}/body_rates',
                partial(self._body_rates_callback, index=n, cf_states=self.cf_blue_states, ready=self.blue_ready, ready_index=1),
                sensor_qos)

            if self.require_accel:
                self.create_subscription(
                    LogDataGeneric,
                    f'/{cf_name}/accel',
                    partial(self._acceleration_callback, index=n, cf_states=self.cf_blue_states, ready=self.blue_ready, ready_index=2),
                    sensor_qos)

        # Create subscriptions for red agents (odom only, no body_rates)
        for n in range(n_red):
            cf_name = self.red_cf_names[n]

            self.create_subscription(
                Odometry,
                f'/{cf_name}/odom',
                partial(self._odom_callback, index=n, cf_states=self.cf_red_states, ready=self.red_ready, ready_index=0),
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

        # ReadyForLowLevel service - teams call this when ready for low-level control
        self.ready_srv = self.create_service(
            ReadyForLowLevel, 'ready_for_low_level', self._ready_for_low_level_callback
        )

        # Status publishing thread (bypasses ROS2 executor scheduling)
        self.publish_period = 1.0 / STATUS_PUBLISHER_FREQUENCY
        self.running = True
        self.publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
        self.publisher_thread.start()

        self.get_logger().info(
            f'MAPE Server initialized: {n_blue} blue vs {n_red} red agents (require_accel={require_accel})'
        )
        self.get_logger().info(
            f'Collision tolerances: BB={self.bb_collision_tolerance}, RR={self.rr_collision_tolerance}, BR={self.rb_collision_tolerance}'
        )
        self.get_logger().info(
            f'Boundary: size={self.boundary_size}, alt=[{self.min_altitude}, {self.max_altitude}]'
        )

    def _command_callback(self, request: Command.Request, response: Command.Response):
        """Handle command service requests."""
        if request.command == Command.Request.MAPE_CMD_OFF:
            self.get_logger().info('Command: OFF')
            self.status = Status.MAPE_STATUS_OFF
            # Reset all agents to active and ready states when turning off
            for state in self.cf_blue_states:
                state.active = True
            for state in self.cf_red_states:
                state.active = True
            self.teams_ready_for_low_level = {'evader': False, 'pursuer': False}
            self.game_over_reported = False
        elif request.command == Command.Request.MAPE_CMD_TAKEOFF:
            if self._all_ready():
                self.get_logger().info('Command: TAKEOFF accepted')
                self.status = Status.MAPE_STATUS_TAKEOFF
                # Reset low-level ready states for new takeoff
                self.teams_ready_for_low_level = {'evader': False, 'pursuer': False}
            else:
                self.get_logger().info('Command: TAKEOFF denied - agents not ready')
                self._log_ready_status()
        elif request.command == Command.Request.MAPE_CMD_RUN:
            if self.status != Status.MAPE_STATUS_INITIALIZED:
                self.get_logger().info('Command: RUN denied - not in INITIALIZED state (drones must complete takeoff first)')
            elif self._all_ready():
                self.get_logger().info('Command: RUN accepted')
                self.status = Status.MAPE_STATUS_RUNNING
            else:
                self.get_logger().info('Command: RUN denied - agents not ready')
                self._log_ready_status()
        else:
            self.get_logger().warn(f'Invalid command: {request.command}')

        return response

    def _ready_for_low_level_callback(self, request: ReadyForLowLevel.Request,
                                       response: ReadyForLowLevel.Response):
        """Handle team ready for low-level control notification."""
        team_name = request.team_name.lower()

        if team_name in self.teams_ready_for_low_level:
            self.teams_ready_for_low_level[team_name] = True
            self.get_logger().info(f'Team {team_name} ready for low-level control')

            # Check if all teams are ready
            if all(self.teams_ready_for_low_level.values()):
                self.get_logger().info('All teams ready - switching to INITIALIZED state')
                self.status = Status.MAPE_STATUS_INITIALIZED

            response.success = True
        else:
            self.get_logger().warn(f'Unknown team name: {team_name}')
            response.success = False

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
                if self.require_accel and not self.blue_ready[i, 2]:
                    missing.append('accel')
                self.get_logger().info(f'Blue {i} ({self.blue_cf_names[i]}) missing: {missing}')
        for i in range(self.n_red):
            if not np.all(self.red_ready[i]):
                missing = []
                if not self.red_ready[i, 0]:
                    missing.append('odom')
                self.get_logger().info(f'Red {i} ({self.red_cf_names[i]}) missing: {missing}')

    def _odom_callback(self, msg: Odometry, index: int, cf_states: list, ready: np.ndarray, ready_index: int):
        """Handle odometry message (contains pose and linear velocity)."""
        cf_states[index].position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
        cf_states[index].attitude = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        cf_states[index].velocity = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ]
        ready[index, ready_index] = True

    def _body_rates_callback(self, msg: LogDataGeneric, index: int, cf_states: list, ready: np.ndarray, ready_index: int):
        """Handle body rates message (angular velocity).

        Values are compressed (multiplied by 1000), so divide to get actual rad/s.
        msg.values contains [roll_rate, pitch_rate, yaw_rate].
        """
        cf_states[index].angular_velocity = [
            msg.values[0] / 1000.0,
            msg.values[1] / 1000.0,
            msg.values[2] / 1000.0
        ]
        ready[index, ready_index] = True

    def _acceleration_callback(self, msg: LogDataGeneric, index: int, cf_states: list, ready: np.ndarray, ready_index: int):
        """Handle acceleration message.

        Values are compressed (multiplied by 1000) and include gravity.
        Divide by 1000 and subtract gravity from z to get actual acceleration.
        msg.values contains [accel_x, accel_y, accel_z].
        """
        cf_states[index].acceleration = [
            msg.values[0] / 1000.0,
            msg.values[1] / 1000.0,
            msg.values[2] / 1000.0 - GRAVITY
        ]
        ready[index, ready_index] = True

    def _evader_states_to_np(self, states: list) -> np.ndarray:
        """Convert EvaderState list to numpy array.

        Returns:
            Array of shape (n_agents, 17): [pos(3), vel(3), quat(4), ang_vel(3), accel(3), active(1)]
        """
        np_states = np.zeros((len(states), 17))
        for idx, state in enumerate(states):
            np_states[idx, :] = np.array([
                *state.position,          # 0:3
                *state.velocity,          # 3:6
                *state.attitude,          # 6:10 (quaternion)
                *state.angular_velocity,  # 10:13
                *state.acceleration,      # 13:16
                float(state.active)       # 16
            ])
        return np_states

    def _pursuer_states_to_np(self, states: list) -> np.ndarray:
        """Convert PursuerState list to numpy array.

        Returns:
            Array of shape (n_agents, 11): [pos(3), vel(3), quat(4), active(1)]
        """
        np_states = np.zeros((len(states), 11))
        for idx, state in enumerate(states):
            np_states[idx, :] = np.array([
                *state.position,    # 0:3
                *state.velocity,    # 3:6
                *state.attitude,    # 6:10 (quaternion)
                float(state.active) # 10
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
        # Don't process collisions if game is already over
        if self.game_over_reported:
            return

        blue_states = self._evader_states_to_np(self.cf_blue_states)
        red_states = self._pursuer_states_to_np(self.cf_red_states)

        blue_pos = blue_states[:, 0:3]
        red_pos = red_states[:, 0:3]

        # Blue-blue collisions (vectorized, excluding self via meshgrid indexing)
        # Computes pairwise distances and uses permutation indexing to exclude diagonal
        bb_dist = norm(blue_pos[self.blue_meshgrid, :] - blue_pos[:, None, :], axis=2, keepdims=True)
        bb_collision = bb_dist < self.bb_collision_tolerance
        # Index with [meshgrid[:-1,:].T, permutations[:,1:]] to get off-diagonal elements
        bb_crash = np.any(bb_collision[self.blue_meshgrid[:-1, :].T, self.blue_permutations[:, 1:]], axis=1).flatten()

        # Red-red collisions (vectorized, excluding self via meshgrid indexing)
        rr_dist = norm(red_pos[self.red_meshgrid, :] - red_pos[:, None, :], axis=2, keepdims=True)
        rr_collision = rr_dist < self.rr_collision_tolerance
        rr_crash = np.any(rr_collision[self.red_meshgrid[:-1, :].T, self.red_permutations[:, 1:]], axis=1).flatten()

        # Blue-red captures: blue dies if caught by active red (vectorized)
        # Use red_meshgrid truncated to n_blue rows for broadcasting
        br_dist = norm(red_pos[self.red_meshgrid[0:self.n_blue, :], :] - blue_pos[:, None, :], axis=2, keepdims=True)
        br_crash = np.any((br_dist < self.rb_collision_tolerance) * red_states[:, [-1]], axis=1).flatten()

        # Red-blue captures: red dies after capturing active blue (vectorized)
        # Use blue_meshgrid truncated to n_red rows for broadcasting
        rb_dist = norm(blue_pos[self.blue_meshgrid[0:self.n_red, :], :] - red_pos[:, None, :], axis=2, keepdims=True)
        rb_crash = np.any((rb_dist < self.rb_collision_tolerance) * blue_states[:, [-1]], axis=1).flatten()

        # Log pairwise distances when close to collision threshold
        for i in range(self.n_red):
            for j in range(i + 1, self.n_red):
                d = norm(red_pos[i] - red_pos[j])
                if d < self.rr_collision_tolerance * 2:
                    self.get_logger().info(
                        f'RR dist r{i}-r{j}: {d:.3f}m (tol={self.rr_collision_tolerance})'
                    )
        for i in range(self.n_blue):
            for j in range(i + 1, self.n_blue):
                d = norm(blue_pos[i] - blue_pos[j])
                if d < self.bb_collision_tolerance * 2:
                    self.get_logger().info(
                        f'BB dist b{i}-b{j}: {d:.3f}m (tol={self.bb_collision_tolerance})'
                    )
        for i in range(self.n_blue):
            for j in range(self.n_red):
                d = norm(blue_pos[i] - red_pos[j])
                if d < self.rb_collision_tolerance * 2:
                    self.get_logger().info(
                        f'BR dist b{i}-r{j}: {d:.3f}m (tol={self.rb_collision_tolerance})'
                    )

        # Boundary violations (blue only)
        out_of_bounds = (
            (np.abs(blue_pos[:, 0]) > self.boundary_size) |
            (np.abs(blue_pos[:, 1]) > self.boundary_size) |
            (blue_pos[:, 2] < self.min_altitude) |
            (blue_pos[:, 2] > self.max_altitude)
        )

        blue_deactivated = np.where(bb_crash | br_crash | out_of_bounds)[0]
        red_deactivated = np.where(rr_crash | rb_crash)[0]

        # Update states
        for idx in blue_deactivated:
            if self.cf_blue_states[idx].active:  # Only log if actually changing
                reason = []
                if bb_crash[idx]:
                    reason.append("blue-blue collision")
                if br_crash[idx]:
                    reason.append("captured by red")
                if out_of_bounds[idx]:
                    pos = blue_pos[idx]
                    reason.append(f"out of bounds at pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                self.get_logger().info(f'BLUE {idx} deactivated: {", ".join(reason)}')
            self.cf_blue_states[idx].active = False
        for idx in red_deactivated:
            if self.cf_red_states[idx].active:  # Only log if actually changing
                reason = []
                if rr_crash[idx]:
                    reason.append("red-red collision")
                if rb_crash[idx]:
                    reason.append("captured blue (mutual)")
                self.get_logger().info(f'RED {idx} deactivated: {", ".join(reason)}')
            self.cf_red_states[idx].active = False

        # Check for game over conditions (only report once)
        if not self.game_over_reported and self.status == Status.MAPE_STATUS_RUNNING:
            blue_alive = np.array([s.active for s in self.cf_blue_states])
            red_alive = np.array([s.active for s in self.cf_red_states])

            if not np.any(blue_alive):
                self.get_logger().info('========== RED WINS! All blues eliminated ==========')
                self.status = Status.MAPE_STATUS_RED_WON
                self.game_over_reported = True
            elif not np.any(red_alive) and np.any(blue_alive):
                self.get_logger().info('========== BLUE WINS! All reds eliminated with blues surviving ==========')
                self.status = Status.MAPE_STATUS_BLUE_WON
                self.game_over_reported = True

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
        'bb_collision_tolerance': 0.2,
        'rr_collision_tolerance': 0.2,
        'rb_collision_tolerance': 0.2,
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
