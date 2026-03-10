"""Pursuer and Evader team nodes for hardware experiments.

Contains TeamBase with common functionality, PursuerTeam with pure pursuit/ProNav control,
and EvaderTeam with learned policy inference.
"""

import numpy as np
from numpy import cos, sin, tan
from numpy.linalg import norm

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from builtin_interfaces.msg import Duration
from crazyflie_interfaces.msg import AttitudeSetpoint
from crazyflie_interfaces.srv import NotifySetpointsStop, Takeoff
from multiagent_pursuit_evasion_interfaces.msg import Status, EvaderState, PursuerState
from multiagent_pursuit_evasion_interfaces.srv import ReadyForLowLevel
from scipy.spatial.transform import Rotation
from drone_models.core import load_params


# =============================================================================
# DEFAULT CONFIGURATION VALUES
# =============================================================================
# These defaults are used when values are not provided via config dict
# (i.e. mape_config.yaml). Prefer editing mape_config.yaml instead.
# =============================================================================

_DEFAULT_BLUE_INITIAL_POS = np.array([
    [0.02, -0.57, 1.01],
    [0.01,  0.41, 0.90],
])

_DEFAULT_RED_INITIAL_POS = np.array([
    [3.48, -0.65, 0.81],
    [2.58,  0.25, 0.95],
])

_DEFAULT_ALTITUDE_THRESHOLD = 0.05
_DEFAULT_VELOCITY_THRESHOLD = 0.05
_DEFAULT_TAKEOFF_DURATION = 3.0
_DEFAULT_ROLL_PITCH_MAX = 0.2
_DEFAULT_YAW_MAX = 0.1
_DEFAULT_SETTLING_VELOCITY_THRESHOLD = 0.2
_DEFAULT_HOVER_KI_Z = 6.0
_DEFAULT_HOVER_INTEGRAL_CAP = 1.0
_DEFAULT_BRAKING_VEL_MULTIPLIER = 3.0


def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix, flattened row-major.

    Args:
        x, y, z, w: Quaternion components.

    Returns:
        Flattened rotation matrix of shape (9,).
    """
    # Pre-compute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array([
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
    ], dtype=np.float32)
    return R


def quat_to_euler(x: float, y: float, z: float, w: float) -> tuple:
    """Convert quaternion to euler (roll, pitch, yaw).

    Args:
        x, y, z, w: Quaternion components.

    Returns:
        Tuple of (roll, pitch, yaw) in radians.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class TeamBase(Node):
    """Base class for pursuer and evader teams.

    Provides common functionality for loading drone parameters, converting
    thrust to PWM, and publishing attitude setpoints.
    """

    def __init__(self, node_name: str, team_type: str, config: dict):
        """Initialize team base.

        Args:
            node_name: ROS node name.
            team_type: Team type for server notification ('evader' or 'pursuer').
            config: Environment configuration dictionary.
        """
        super().__init__(node_name)

        self.config = config
        self.team_type = team_type  # 'evader' or 'pursuer' for server notification

        # Load physical parameters from drone-models
        drone_model = config.get('drone_model', 'cf2x_T350')
        drone_params = load_params("so_rpy", drone_model)

        self.mass = config.get('mass', 0.0406)
        self.gravity = float(np.abs(drone_params["gravity_vec"][2]))
        self.thrust_min_hw = float(np.abs(drone_params["thrust_min"])) * 4  # Per motor -> collective
        self.thrust_max_hw = float(np.abs(drone_params["thrust_max"])) * 4  # Per motor -> collective
        # Use config values if provided, otherwise compute from mass/gravity
        self.thrust_min = config.get('thrust_min', self.mass * self.gravity * 0.5)
        self.thrust_max = config.get('thrust_max', self.mass * self.gravity * 1.5)

        # Initial positions from config (mape_config.yaml), with defaults
        blue_pos = config.get('blue_initial_pos', _DEFAULT_BLUE_INITIAL_POS)
        red_pos = config.get('red_initial_pos', _DEFAULT_RED_INITIAL_POS)
        self.blue_initial_pos = np.array(blue_pos, dtype=np.float64).copy()
        self.red_initial_pos = np.array(red_pos, dtype=np.float64).copy()

        # Heights for takeoff command (high-level commander uses these)
        self.blue_initial_height = self.blue_initial_pos[:, 2].tolist()
        self.red_initial_height = self.red_initial_pos[:, 2].tolist()

        # Attitude limits from config
        self.roll_pitch_max = config.get('roll_pitch_max', _DEFAULT_ROLL_PITCH_MAX)
        self.yaw_max = config.get('yaw_max', _DEFAULT_YAW_MAX)

        # Thrust-to-PWM parameters (linear mapping matching Mellinger controller)
        self.pwm_max = 65535
        self.pwm_min = 7000

        self.team_size = None
        self.cf_list = []
        self.attitude_setpoint_publishers = None
        self.team_initialized = False
        self.initialized_control = False

        # Takeoff and low-level control state from config
        self.takeoff_commanded = False
        self.low_level_enabled = False
        self.altitude_threshold = config.get('altitude_threshold', _DEFAULT_ALTITUDE_THRESHOLD)
        self.velocity_threshold = config.get('velocity_threshold', _DEFAULT_VELOCITY_THRESHOLD)
        self.takeoff_duration_sec = config.get('takeoff_duration', _DEFAULT_TAKEOFF_DURATION)

        # Settling state for inactive agents (initialized in initialize_team)
        self.settling_velocity_threshold = config.get('settling_velocity_threshold', _DEFAULT_SETTLING_VELOCITY_THRESHOLD)
        self.agent_settled = None  # Bool array: True if agent has settled (velocity below threshold)
        self.settled_pos = None    # Position where agent settled

        # PID hover control state (Z-axis integral)
        self.hover_ki_z = config.get('hover_ki_z', _DEFAULT_HOVER_KI_Z)
        self.hover_integral_cap = config.get('hover_integral_cap', _DEFAULT_HOVER_INTEGRAL_CAP)
        self.braking_vel_multiplier = config.get('braking_vel_multiplier', _DEFAULT_BRAKING_VEL_MULTIPLIER)
        self.z_error_integral = None  # Accumulated Z position error integral
        self.last_control_time = None  # For computing dt

    def initialize_team(self, cf_list: list):
        """Initialize team with CF names and create publishers.

        Args:
            cf_list: List of CF names (e.g., ['cf_1', 'cf_2']).
        """
        self.team_initialized = True
        self.cf_list = cf_list
        self.team_size = len(cf_list)
        self.attitude_setpoint_publishers = [
            self.create_publisher(
                AttitudeSetpoint,
                f'/{cf}/cmd_attitude',
                10
            ) for cf in cf_list
        ]

        # Initialize settling state tracking for inactive agents
        self.agent_settled = np.zeros(self.team_size, dtype=bool)
        self.settled_pos = np.zeros((self.team_size, 3))

        # Initialize PID hover integral state
        self.z_error_integral = np.zeros(self.team_size)
        self.last_control_time = None

    def command_takeoff(self, target_heights: list):
        """Command takeoff for all drones to their target heights.

        Args:
            target_heights: List of target heights for each drone.
        """
        if self.takeoff_commanded:
            return

        for i, cf in enumerate(self.cf_list):
            height = target_heights[i] if i < len(target_heights) else target_heights[-1]
            client = self.create_client(Takeoff, f'/{cf}/takeoff')
            if client.wait_for_service(timeout_sec=1.0):
                request = Takeoff.Request()
                request.group_mask = 0
                request.height = float(height)
                request.duration = Duration(sec=int(self.takeoff_duration_sec),
                                            nanosec=int((self.takeoff_duration_sec % 1) * 1e9))
                future = client.call_async(request)
                self.get_logger().info(f'Commanded takeoff for {cf} to height {height:.2f}m')
            else:
                self.get_logger().warn(f'Takeoff service not available for {cf}')

        self.takeoff_commanded = True

    def check_ready_for_low_level(self, states: np.ndarray, target_heights: list) -> bool:
        """Check if all drones have reached altitude and are stable.

        Args:
            states: Current states array (n_drones, 10) with [pos, vel, rpy, active].
            target_heights: Target heights for each drone.

        Returns:
            True if all drones are ready for low-level control.
        """
        if self.low_level_enabled:
            return True

        for i in range(len(self.cf_list)):
            height = target_heights[i] if i < len(target_heights) else target_heights[-1]
            current_z = states[i, 2]
            velocity = states[i, 3:6]
            vel_magnitude = np.linalg.norm(velocity)

            altitude_error = abs(current_z - height)
            if altitude_error > self.altitude_threshold:
                return False
            if vel_magnitude > self.velocity_threshold:
                return False

        return True

    def switch_to_low_level(self):
        """Switch all drones from high-level to low-level attitude control."""
        if self.low_level_enabled:
            return

        for i, cf in enumerate(self.cf_list):
            client = self.create_client(NotifySetpointsStop, f'/{cf}/notify_setpoints_stop')
            if client.wait_for_service(timeout_sec=1.0):
                request = NotifySetpointsStop.Request()
                request.remain_valid_millisecs = 0
                future = client.call_async(request)
                self.get_logger().info(f'Switched {cf} to low-level control')

                # Send zero attitude setpoint immediately after switching
                # This is required before streaming actual setpoints
                zero_setpoint = AttitudeSetpoint()
                zero_setpoint.roll = 0.0
                zero_setpoint.pitch = 0.0
                zero_setpoint.yaw_rate = 0.0
                zero_setpoint.thrust = 0
                self.attitude_setpoint_publishers[i].publish(zero_setpoint)
                self.get_logger().info(f'Sent zero setpoint to {cf}')
            else:
                self.get_logger().warn(f'notify_setpoints_stop service not available for {cf}')

        self.low_level_enabled = True
        self.get_logger().info('All drones switched to low-level attitude control')

        # Notify server that this team is ready for low-level control
        self._notify_server_ready()

    def _notify_server_ready(self):
        """Notify server that this team is ready for low-level control."""
        client = self.create_client(ReadyForLowLevel, '/ready_for_low_level')
        if client.wait_for_service(timeout_sec=1.0):
            request = ReadyForLowLevel.Request()
            request.team_name = self.team_type
            future = client.call_async(request)
            self.get_logger().info(f'Notified server that {self.team_type} team is ready')
        else:
            self.get_logger().warn('ready_for_low_level service not available')

    def cmd_attitude_setpoint(self, control: np.ndarray, index: int,
                              current_yaw: float = 0.0):
        """Publish attitude setpoint command.

        Args:
            control: [roll_rad, pitch_rad, yaw_rad, thrust_N] in physical units.
            index: Agent index in team.
            current_yaw: Current yaw angle in radians for yaw tracking.
        """
        roll_rad, pitch_rad, yaw_des, thrust_N = control.ravel()

        # PID controller: convert desired yaw angle to yaw rate
        yaw_error = yaw_des - current_yaw
        # Wrap to [-pi, pi]
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        kP_yaw = 2.5
        kD_yaw = 0.5
        kI_yaw = 0.3
        dt = 0.02  # ~50Hz control loop
        if not hasattr(self, '_prev_yaw_error'):
            self._prev_yaw_error = {}
        if not hasattr(self, '_yaw_error_integral'):
            self._yaw_error_integral = {}
        prev_error = self._prev_yaw_error.get(index, yaw_error)
        integral = self._yaw_error_integral.get(index, 0.0)
        integral = np.clip(integral + yaw_error * dt, -0.5, 0.5)
        d_error = (yaw_error - prev_error) / dt
        self._prev_yaw_error[index] = yaw_error
        self._yaw_error_integral[index] = integral
        yaw_rate = -kP_yaw * yaw_error - kD_yaw * d_error - kI_yaw * integral

        setpoint = AttitudeSetpoint()
        setpoint.roll = roll_rad
        setpoint.pitch = pitch_rad
        setpoint.yaw_rate = yaw_rate
        setpoint.thrust = self.thrust_to_pwm(thrust_N)

        self.attitude_setpoint_publishers[index].publish(setpoint)

    def thrust_to_pwm(self, collective_thrust: float) -> int:
        """Convert collective thrust (N) to PWM value using linear mapping.

        Uses the same linear mapping as the Mellinger controller:
        pwm = (thrust / thrust_max) * pwm_max

        Args:
            collective_thrust: Total thrust in Newtons.

        Returns:
            PWM value [0, 65535].
        """
        # Clamp thrust to valid range
        collective_thrust = np.clip(collective_thrust, self.thrust_min, self.thrust_max)

        # Linear mapping: pwm = (thrust / thrust_max) * pwm_max
        pwm = (collective_thrust / self.thrust_max_hw) * self.pwm_max

        return int(np.clip(pwm, self.pwm_min, self.pwm_max))

    def states_to_np(self, states: list) -> np.ndarray:
        """Convert PursuerState list to numpy array (10D).

        Converts quaternion attitude to euler angles for control.

        Returns:
            Array of shape (n_agents, 10): [pos(3), vel(3), rpy(3), active(1)]
        """
        np_states = np.zeros((len(states), 10))
        for idx, state in enumerate(states):
            # Convert quaternion to euler
            rpy = quat_to_euler(
                state.attitude[0],
                state.attitude[1],
                state.attitude[2],
                state.attitude[3]
            )
            np_states[idx, :] = np.array([
                *state.position,
                *state.velocity,
                *rpy,
                float(state.active)
            ])
        return np_states

    def states_to_np_extended(self, states: list) -> np.ndarray:
        """Convert EvaderState list to numpy array with rotation matrix and body rates (19D).

        Converts quaternion attitude to flattened rotation matrix.

        Returns:
            Array of shape (n_agents, 19): [pos(3), vel(3), rotmat_flat(9), body_rates(3), active(1)]
        """
        np_states = np.zeros((len(states), 19))
        for idx, state in enumerate(states):
            rotmat_flat = quat_to_rotmat(
                state.attitude[0],
                state.attitude[1],
                state.attitude[2],
                state.attitude[3]
            )
            np_states[idx, :] = np.array([
                *state.position,          # 0:3
                *state.velocity,          # 3:6
                *rotmat_flat,             # 6:15
                *state.angular_velocity,  # 15:18
                float(state.active)       # 18
            ])
        return np_states

    def get_acceleration(self, states: list) -> np.ndarray:
        """Extract acceleration from State list.

        Returns:
            Array of shape (n_agents, 3) with acceleration in m/s^2.
        """
        accel = np.zeros((len(states), 3))
        for idx, state in enumerate(states):
            accel[idx] = state.acceleration
        return accel

    def ang_vel_to_rpy_rates(self, rpy: np.ndarray, ang_vel: np.ndarray) -> np.ndarray:
        """Convert body angular velocity [p, q, r] to Euler rates [droll, dpitch, dyaw].

        Uses the transformation matrix W: rpy_rates = W @ ang_vel

        Args:
            rpy: Euler angles [roll, pitch, yaw], shape (..., 3).
            ang_vel: Body angular velocity [p, q, r], shape (..., 3).

        Returns:
            Euler angle rates [droll, dpitch, dyaw], shape (..., 3).
        """
        phi = rpy[..., 0]    # roll
        theta = rpy[..., 1]  # pitch

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)
        inv_cos_theta = 1.0 / (cos_theta + 1e-8)

        p = ang_vel[..., 0]
        q = ang_vel[..., 1]
        r = ang_vel[..., 2]

        droll = p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
        dpitch = cos_phi * q - sin_phi * r
        dyaw = sin_phi * inv_cos_theta * q + cos_phi * inv_cos_theta * r

        return np.stack([droll, dpitch, dyaw], axis=-1)


class PursuerTeam(TeamBase):
    """Red team (pursuers) using Pure Pursuit, ProNav, Augmented ProNav, or TPN.

    Control logic matches the simulation environment's pursuer strategies.
    """

    def __init__(self, config: dict):
        """Initialize pursuer team.

        Args:
            config: Environment configuration dictionary.
        """
        super().__init__('pursuer_team', 'pursuer', config)

        self.pursuer_targets = []
        self.initial_pos = None

        # Get pursuit gains from config
        pursuit_gains = config.get('pursuit_gains', {})
        self.k_pxy = pursuit_gains.get('pp_k_pxy', 6.1624)
        self.k_vxy = pursuit_gains.get('pp_k_vxy', 3.39)
        self.k_pz = pursuit_gains.get('pp_k_pz', 20.0)
        self.k_vz = pursuit_gains.get('pp_k_vz', 10.0)

        # ProNav parameters
        self.N_pronav_fb = pursuit_gains.get('N_pronav_fb', 5.0)
        self.N_pronav_ff = pursuit_gains.get('N_pronav_ff', 1.0)
        self.velocity_closure_threshold = pursuit_gains.get('velocity_closure_threshold', 0.1)

        # Augmented ProNav parameters
        self.N_gain = pursuit_gains.get('N_gain', 3.0)
        self.V_min = pursuit_gains.get('V_min', 0.3)
        self.K_v = pursuit_gains.get('K_v', 2.5)

        # Strategy
        self.pursuer_strategy = config.get('pursuer_strategy', 'ProNav')

        # QoS: best effort, keep last 1 to process latest status only
        status_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(
            Status,
            '/multiagent_pursuit_evasion/status',
            self._status_callback,
            status_qos
        )

        self.get_logger().info(f'Pursuer team initialized with strategy: {self.pursuer_strategy}')

    def _status_callback(self, msg: Status):
        """Handle status message and compute control."""
        if not self.team_initialized:
            self.initialize_team(msg.pursuer_cf_names)
            # Use full XYZ initial positions from config
            self.initial_pos = self.red_initial_pos.copy()

        self.pursuer_targets = list(msg.pursuer_targets)
        blue_states = msg.cf_evader_states
        red_states = msg.cf_pursuer_states
        status = msg.status

        if status == Status.MAPE_STATUS_OFF:
            return

        # Command takeoff when in TAKEOFF status
        if status == Status.MAPE_STATUS_TAKEOFF and not self.takeoff_commanded:
            self.command_takeoff(self.red_initial_height)
            return

        # Check if ready to switch to low-level control during TAKEOFF
        if status == Status.MAPE_STATUS_TAKEOFF and not self.low_level_enabled:
            red_states_np = self.states_to_np(red_states)
            if self.check_ready_for_low_level(red_states_np, self.red_initial_height):
                self.switch_to_low_level()
            return  # Don't send attitude commands until switched

        # Only send attitude commands if low-level control is enabled
        if not self.low_level_enabled:
            return

        if not self.initialized_control:
            self.initialized_control = True
            for n in range(self.team_size):
                self.cmd_attitude_setpoint(np.array([0., 0., 0., 0.]), n)

        controls = self.compute_control(red_states, blue_states, status)

        red_states_np = self.states_to_np(red_states)
        for n in range(self.team_size):
            current_yaw = red_states_np[n, 8]
            self.cmd_attitude_setpoint(controls[n, :], n, current_yaw=current_yaw)

    def compute_control(self, red_states: list, blue_states: list, status: int) -> np.ndarray:
        """Compute control for all pursuers.

        Args:
            red_states: List of pursuer State messages.
            blue_states: List of evader State messages.
            status: Current game status.

        Returns:
            Control array of shape (n_pursuers, 4): [roll, pitch, yaw, thrust].
        """
        red_states_np = self.states_to_np(red_states)
        blue_states_np = self.states_to_np(blue_states)

        if status == Status.MAPE_STATUS_OFF:
            return np.zeros((red_states_np.shape[0], 4))

        elif status == Status.MAPE_STATUS_TAKEOFF or status == Status.MAPE_STATUS_INITIALIZED:
            # Hover at initial position (TAKEOFF included for brief transition period)
            return self._pid_control(red_states_np, self.initial_pos, np.zeros_like(self.initial_pos))

        elif status == Status.MAPE_STATUS_RUNNING:
            # Get target positions and velocities
            target_pos = blue_states_np[self.pursuer_targets, 0:3].copy()
            target_vel = blue_states_np[self.pursuer_targets, 3:6].copy()

            active_agents = (red_states_np[:, -1] == 1)
            inactive_agents = ~active_agents

            # Reset settled state and integral for agents that become active again
            self.agent_settled[active_agents] = False
            # Reset integral for active agents (using pursuit control, not PID)
            self.z_error_integral[active_agents] = 0.0

            # Update initial pos for active agents (for deactivated hover reference)
            self.initial_pos[active_agents] = red_states_np[active_agents, 0:3].copy()

            controls = np.zeros((self.team_size, 4))

            # Get blue acceleration for ProNav feedforward (measured from accel topic)
            blue_accel = self.get_acceleration(blue_states)
            target_accel = blue_accel[self.pursuer_targets, :].copy()
            target_accel[inactive_agents] = 0.

            # Handle inactive agents with settling behavior
            if np.any(inactive_agents):
                inactive_indices = np.where(inactive_agents)[0]
                inactive_pos = red_states_np[inactive_agents, 0:3]
                inactive_vel = red_states_np[inactive_agents, 3:6]
                inactive_vel_mag = np.linalg.norm(inactive_vel, axis=1)

                # Check which inactive agents have settled (velocity below threshold)
                for i, idx in enumerate(inactive_indices):
                    if not self.agent_settled[idx] and inactive_vel_mag[i] < self.settling_velocity_threshold:
                        self.agent_settled[idx] = True
                        self.settled_pos[idx] = inactive_pos[i]
                        self.get_logger().info(f'RED {idx} settled at pos {self.settled_pos[idx]}')

                # Build target positions for inactive agents
                inactive_target_pos = np.zeros_like(inactive_pos)
                # Track which inactive agents need braking (not yet settled)
                needs_braking = np.zeros(len(inactive_indices), dtype=bool)
                for i, idx in enumerate(inactive_indices):
                    if self.agent_settled[idx]:
                        # Use settled position for hover
                        inactive_target_pos[i] = self.settled_pos[idx]
                    else:
                        # Use deactivation position (last active frame) as fixed reference
                        # so the PID has position error to maintain altitude
                        inactive_target_pos[i] = self.initial_pos[idx]
                        needs_braking[i] = True

                # Settled agents: gentle PID hover
                if np.any(~needs_braking):
                    settled_mask = ~needs_braking
                    settled_idx = inactive_indices[settled_mask]
                    controls_settled = self._pid_control(
                        red_states_np[settled_idx, :],
                        inactive_target_pos[settled_mask],
                        np.zeros_like(inactive_target_pos[settled_mask]),
                        agent_indices=settled_idx
                    )
                    controls[settled_idx, :] = controls_settled

                # Unsettled agents: aggressive braking
                if np.any(needs_braking):
                    braking_idx = inactive_indices[needs_braking]
                    controls_braking = self._pid_control(
                        red_states_np[braking_idx, :],
                        inactive_target_pos[needs_braking],
                        np.zeros_like(inactive_target_pos[needs_braking]),
                        agent_indices=braking_idx,
                        braking=True
                    )
                    controls[braking_idx, :] = controls_braking

            if np.any(active_agents):
                if self.pursuer_strategy == "PP":
                    controls_active = self._pure_pursuit_control(
                        red_states_np[active_agents, :],
                        target_pos[active_agents],
                        target_vel[active_agents]
                    )
                elif self.pursuer_strategy == "AugProNav":
                    controls_active = self._aug_pronav_control(
                        red_states_np[active_agents, :],
                        target_pos[active_agents],
                        target_vel[active_agents],
                        target_accel[active_agents]
                    )
                elif self.pursuer_strategy == "TPN":
                    controls_active = self._tpn_control(
                        red_states_np[active_agents, :],
                        target_pos[active_agents],
                        target_vel[active_agents]
                    )
                else:  # ProNav (default)
                    controls_active = self._pronav_control(
                        red_states_np[active_agents, :],
                        target_pos[active_agents],
                        target_vel[active_agents],
                        target_accel[active_agents]
                    )
                controls[active_agents, :] = controls_active

            return controls

        elif status == Status.MAPE_STATUS_BLUE_WON or status == Status.MAPE_STATUS_RED_WON:
            # Game over - hover at current/settled position
            hover_pos = np.zeros_like(red_states_np[:, 0:3])
            for i in range(self.team_size):
                if self.agent_settled[i]:
                    hover_pos[i] = self.settled_pos[i]
                else:
                    hover_pos[i] = red_states_np[i, 0:3]
                    # Mark as settled if not already
                    vel_mag = np.linalg.norm(red_states_np[i, 3:6])
                    if vel_mag < self.settling_velocity_threshold:
                        self.agent_settled[i] = True
                        self.settled_pos[i] = red_states_np[i, 0:3]
            return self._pid_control(red_states_np, hover_pos, np.zeros_like(hover_pos))

        return np.zeros((len(red_states), 4))

    def _pure_pursuit(self, pos_rb: np.ndarray, vel_rb: np.ndarray) -> np.ndarray:
        """Pure pursuit guidance law matching env.

        LINEAR (no tanh), using env gains.

        Args:
            pos_rb: Relative position (target - pursuer), shape (n, 3).
            vel_rb: Relative velocity (target - pursuer), shape (n, 3).

        Returns:
            Acceleration command, shape (n, 3).
        """
        accel_xy = self.k_pxy * pos_rb[:, :2] + self.k_vxy * vel_rb[:, :2]
        accel_z = self.k_pz * pos_rb[:, 2:3] + self.k_vz * vel_rb[:, 2:3] + self.gravity

        return np.concatenate([accel_xy, accel_z], axis=-1)

    def _proportional_nav(self, pos_rb: np.ndarray, vel_rb: np.ndarray,
                          accel_target: np.ndarray) -> np.ndarray:
        """Proportional navigation guidance law matching env.

        Args:
            pos_rb: Relative position (target - pursuer), shape (n, 3).
            vel_rb: Relative velocity (target - pursuer), shape (n, 3).
            accel_target: Target acceleration, shape (n, 3).

        Returns:
            Acceleration command, shape (n, 3).
        """
        range_rb = norm(pos_rb, axis=-1, keepdims=True) + 1e-6
        direction_rb = pos_rb / range_rb

        # LOS angular rate
        omega_los = np.cross(pos_rb, vel_rb) / (range_rb ** 2 + 1e-6)

        # Orthogonal component of target acceleration
        accel_proj = np.sum(accel_target * direction_rb, axis=-1, keepdims=True) * direction_rb
        accel_orthogonal = accel_target - accel_proj

        # ProNav acceleration
        accel_pronav = self.N_pronav_fb * np.cross(vel_rb, omega_los) + self.N_pronav_ff * accel_orthogonal
        accel_pronav[:, 2] += self.gravity

        # Check velocity closure
        velocity_closure = -np.sum(vel_rb * direction_rb, axis=-1)
        use_pure_pursuit = velocity_closure < self.velocity_closure_threshold

        accel_pp = self._pure_pursuit(pos_rb, vel_rb)

        # Select between ProNav and Pure Pursuit
        accel = np.where(use_pure_pursuit[:, None], accel_pp, accel_pronav)

        return accel

    def _augmented_pronav(self, pos_rb: np.ndarray, vel_rb: np.ndarray,
                          vel_pursuer: np.ndarray, accel_target: np.ndarray) -> np.ndarray:
        """Augmented proportional navigation with speed floor matching env.

        Args:
            pos_rb: Relative position (target - pursuer), shape (n, 3).
            vel_rb: Relative velocity (target - pursuer), shape (n, 3).
            vel_pursuer: Pursuer velocity, shape (n, 3).
            accel_target: Target acceleration, shape (n, 3).

        Returns:
            Acceleration command, shape (n, 3).
        """
        dist = norm(pos_rb, axis=-1, keepdims=True) + 1e-6
        u_r = pos_rb / dist  # LOS unit vector
        omega = np.cross(pos_rb, vel_rb) / (dist ** 2)

        # Closing velocity
        Vc = -np.sum(u_r * vel_rb, axis=-1, keepdims=True)

        # Orthogonal component of target acceleration
        accel_proj = np.sum(accel_target * u_r, axis=-1, keepdims=True) * u_r
        accel_orthogonal = accel_target - accel_proj

        # Lateral steering
        a_lat = self.N_gain * Vc * np.cross(omega, u_r) + (self.N_gain * accel_orthogonal / 2.0)

        # Axial speed floor
        pursuer_speed = norm(vel_pursuer, axis=-1, keepdims=True) + 1e-6
        vel_pursuer_unit = vel_pursuer / pursuer_speed
        a_axial_mag = np.maximum(0.0, self.K_v * (self.V_min - pursuer_speed))
        a_axial = a_axial_mag * vel_pursuer_unit

        # Combined acceleration with gravity compensation
        accel_pronav = a_lat + a_axial
        accel_pronav[:, 2] += self.gravity

        # Fall back to pure pursuit when closure is insufficient
        velocity_closure = Vc.squeeze(-1)
        use_pure_pursuit = velocity_closure < self.velocity_closure_threshold

        accel_pp = self._pure_pursuit(pos_rb, vel_rb)

        accel = np.where(use_pure_pursuit[:, None], accel_pp, accel_pronav)

        return accel

    def _tpn_with_axial(self, pos_rb: np.ndarray, vel_rb: np.ndarray,
                        vel_pursuer: np.ndarray) -> np.ndarray:
        """True Proportional Navigation with axial speed floor matching env.

        Uses TPN (||vel_rb|| instead of Vc for stability) with axial speed floor.
        Unlike AugProNav:
        - Does NOT fall back to pure pursuit (always uses TPN)
        - Does NOT use target acceleration feedforward
        - Uses ||vel_rb|| instead of Vc for more stable behavior

        Args:
            pos_rb: Relative position (target - pursuer), shape (n, 3).
            vel_rb: Relative velocity (target - pursuer), shape (n, 3).
            vel_pursuer: Pursuer velocity, shape (n, 3).

        Returns:
            Acceleration command, shape (n, 3).
        """
        dist = norm(pos_rb, axis=-1, keepdims=True) + 1e-6
        u_r = pos_rb / dist  # LOS unit vector
        omega = np.cross(pos_rb, vel_rb) / (dist ** 2)

        # Lateral (Steering) - uses ||vel_rb|| instead of Vc for TPN
        vel_rb_norm = norm(vel_rb, axis=-1, keepdims=True)
        a_lat = self.N_gain * vel_rb_norm * np.cross(omega, u_r)

        # Axial (Speed Floor) - only accelerate if below V_min
        # Use signed dot product: positive when closing, negative when moving away
        pursuer_speed = np.sum(vel_pursuer * u_r, axis=-1, keepdims=True)
        a_axial_mag = np.maximum(0.0, self.K_v * (self.V_min - pursuer_speed))
        a_axial = a_axial_mag * u_r

        # TPN acceleration with gravity compensation
        accel = a_lat + a_axial
        accel[:, 2] += self.gravity

        return accel

    def _accel_to_attitude(self, accel: np.ndarray, current_rpy: np.ndarray) -> tuple:
        """Convert desired acceleration to attitude command matching env.

        Uses rotation-matrix-based conversion.

        Args:
            accel: Desired acceleration, shape (n, 3).
            current_rpy: Current roll-pitch-yaw, shape (n, 3).

        Returns:
            Tuple of (rpy_des, thrust_des).
        """
        # Desired thrust vector
        target_thrust = accel * self.mass

        # Get current body z-axis from rotation matrix
        R = Rotation.from_euler('xyz', current_rpy, degrees=False).as_matrix()
        z_axis = R[:, :, 2]

        # Current thrust along body z-axis
        current_thrust = np.sum(target_thrust * z_axis, axis=-1)

        # Desired body z-axis
        force_norm = norm(target_thrust, axis=-1, keepdims=True) + 1e-6
        z_axis_desired = target_thrust / force_norm

        # Desired yaw = 0
        psi_des = np.zeros(z_axis_desired.shape[0])
        x_c_des = np.stack([np.cos(psi_des), np.sin(psi_des), np.zeros_like(psi_des)], axis=-1)

        # Build rotation matrix
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_norm = norm(y_axis_desired, axis=-1, keepdims=True) + 1e-6
        y_axis_desired = y_axis_desired / y_norm

        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_des = np.stack([x_axis_desired, y_axis_desired, z_axis_desired], axis=-1)
        rpy_des = Rotation.from_matrix(R_des).as_euler('xyz', degrees=False)

        return rpy_des, current_thrust

    def _pure_pursuit_control(self, states: np.ndarray, target_pos: np.ndarray,
                               target_vel: np.ndarray) -> np.ndarray:
        """Pure pursuit control.

        Args:
            states: Pursuer states [pos, vel, rpy, active], shape (n, 10).
            target_pos: Target positions, shape (n, 3).
            target_vel: Target velocities, shape (n, 3).

        Returns:
            Control array [roll, pitch, yaw, thrust], shape (n, 4).
        """
        pos_rb = target_pos - states[:, 0:3]
        vel_rb = target_vel - states[:, 3:6]

        accel = self._pure_pursuit(pos_rb, vel_rb)
        rpy_des, thrust_des = self._accel_to_attitude(accel, states[:, 6:9])

        return self._clip_and_pack(rpy_des, thrust_des)

    def _pronav_control(self, states: np.ndarray, target_pos: np.ndarray,
                        target_vel: np.ndarray, target_accel: np.ndarray) -> np.ndarray:
        """Proportional navigation control.

        Args:
            states: Pursuer states [pos, vel, rpy, active], shape (n, 10).
            target_pos: Target positions, shape (n, 3).
            target_vel: Target velocities, shape (n, 3).
            target_accel: Target accelerations, shape (n, 3).

        Returns:
            Control array [roll, pitch, yaw, thrust], shape (n, 4).
        """
        pos_rb = target_pos - states[:, 0:3]
        vel_rb = target_vel - states[:, 3:6]

        accel = self._proportional_nav(pos_rb, vel_rb, target_accel)
        rpy_des, thrust_des = self._accel_to_attitude(accel, states[:, 6:9])

        return self._clip_and_pack(rpy_des, thrust_des)

    def _aug_pronav_control(self, states: np.ndarray, target_pos: np.ndarray,
                             target_vel: np.ndarray, target_accel: np.ndarray) -> np.ndarray:
        """Augmented proportional navigation control.

        Args:
            states: Pursuer states [pos, vel, rpy, active], shape (n, 10).
            target_pos: Target positions, shape (n, 3).
            target_vel: Target velocities, shape (n, 3).
            target_accel: Target accelerations, shape (n, 3).

        Returns:
            Control array [roll, pitch, yaw, thrust], shape (n, 4).
        """
        pos_rb = target_pos - states[:, 0:3]
        vel_rb = target_vel - states[:, 3:6]
        vel_pursuer = states[:, 3:6]

        accel = self._augmented_pronav(pos_rb, vel_rb, vel_pursuer, target_accel)
        rpy_des, thrust_des = self._accel_to_attitude(accel, states[:, 6:9])

        return self._clip_and_pack(rpy_des, thrust_des)

    def _tpn_control(self, states: np.ndarray, target_pos: np.ndarray,
                     target_vel: np.ndarray) -> np.ndarray:
        """True Proportional Navigation control.

        Args:
            states: Pursuer states [pos, vel, rpy, active], shape (n, 10).
            target_pos: Target positions, shape (n, 3).
            target_vel: Target velocities, shape (n, 3).

        Returns:
            Control array [roll, pitch, yaw, thrust], shape (n, 4).
        """
        pos_rb = target_pos - states[:, 0:3]
        vel_rb = target_vel - states[:, 3:6]
        vel_pursuer = states[:, 3:6]

        accel = self._tpn_with_axial(pos_rb, vel_rb, vel_pursuer)
        rpy_des, thrust_des = self._accel_to_attitude(accel, states[:, 6:9])

        return self._clip_and_pack(rpy_des, thrust_des)

    def _pid_control(self, states: np.ndarray, target_pos: np.ndarray,
                     target_vel: np.ndarray, agent_indices: np.ndarray = None,
                     braking: bool = False) -> np.ndarray:
        """PID position control for hover.

        Uses PD control for XY axes and PID control for Z axis to eliminate
        steady-state error in altitude hover.

        Args:
            states: Agent states [pos, vel, rpy, active], shape (n, 10).
            target_pos: Target positions, shape (n, 3).
            target_vel: Target velocities, shape (n, 3).
            agent_indices: Optional indices of agents being controlled. If None,
                assumes all agents [0, 1, ..., n-1]. Required when controlling
                a subset of agents to correctly update per-agent integral state.
            braking: If True, amplify velocity damping gains by
                braking_vel_multiplier for more aggressive deceleration.

        Returns:
            Control array [roll, pitch, yaw, thrust], shape (n, 4).
        """
        import time
        current_time = time.perf_counter()

        n_agents = states.shape[0]
        if agent_indices is None:
            agent_indices = np.arange(n_agents)

        pos_rb = target_pos - states[:, 0:3]
        vel_rb = target_vel - states[:, 3:6]

        # Compute dt for integration
        if self.last_control_time is not None:
            dt = current_time - self.last_control_time
            dt = np.clip(dt, 0.001, 0.1)  # Clamp dt to reasonable range
        else:
            dt = 0.02  # Default ~50 Hz

        self.last_control_time = current_time

        # Update Z error integral only for the agents being controlled
        z_error = pos_rb[:, 2]
        self.z_error_integral[agent_indices] = self.z_error_integral[agent_indices] + z_error * dt
        self.z_error_integral = np.clip(
            self.z_error_integral,
            -self.hover_integral_cap,
            self.hover_integral_cap
        )

        # PD acceleration from pure pursuit
        accel = self._pure_pursuit(pos_rb, vel_rb)

        # Amplify velocity damping for aggressive braking
        if braking:
            extra_vel_accel = (self.braking_vel_multiplier - 1.0) * vel_rb
            accel[:, :2] += self.k_vxy * extra_vel_accel[:, :2]
            accel[:, 2:3] += self.k_vz * extra_vel_accel[:, 2:3]

        # Add integral term to Z acceleration (use indices to get correct integral values)
        accel[:, 2] = accel[:, 2] + self.hover_ki_z * self.z_error_integral[agent_indices]

        rpy_des, thrust_des = self._accel_to_attitude(accel, states[:, 6:9])

        return self._clip_and_pack(rpy_des, thrust_des)

    def _clip_and_pack(self, rpy_des: np.ndarray, thrust_des: np.ndarray) -> np.ndarray:
        """Clip attitude/thrust to limits and pack into control array.

        Args:
            rpy_des: Desired roll-pitch-yaw, shape (n, 3).
            thrust_des: Desired thrust, shape (n,).

        Returns:
            Control array [roll, pitch, yaw, thrust], shape (n, 4).
        """
        roll_des = np.clip(rpy_des[:, 0], -self.roll_pitch_max, self.roll_pitch_max)
        pitch_des = np.clip(rpy_des[:, 1], -self.roll_pitch_max, self.roll_pitch_max)
        yaw_des = np.clip(rpy_des[:, 2], -self.yaw_max, self.yaw_max)
        thrust_clipped = np.clip(thrust_des, self.thrust_min, self.thrust_max)

        return np.stack([roll_des, pitch_des, yaw_des, thrust_clipped], axis=-1)


class EvaderTeam(TeamBase):
    """Blue team (evaders) using learned policy for control.

    Builds 46D observations matching the simulation environment and uses
    a loaded policy (FFN or ACMPC) for action inference.
    """

    def __init__(self, config: dict, policy=None, policy_type: str = "ffn",
                 obs_preprocessor=None):
        """Initialize evader team.

        Args:
            config: Environment configuration dictionary.
            policy: Loaded policy object with compute() method.
            policy_type: Type of policy ("ffn" or "acmpc").
            obs_preprocessor: Optional observation preprocessor from training checkpoint.
        """
        super().__init__('evader_team', 'evader', config)

        self.policy = policy
        self.policy_type = policy_type
        self.obs_preprocessor = obs_preprocessor
        self.pursuer_targets = []
        self.pursuer_target_one_hot = None
        self.initial_pos = None
        self.n_blue = config.get('n_pairs', 2)
        self.n_red = config.get('n_pairs', 2)

        # QoS: best effort, keep last 1 to process latest status only
        status_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(
            Status,
            '/multiagent_pursuit_evasion/status',
            self._status_callback,
            status_qos
        )

        self.get_logger().info(f'Evader team initialized with policy type: {policy_type}')

    def _status_callback(self, msg: Status):
        """Handle status message and compute control."""
        if not self.team_initialized:
            self.initialize_team(msg.evader_cf_names)
            # Use full XYZ initial positions from config
            self.initial_pos = self.blue_initial_pos.copy()

            # Build one-hot encoding for pursuer targets
            self.pursuer_targets = np.array(msg.pursuer_targets)
            self.pursuer_target_one_hot = np.zeros((len(self.pursuer_targets), msg.n_evaders))
            for i, target in enumerate(self.pursuer_targets):
                self.pursuer_target_one_hot[i, target] = 1.0

        # Update targets in case they changed
        self.pursuer_targets = np.array(msg.pursuer_targets)
        for i, target in enumerate(self.pursuer_targets):
            self.pursuer_target_one_hot[i, :] = 0.0
            self.pursuer_target_one_hot[i, target] = 1.0

        blue_states = msg.cf_evader_states
        red_states = msg.cf_pursuer_states
        status = msg.status

        if status == Status.MAPE_STATUS_OFF:
            return

        # Command takeoff when in TAKEOFF status
        if status == Status.MAPE_STATUS_TAKEOFF and not self.takeoff_commanded:
            self.command_takeoff(self.blue_initial_height)
            return

        # Check if ready to switch to low-level control during TAKEOFF
        if status == Status.MAPE_STATUS_TAKEOFF and not self.low_level_enabled:
            blue_states_np = self.states_to_np(blue_states)
            if self.check_ready_for_low_level(blue_states_np, self.blue_initial_height):
                self.switch_to_low_level()
            return  # Don't send attitude commands until switched

        # Only send attitude commands if low-level control is enabled
        if not self.low_level_enabled:
            return

        if not self.initialized_control:
            self.initialized_control = True
            for n in range(self.team_size):
                self.cmd_attitude_setpoint(np.array([0., 0., 0., 0.]), n)

        controls = self.compute_control(blue_states, red_states, status)

        blue_states_np = self.states_to_np(blue_states)
        for n in range(self.team_size):
            current_yaw = blue_states_np[n, 8]
            self.cmd_attitude_setpoint(controls[n, :], n, current_yaw=current_yaw)

    def compute_control(self, blue_states: list, red_states: list, status: int) -> np.ndarray:
        """Compute control for all evaders.

        Args:
            blue_states: List of evader State messages.
            red_states: List of pursuer State messages.
            status: Current game status.

        Returns:
            Control array of shape (n_evaders, 4): [roll, pitch, yaw, thrust].
        """
        blue_states_np = self.states_to_np_extended(blue_states)
        red_states_np = self.states_to_np(red_states)

        if status == Status.MAPE_STATUS_OFF:
            return np.zeros((len(blue_states), 4))

        elif status == Status.MAPE_STATUS_TAKEOFF or status == Status.MAPE_STATUS_INITIALIZED:
            # Use PD control for hover (TAKEOFF included for brief transition period)
            # _pid_control expects 10D: [pos(3), vel(3), rpy(3), active(1)]
            # Extract RPY from rotation matrix for PID controller
            blue_10d = self._extract_pid_states(blue_states_np)
            return self._pid_control(blue_10d, self.initial_pos, np.zeros_like(self.initial_pos))

        elif status == Status.MAPE_STATUS_RUNNING:
            active_agents = (blue_states_np[:, 18] == 1)
            inactive_agents = ~active_agents

            # Reset settled state and integral for agents that become active again
            self.agent_settled[active_agents] = False
            # Reset integral for active agents (using policy, not PID)
            self.z_error_integral[active_agents] = 0.0

            # Update initial pos for active agents
            self.initial_pos[active_agents] = blue_states_np[active_agents, 0:3].copy()

            controls = np.zeros((self.team_size, 4))

            # Handle inactive agents with settling behavior
            if np.any(inactive_agents):
                inactive_indices = np.where(inactive_agents)[0]
                inactive_pos = blue_states_np[inactive_agents, 0:3]
                inactive_vel = blue_states_np[inactive_agents, 3:6]
                inactive_vel_mag = np.linalg.norm(inactive_vel, axis=1)

                # Check which inactive agents have settled (velocity below threshold)
                for i, idx in enumerate(inactive_indices):
                    if not self.agent_settled[idx] and inactive_vel_mag[i] < self.settling_velocity_threshold:
                        self.agent_settled[idx] = True
                        self.settled_pos[idx] = inactive_pos[i]
                        self.get_logger().info(f'BLUE {idx} settled at pos {self.settled_pos[idx]}')

                # Build target positions for inactive agents
                inactive_target_pos = np.zeros_like(inactive_pos)
                # Track which inactive agents need braking (not yet settled)
                needs_braking = np.zeros(len(inactive_indices), dtype=bool)
                for i, idx in enumerate(inactive_indices):
                    if self.agent_settled[idx]:
                        # Use settled position for hover
                        inactive_target_pos[i] = self.settled_pos[idx]
                    else:
                        # Use deactivation position (last active frame) as fixed reference
                        # so the PID has position error to maintain altitude
                        inactive_target_pos[i] = self.initial_pos[idx]
                        needs_braking[i] = True

                # Settled agents: gentle PID hover
                if np.any(~needs_braking):
                    settled_mask = ~needs_braking
                    settled_idx = inactive_indices[settled_mask]
                    blue_10d = self._extract_pid_states(blue_states_np[settled_idx])
                    controls_settled = self._pid_control(
                        blue_10d, inactive_target_pos[settled_mask],
                        np.zeros_like(inactive_target_pos[settled_mask]),
                        agent_indices=settled_idx
                    )
                    controls[settled_idx, :] = controls_settled

                # Unsettled agents: aggressive braking
                if np.any(needs_braking):
                    braking_idx = inactive_indices[needs_braking]
                    blue_10d = self._extract_pid_states(blue_states_np[braking_idx])
                    controls_braking = self._pid_control(
                        blue_10d, inactive_target_pos[needs_braking],
                        np.zeros_like(inactive_target_pos[needs_braking]),
                        agent_indices=braking_idx,
                        braking=True
                    )
                    controls[braking_idx, :] = controls_braking

            if np.any(active_agents) and self.policy is not None:
                controls_active = self._policy_control(
                    blue_states_np,
                    red_states_np,
                    active_agents
                )
                controls[active_agents, :] = controls_active
                # self.get_logger().info(f'policy control {controls}')

            return controls

        elif status == Status.MAPE_STATUS_BLUE_WON or status == Status.MAPE_STATUS_RED_WON:
            # Game over - hover at current/settled position
            hover_pos = np.zeros_like(blue_states_np[:, 0:3])
            for i in range(self.team_size):
                if self.agent_settled[i]:
                    hover_pos[i] = self.settled_pos[i]
                else:
                    hover_pos[i] = blue_states_np[i, 0:3]
                    # Mark as settled if not already
                    vel_mag = np.linalg.norm(blue_states_np[i, 3:6])
                    if vel_mag < self.settling_velocity_threshold:
                        self.agent_settled[i] = True
                        self.settled_pos[i] = blue_states_np[i, 0:3]
            blue_10d = self._extract_pid_states(blue_states_np)
            return self._pid_control(blue_10d, hover_pos, np.zeros_like(hover_pos))

        return np.zeros((len(blue_states), 4))

    def _extract_pid_states(self, blue_states_ext: np.ndarray) -> np.ndarray:
        """Extract 10D PID states from 19D extended blue states.

        Converts rotation matrix back to RPY for the PID controller.

        Args:
            blue_states_ext: shape (n, 19) = [pos(3), vel(3), rotmat_flat(9), body_rates(3), active(1)]

        Returns:
            Array of shape (n, 10) = [pos(3), vel(3), rpy(3), active(1)]
        """
        n = blue_states_ext.shape[0]
        pid_states = np.zeros((n, 10), dtype=np.float32)
        pid_states[:, 0:3] = blue_states_ext[:, 0:3]   # pos
        pid_states[:, 3:6] = blue_states_ext[:, 3:6]   # vel
        # Convert rotation matrix to RPY
        for i in range(n):
            R = blue_states_ext[i, 6:15].reshape(3, 3)
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            pitch = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pid_states[i, 6:9] = [roll, pitch, yaw]
        pid_states[:, 9] = blue_states_ext[:, 18]       # active
        return pid_states

    def _build_observation(self, blue_states: np.ndarray, red_states: np.ndarray,
                            agent_idx: int) -> np.ndarray:
        """Build observation for a single agent matching env._get_observations().

        Observation format (for n_pairs=2):
        - Own state: pos(3) + vel(3) + rotmat_flat(9) + body_rates(3) = 18
        - Own one-hot: n_blue = 2
        - All blue: n_blue * (pos(3) + vel(3) + alive(1)) = 14
        - All red: n_red * (pos(3) + vel(3) + alive(1)) = 14
        - Target assignments: n_red * n_blue = 4

        Total: 18 + 2 + 14 + 14 + 4 = 52D

        Args:
            blue_states: Blue agent states, shape (n_blue, 19).
                [pos(3), vel(3), rotmat_flat(9), body_rates(3), active(1)]
            red_states: Red agent states, shape (n_red, 10).
                [pos(3), vel(3), rpy(3), active(1)]
            agent_idx: Index of the agent building observation for.

        Returns:
            Observation array of shape (52,).
        """
        n_blue = blue_states.shape[0]
        n_red = red_states.shape[0]

        blue_alive = blue_states[:, 18]
        red_alive = red_states[:, 9]

        # Own state (18D): pos + vel + rotmat_flat + body_rates
        own_state = blue_states[agent_idx, 0:18]

        # Own one-hot masked by alive (n_blue D)
        own_one_hot = np.zeros(n_blue)
        if blue_alive[agent_idx] > 0.5:
            own_one_hot[agent_idx] = 1.0

        # All blue states masked by alive (n_blue * 7 D)
        blue_states_flat = []
        for b in range(n_blue):
            alive_mask = blue_alive[b]
            blue_states_flat.extend([
                blue_states[b, 0] * alive_mask,  # x
                blue_states[b, 1] * alive_mask,  # y
                blue_states[b, 2] * alive_mask,  # z
                blue_states[b, 3] * alive_mask,  # vx
                blue_states[b, 4] * alive_mask,  # vy
                blue_states[b, 5] * alive_mask,  # vz
                alive_mask,                       # alive
            ])

        # All red states masked by alive (n_red * 7 D)
        red_states_flat = []
        for r in range(n_red):
            alive_mask = red_alive[r]
            red_states_flat.extend([
                red_states[r, 0] * alive_mask,  # x
                red_states[r, 1] * alive_mask,  # y
                red_states[r, 2] * alive_mask,  # z
                red_states[r, 3] * alive_mask,  # vx
                red_states[r, 4] * alive_mask,  # vy
                red_states[r, 5] * alive_mask,  # vz
                alive_mask,                      # alive
            ])

        # Target assignments masked by red alive (n_red * n_blue D)
        target_one_hot_masked = (self.pursuer_target_one_hot * red_alive[:, None]).flatten()

        obs = np.concatenate([
            own_state,                          # 18
            own_one_hot,                        # n_blue
            np.array(blue_states_flat),         # n_blue * 7
            np.array(red_states_flat),          # n_red * 7
            target_one_hot_masked,              # n_red * n_blue
        ])

        return obs.astype(np.float32)

    def _policy_control(self, blue_states: np.ndarray, red_states: np.ndarray,
                         active_agents: np.ndarray) -> np.ndarray:
        """Compute control using learned policy.

        Args:
            blue_states: Blue agent states, shape (n_blue, 19).
            red_states: Red agent states, shape (n_red, 10).
            active_agents: Boolean mask of active agents, shape (n_blue,).

        Returns:
            Control array for active agents, shape (n_active, 4).
        """
        import torch

        active_indices = np.where(active_agents)[0]
        n_active = len(active_indices)
        n = self.n_blue
        obs_dim = 18 + n + n * 7 + n * 7 + n * n

        # Build observations for active agents
        observations = np.zeros((n_active, obs_dim), dtype=np.float32)
        for i, agent_idx in enumerate(active_indices):
            observations[i] = self._build_observation(blue_states, red_states, agent_idx)

        # Run policy inference (deterministic, no sampling)
        with torch.no_grad():
            obs_tensor = torch.tensor(observations, dtype=torch.float32, device='cpu')

            # Build raw MPC state [pos(3), rpy(3), vel(3), drpy(3)] from blue_states
            # for ACMPC policies that need physical values (not normalized)
            mpc_state = np.zeros((n_active, 12), dtype=np.float32)
            for i, agent_idx in enumerate(active_indices):
                mpc_state[i, 0:3] = blue_states[agent_idx, 0:3]   # pos
                # Convert rotation matrix to RPY
                R = blue_states[agent_idx, 6:15].reshape(3, 3)
                sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
                pitch = np.arctan2(-R[2, 0], sy)
                roll = np.arctan2(R[2, 1], R[2, 2])
                yaw = np.arctan2(R[1, 0], R[0, 0])
                mpc_state[i, 3:6] = [roll, pitch, yaw]            # rpy
                mpc_state[i, 6:9] = blue_states[agent_idx, 3:6]   # vel
                mpc_state[i, 9:12] = blue_states[agent_idx, 15:18] # body_rates (drpy)
            mpc_state_tensor = torch.tensor(mpc_state, dtype=torch.float32, device='cpu')

            # Apply observation preprocessor (normalization) if available
            if self.obs_preprocessor is not None:
                obs_tensor = self.obs_preprocessor(obs_tensor)

            inputs = {"observations": obs_tensor}
            inputs["mpc_state"] = mpc_state_tensor
            mean_actions, _ = self.policy.compute(inputs)

            if isinstance(mean_actions, torch.Tensor):
                normalized_actions = mean_actions.numpy()
            else:
                normalized_actions = mean_actions

        # Denormalize actions
        # Policy outputs normalized [-1, 1], physical = mean + normalized * scale
        thrust_mean = (self.thrust_min + self.thrust_max) / 2.0
        thrust_scale = (self.thrust_max - self.thrust_min) / 2.0

        action_mean = np.array([0.0, 0.0, 0.0, thrust_mean])
        action_scale = np.array([self.roll_pitch_max, self.roll_pitch_max, self.yaw_max, thrust_scale])

        physical_actions = action_mean + normalized_actions * action_scale

        # Clip to bounds
        physical_actions[:, 0] = np.clip(physical_actions[:, 0], -self.roll_pitch_max, self.roll_pitch_max)
        physical_actions[:, 1] = np.clip(physical_actions[:, 1], -self.roll_pitch_max, self.roll_pitch_max)
        physical_actions[:, 2] = np.clip(physical_actions[:, 2], -self.yaw_max, self.yaw_max)
        physical_actions[:, 3] = np.clip(physical_actions[:, 3], self.thrust_min, self.thrust_max)

        return physical_actions

    def _pid_control(self, states: np.ndarray, target_pos: np.ndarray,
                     target_vel: np.ndarray, agent_indices: np.ndarray = None) -> np.ndarray:
        """PID position control for hover.

        Uses PD control for XY axes and PID control for Z axis to eliminate
        steady-state error in altitude hover.

        Args:
            states: Agent states [pos, vel, rpy, active], shape (n, 10).
            target_pos: Target positions, shape (n, 3).
            target_vel: Target velocities, shape (n, 3).
            agent_indices: Optional indices of agents being controlled. If None,
                assumes all agents [0, 1, ..., n-1]. Required when controlling
                a subset of agents to correctly update per-agent integral state.

        Returns:
            Control array [roll, pitch, yaw, thrust], shape (n, 4).
        """
        import time
        current_time = time.perf_counter()

        n_agents = states.shape[0]
        if agent_indices is None:
            agent_indices = np.arange(n_agents)

        k_pxy = self.config.get('pursuit_gains', {}).get('pp_k_pxy', 6.1624)
        k_vxy = self.config.get('pursuit_gains', {}).get('pp_k_vxy', 3.39)
        k_pz = self.config.get('pursuit_gains', {}).get('pp_k_pz', 20.0)
        k_vz = self.config.get('pursuit_gains', {}).get('pp_k_vz', 10.0)

        pos_rb = target_pos - states[:, 0:3]
        vel_rb = target_vel - states[:, 3:6]

        # Compute dt for integration
        if self.last_control_time is not None:
            dt = current_time - self.last_control_time
            dt = np.clip(dt, 0.001, 0.1)  # Clamp dt to reasonable range
        else:
            dt = 0.02  # Default ~50 Hz

        self.last_control_time = current_time

        # Update Z error integral only for the agents being controlled
        z_error = pos_rb[:, 2]
        self.z_error_integral[agent_indices] = self.z_error_integral[agent_indices] + z_error * dt
        self.z_error_integral = np.clip(
            self.z_error_integral,
            -self.hover_integral_cap,
            self.hover_integral_cap
        )

        accel_xy = k_pxy * pos_rb[:, :2] + k_vxy * vel_rb[:, :2]
        accel_z = k_pz * pos_rb[:, 2:3] + k_vz * vel_rb[:, 2:3] + self.gravity
        # Add integral term to Z acceleration (use indices to get correct integral values)
        accel_z = accel_z + self.hover_ki_z * self.z_error_integral[agent_indices, None]

        accel = np.concatenate([accel_xy, accel_z], axis=-1)

        # Acceleration to attitude
        target_thrust = accel * self.mass
        R = Rotation.from_euler('xyz', states[:, 6:9], degrees=False).as_matrix()
        z_axis = R[:, :, 2]
        current_thrust = np.sum(target_thrust * z_axis, axis=-1)

        force_norm = norm(target_thrust, axis=-1, keepdims=True) + 1e-6
        z_axis_desired = target_thrust / force_norm

        psi_des = np.zeros(z_axis_desired.shape[0])
        x_c_des = np.stack([np.cos(psi_des), np.sin(psi_des), np.zeros_like(psi_des)], axis=-1)

        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_norm = norm(y_axis_desired, axis=-1, keepdims=True) + 1e-6
        y_axis_desired = y_axis_desired / y_norm

        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_des = np.stack([x_axis_desired, y_axis_desired, z_axis_desired], axis=-1)
        rpy_des = Rotation.from_matrix(R_des).as_euler('xyz', degrees=False)

        roll_des = np.clip(rpy_des[:, 0], -self.roll_pitch_max, self.roll_pitch_max)
        pitch_des = np.clip(rpy_des[:, 1], -self.roll_pitch_max, self.roll_pitch_max)
        yaw_des = np.clip(rpy_des[:, 2], -self.yaw_max, self.yaw_max)
        thrust_clipped = np.clip(current_thrust, self.thrust_min, self.thrust_max)

        return np.stack([roll_des, pitch_des, yaw_des, thrust_clipped], axis=-1)
