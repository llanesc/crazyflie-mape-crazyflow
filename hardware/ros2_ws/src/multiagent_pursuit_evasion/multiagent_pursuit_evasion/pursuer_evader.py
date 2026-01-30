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

from crazyflie_interfaces.msg import AttitudeSetpoint
from multiagent_pursuit_evasion_interfaces.msg import Status, State
from scipy.spatial.transform import Rotation
from drone_models.core import load_params


class TeamBase(Node):
    """Base class for pursuer and evader teams.

    Provides common functionality for loading drone parameters, converting
    thrust to PWM, and publishing attitude setpoints.
    """

    def __init__(self, team_name: str, config: dict):
        """Initialize team base.

        Args:
            team_name: ROS node name.
            config: Environment configuration dictionary.
        """
        super().__init__(team_name)

        self.config = config

        # Load physical parameters from drone-models
        drone_model = config.get('drone_model', 'cf2x_T350')
        drone_params = load_params("so_rpy", drone_model)

        self.mass = float(drone_params["mass"])
        self.gravity = float(np.abs(drone_params["gravity_vec"][2]))
        self.min_thrust = float(drone_params["thrust_min"]) * 4  # Per motor -> collective
        self.max_thrust = float(drone_params["thrust_max"]) * 4

        self.blue_initial_height = [1.0, 1.0]
        # self.red_initial_height = [0.67, 0.80]
        self.red_initial_height = [0.91, 0.98]

        # Attitude limits
        self.roll_pitch_max = config.get('roll_pitch_max', 0.5)  # rad
        self.yaw_max = config.get('yaw_max', 0.1)  # rad

        # Thrust-to-PWM parameters (linear mapping matching Mellinger controller)
        self.pwm_max = 65535

        self.team_size = None
        self.cf_list = []
        self.attitude_setpoint_publishers = None
        self.team_initialized = False
        self.initialized_control = False

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

    def cmd_attitude_setpoint(self, control: np.ndarray, index: int):
        """Publish attitude setpoint command.

        Args:
            control: [roll_rad, pitch_rad, yaw_rad, thrust_N] in physical units.
            index: Agent index in team.
        """
        roll_rad, pitch_rad, yaw_rad, thrust_N = control.ravel()

        setpoint = AttitudeSetpoint()
        setpoint.roll = roll_rad
        setpoint.pitch = pitch_rad
        # Yaw treated as yaw_rate for hardware (degrees/s)
        setpoint.yaw_rate = 0
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
        collective_thrust = np.clip(collective_thrust, 0, self.max_thrust)

        # Linear mapping: pwm = (thrust / thrust_max) * pwm_max
        pwm = (collective_thrust / self.max_thrust) * self.pwm_max

        return int(np.clip(pwm, 0, self.pwm_max))

    def states_to_np(self, states: list) -> np.ndarray:
        """Convert State list to numpy array (10D).

        Returns:
            Array of shape (n_agents, 10): [pos(3), vel(3), rpy(3), active(1)]
        """
        np_states = np.zeros((len(states), 10))
        for idx, state in enumerate(states):
            np_states[idx, :] = np.array([
                *state.position,
                *state.velocity,
                *state.attitude,
                float(state.active)
            ])
        return np_states

    def states_to_np_extended(self, states: list) -> np.ndarray:
        """Convert State list to numpy array with angular velocity (13D).

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
    """Red team (pursuers) using Pure Pursuit, ProNav, or Augmented ProNav.

    Control logic matches the simulation environment's pursuer strategies.
    """

    def __init__(self, config: dict):
        """Initialize pursuer team.

        Args:
            config: Environment configuration dictionary.
        """
        super().__init__('pursuer_team', config)

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
            initial_pos = self.states_to_np(msg.cf_pursuer_states)[:, 0:3]
            initial_pos[:, 2] = self.red_initial_height  # Set initial hover height
            self.initial_pos = initial_pos

        self.pursuer_targets = list(msg.pursuer_targets)
        blue_states = msg.cf_evader_states
        red_states = msg.cf_pursuer_states
        status = msg.status

        if status == Status.MAPE_STATUS_OFF:
            return

        if not self.initialized_control:
            self.initialized_control = True
            for n in range(self.team_size):
                self.cmd_attitude_setpoint(np.array([0., 0., 0., 0.]), n)

        controls = self.compute_control(red_states, blue_states, status)

        for n in range(self.team_size):
            self.cmd_attitude_setpoint(controls[n, :], n)

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

        elif status == Status.MAPE_STATUS_INITIALIZED:
            # Hover at initial position
            return self._pd_control(red_states_np, self.initial_pos, np.zeros_like(self.initial_pos))

        elif status == Status.MAPE_STATUS_RUNNING:
            # Get target positions and velocities
            target_pos = blue_states_np[self.pursuer_targets, 0:3].copy()
            target_vel = blue_states_np[self.pursuer_targets, 3:6].copy()

            active_agents = (red_states_np[:, -1] == 1)

            # Update initial pos for active agents (for deactivated hover reference)
            self.initial_pos[active_agents] = red_states_np[active_agents, 0:3].copy()

            # For inactive agents, hover slightly above last position
            target_pos[~active_agents] = self.initial_pos[~active_agents]
            target_pos[~active_agents, 2] += 0.1
            target_vel[~active_agents] = 0.

            controls = np.zeros((self.team_size, 4))

            # Get blue acceleration for ProNav feedforward (measured from accel topic)
            blue_accel = self.get_acceleration(blue_states)
            target_accel = blue_accel[self.pursuer_targets, :].copy()
            target_accel[~active_agents] = 0.

            if np.any(~active_agents):
                controls_inactive = self._pd_control(
                    red_states_np[~active_agents, :],
                    target_pos[~active_agents],
                    np.zeros_like(target_pos[~active_agents])
                )
                controls[~active_agents, :] = controls_inactive

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
                else:  # ProNav (default)
                    controls_active = self._pronav_control(
                        red_states_np[active_agents, :],
                        target_pos[active_agents],
                        target_vel[active_agents],
                        target_accel[active_agents]
                    )
                controls[active_agents, :] = controls_active

            return controls

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

    def _pd_control(self, states: np.ndarray, target_pos: np.ndarray,
                    target_vel: np.ndarray) -> np.ndarray:
        """PD position control for hover.

        Args:
            states: Agent states [pos, vel, rpy, active], shape (n, 10).
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
        thrust_clipped = np.clip(thrust_des, self.min_thrust, self.max_thrust)

        return np.stack([roll_des, pitch_des, yaw_des, thrust_clipped], axis=-1)


class EvaderTeam(TeamBase):
    """Blue team (evaders) using learned policy for control.

    Builds 46D observations matching the simulation environment and uses
    a loaded policy (FFN or ACMPC) for action inference.
    """

    def __init__(self, config: dict, policy=None, policy_type: str = "ffn"):
        """Initialize evader team.

        Args:
            config: Environment configuration dictionary.
            policy: Loaded policy object with compute() method.
            policy_type: Type of policy ("ffn" or "acmpc").
        """
        super().__init__('evader_team', config)

        self.policy = policy
        self.policy_type = policy_type
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
            initial_pos = self.states_to_np(msg.cf_evader_states)[:, 0:3]
            initial_pos[:, 2] = self.blue_initial_height  # Set initial hover height
            self.initial_pos = initial_pos

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

        if not self.initialized_control:
            self.initialized_control = True
            for n in range(self.team_size):
                self.cmd_attitude_setpoint(np.array([0., 0., 0., 0.]), n)

        controls = self.compute_control(blue_states, red_states, status)

        for n in range(self.team_size):
            self.cmd_attitude_setpoint(controls[n, :], n)

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

        elif status == Status.MAPE_STATUS_INITIALIZED:
            # Use PD control for hover during initialization
            blue_10d = np.concatenate([blue_states_np[:, :9], blue_states_np[:, 12:13]], axis=-1)
            return self._pd_control(blue_10d, self.initial_pos, np.zeros_like(self.initial_pos))

        elif status == Status.MAPE_STATUS_RUNNING:
            active_agents = (blue_states_np[:, 12] == 1)

            # Update initial pos for active agents
            self.initial_pos[active_agents] = blue_states_np[active_agents, 0:3].copy()

            controls = np.zeros((self.team_size, 4))

            if np.any(~active_agents):
                target_pos = self.initial_pos[~active_agents].copy()
                target_pos[:, 2] += 0.1  # Hover slightly higher
                blue_10d = np.concatenate([blue_states_np[~active_agents, :9],
                                           blue_states_np[~active_agents, 12:13]], axis=-1)
                controls_inactive = self._pd_control(blue_10d, target_pos, np.zeros_like(target_pos))
                controls[~active_agents, :] = controls_inactive

            if np.any(active_agents) and self.policy is not None:
                controls_active = self._policy_control(
                    blue_states_np,
                    red_states_np,
                    active_agents
                )
                controls[active_agents, :] = controls_active
                # self.get_logger().info(f'policy control {controls}')

            return controls

        return np.zeros((len(blue_states), 4))

    def _build_observation(self, blue_states: np.ndarray, red_states: np.ndarray,
                            agent_idx: int) -> np.ndarray:
        """Build 46D observation for a single agent matching env._get_observations().

        Observation format (for n_pairs=2):
        - Own state: pos(3) + vel(3) + rpy(3) + rpy_rates(3) = 12
        - Own one-hot: n_blue = 2
        - All blue: n_blue * (pos(3) + vel(3) + alive(1)) = 14
        - All red: n_red * (pos(3) + vel(3) + alive(1)) = 14
        - Target assignments: n_red * n_blue = 4

        Total: 12 + 2 + 14 + 14 + 4 = 46D

        Args:
            blue_states: Blue agent states, shape (n_blue, 13).
            red_states: Red agent states, shape (n_red, 10).
            agent_idx: Index of the agent building observation for.

        Returns:
            Observation array of shape (46,).
        """
        n_blue = blue_states.shape[0]
        n_red = red_states.shape[0]

        blue_alive = blue_states[:, 12]
        red_alive = red_states[:, 9]

        # Own state (12D)
        pos = blue_states[agent_idx, 0:3]
        vel = blue_states[agent_idx, 3:6]
        rpy = blue_states[agent_idx, 6:9]
        ang_vel = blue_states[agent_idx, 9:12]
        # ang_vel[1] = -ang_vel[1]  # Negate yaw rate to match env convention
        # Convert body angular velocity to Euler rates
        rpy_rates = self.ang_vel_to_rpy_rates(rpy[None, :], ang_vel[None, :])[0]

        own_state = np.concatenate([pos, vel, rpy, rpy_rates])  # 12D

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
            own_state,                          # 12
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
            blue_states: Blue agent states, shape (n_blue, 13).
            red_states: Red agent states, shape (n_red, 10).
            active_agents: Boolean mask of active agents, shape (n_blue,).

        Returns:
            Control array for active agents, shape (n_active, 4).
        """
        import torch

        active_indices = np.where(active_agents)[0]
        n_active = len(active_indices)

        # Build observations for active agents
        observations = np.zeros((n_active, 46), dtype=np.float32)
        for i, agent_idx in enumerate(active_indices):
            observations[i] = self._build_observation(blue_states, red_states, agent_idx)

        # Run policy inference (deterministic, no sampling)
        with torch.no_grad():
            obs_tensor = torch.tensor(observations, dtype=torch.float32, device='cpu')
            inputs = {"states": obs_tensor}
            mean_actions, _, _ = self.policy.compute(inputs)

            if isinstance(mean_actions, torch.Tensor):
                normalized_actions = mean_actions.numpy()
            else:
                normalized_actions = mean_actions

        # Denormalize actions
        # Policy outputs normalized [-1, 1], physical = mean + normalized * scale
        thrust_mean = (self.min_thrust + self.max_thrust) / 2.0
        thrust_scale = (self.max_thrust - self.min_thrust) / 2.0

        action_mean = np.array([0.0, 0.0, 0.0, thrust_mean])
        action_scale = np.array([self.roll_pitch_max, self.roll_pitch_max, self.yaw_max, thrust_scale])

        physical_actions = action_mean + normalized_actions * action_scale

        # Clip to bounds
        physical_actions[:, 0] = np.clip(physical_actions[:, 0], -self.roll_pitch_max, self.roll_pitch_max)
        physical_actions[:, 1] = np.clip(physical_actions[:, 1], -self.roll_pitch_max, self.roll_pitch_max)
        physical_actions[:, 2] = np.clip(physical_actions[:, 2], -self.yaw_max, self.yaw_max)
        physical_actions[:, 3] = np.clip(physical_actions[:, 3], self.min_thrust, self.max_thrust)

        return physical_actions

    def _pd_control(self, states: np.ndarray, target_pos: np.ndarray,
                    target_vel: np.ndarray) -> np.ndarray:
        """PD position control for hover (same as PursuerTeam)."""
        k_pxy = self.config.get('pursuit_gains', {}).get('pp_k_pxy', 6.1624)
        k_vxy = self.config.get('pursuit_gains', {}).get('pp_k_vxy', 3.39)
        k_pz = self.config.get('pursuit_gains', {}).get('pp_k_pz', 20.0)
        k_vz = self.config.get('pursuit_gains', {}).get('pp_k_vz', 10.0)

        pos_rb = target_pos - states[:, 0:3]
        vel_rb = target_vel - states[:, 3:6]

        accel_xy = k_pxy * pos_rb[:, :2] + k_vxy * vel_rb[:, :2]
        accel_z = k_pz * pos_rb[:, 2:3] + k_vz * vel_rb[:, 2:3] + self.gravity

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
        thrust_clipped = np.clip(current_thrust, self.min_thrust, self.max_thrust)

        return np.stack([roll_des, pitch_des, yaw_des, thrust_clipped], axis=-1)
