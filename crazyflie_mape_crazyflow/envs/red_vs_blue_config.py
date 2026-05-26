"""Configuration dataclass for Red vs Blue environment."""

from dataclasses import dataclass, field

import numpy as np
from drone_models.core import load_params

@dataclass
class RedVsBlueEnvConfig:
    """Configuration for Red vs Blue pursuit-evasion environment.

    Multi-agent environment where blue evaders use learned policies
    and red pursuers use scripted control.

    Attributes:
        n_pairs: Number of pursuer-evader pairs (configurable).
        n_worlds: Number of parallel environment instances.
        sim_freq: Physics simulation frequency [Hz].
        mellinger_freq: Pursuer Mellinger controller frequency [Hz].
        control_freq: Control frequency [Hz] (env step frequency).
        episode_length_s: Maximum episode duration [s].

        drone_model: Drone model identifier for parameters.
        pursuer_strategy: Red team pursuit strategy ("PP" or "ProNav").

        mass: Simulation drone mass [kg] (loaded from drone_model).
        gravity: Gravity magnitude [m/s^2] (loaded from drone_model).
        thrust_min: Minimum collective thrust [N] (loaded from drone_model).
        thrust_max: Maximum collective thrust [N] (loaded from drone_model).

        bb_collision_tolerance: Blue-blue collision distance [m].
        rr_collision_tolerance: Red-red collision distance [m].
        rb_collision_tolerance: Blue-red capture distance [m].

        boundary_size: Arena half-size [m].
        min_altitude: Minimum altitude [m].
        max_altitude: Maximum altitude [m].

        N_pronav_fb: ProNav feedback gain.
        N_pronav_ff: ProNav feedforward gain.
        velocity_closure_threshold: Velocity closure threshold for ProNav fallback.

        pp_k_pxy: Pure pursuit position gain (x/y).
        pp_k_vxy: Pure pursuit velocity gain (x/y).
        pp_k_pz: Pure pursuit position gain (z).
        pp_k_vz: Pure pursuit velocity gain (z).

        reward_capture: Reward when blue is captured (negative).
        reward_red_crash: Reward when red crashes (positive for blue).
        reward_blue_crash: Reward for blue-blue collision (negative).
        reward_boundary: Reward for boundary violation (negative).
        reward_pursuer_proximity: Reward when pursuers are close to each other.
        reward_pursuer_proximity_decay: Exponential decay rate for pursuer proximity reward.

        device: JAX device ("cpu" or "cuda").

    Note:
        Spawn configuration is handled separately via spawn functions.
        See crazyflie_mape_crazyflow.envs.spawn for spawn function factories.
    """

    # Multi-agent settings
    n_pairs: int = 2
    n_blue: int = field(init=False)
    n_red: int = field(init=False)
    n_drones: int = field(init=False)

    # Simulation settings
    n_worlds: int = 256
    sim_freq: int = 500  # Physics simulation frequency [Hz]
    mellinger_freq: int = 500  # Internal Mellinger controller [Hz]
    control_freq: int = 100  # MPC and pursuer control frequency [Hz]
    episode_length_s: float = 20.0

    # Control settings
    drone_model: str = "cf2x_L250"
    pursuer_strategy: str = "ProNav"  # "PP" or "ProNav"

    # Physical parameters (computed from drone_model in __post_init__, can be overridden)
    mass: float | None = None  # Simulation mass [kg] for all drones, if None loaded from drone_model
    blue_mass: float | list[float] | None = None  # Override mass [kg] for blue/evader drones only (eval). Scalar=all blue, list=per-drone. None = use mass.
    gravity: float = field(init=False)
    thrust_min: float | None = None  # If None, computed as mass * gravity * 0.5
    thrust_max: float | None = None  # If None, computed as mass * gravity * 1.5

    # Domain randomization
    randomize_mass: bool = False  # Randomize mass with normal distribution
    randomize_inertia: bool = False  # Randomize inertia with normal distribution
    mass_randomization_std: float = 2e-3  # Standard deviation for mass randomization [kg]
    inertia_randomization_std: float = 3e-6  # Standard deviation for inertia randomization [kg*m^2]

    # Disturbance forces/torques
    enable_disturbance: bool = False  # Enable random disturbance forces and torques
    disturbance_force_std: float = 0.01  # Standard deviation for force disturbance [N]
    disturbance_torque_std: float = 1e-4  # Standard deviation for torque disturbance [Nm]

    # Collision tolerances
    bb_collision_tolerance: float = 0.2
    rr_collision_tolerance: float = 0.2
    rb_collision_tolerance: float = 0.2

    # Boundary settings
    boundary_size: float = 3.0
    min_altitude: float = 0.1
    max_altitude: float = 3.0

    # ProNav parameters (used by "ProNav" strategy)
    N_pronav_fb: float = 5.0
    N_pronav_ff: float = 1.0
    velocity_closure_threshold: float = 0.5

    # Augmented ProNav parameters (used by "AugProNav" strategy)
    N_gain: float = 3.0  # Navigation constant
    V_min: float = 0.5  # Minimum speed floor [m/s]
    K_v: float = 2.5  # Speed floor gain

    # Pure pursuit gains
    pp_k_pxy: float = 6.1624
    pp_k_vxy: float = 3.39
    pp_k_pz: float = 20.0
    pp_k_vz: float = 10.0

    # Attitude limits
    roll_pitch_max: float = 0.2
    yaw_max: float = 0.1

    # Reward scales
    reward_capture: float = -30.0
    reward_red_crash: float = 20.0
    reward_blue_crash: float = -20.0
    reward_boundary: float = -5.0
    reward_pursuer_proximity: float = 0.5  # Reward evaders when pursuers are close to each other
    reward_pursuer_proximity_decay: float = 2.0  # Exponential decay rate for pursuer proximity reward
    reward_rr_relative_velocity_coef: float = 0.0  # Bonus for red-red collisions scaled by relative velocity

    # Angle penalty (penalizes orientation deviation from level flight)
    reward_angle_coef: float = 0.04  # Coefficient for ||rpy|| penalty

    # Action penalties (energy and smoothness)
    reward_action_coef: float = 0.04  # Coefficient for thrust² penalty
    reward_action_smoothness_thrust: float = 0.4  # Coefficient for (Δthrust)² penalty
    reward_action_smoothness_rpy: float = 1.0  # Coefficient for (Δroll² + Δpitch² + Δyaw²) penalty

    # Velocity penalty (penalizes high speeds)
    reward_velocity_coef: float = 0.0  # Coefficient for ||velocity||² penalty

    # Ground proximity penalty (penalizes flying too close to ground)
    reward_ground_proximity_coef: float = 0.0  # Coefficient for ground proximity penalty
    reward_ground_proximity_decay: float = 10.0  # Exponential decay rate (higher = sharper transition)

    # Target assignment
    random_target_assignment: bool = False  # Randomize red->blue target assignment at reset

    # MPC state representation
    mpc_state_type: str = "euler"  # "quat" (13D) or "euler" (12D RPY+drpy)

    # Device
    device: str = "cpu"

    def __post_init__(self):
        """Compute derived values and load physical parameters."""
        self.n_blue = self.n_pairs
        self.n_red = self.n_pairs
        self.n_drones = self.n_blue + self.n_red

        # Load physical parameters from drone-models
        drone_params = load_params("first_principles", self.drone_model)
        self.drone_model_mass = float(drone_params["mass"])
        # mass: base mass for all drones (red always uses this)
        if self.mass is None:
            self.mass = self.drone_model_mass
        # blue_mass: optional override for blue/evader drones only (e.g. eval)
        if self.blue_mass is None:
            self.blue_mass = self.mass
        self.gravity = float(np.abs(drone_params["gravity_vec"][2]))
        # Use user-provided thrust limits or compute from mass/gravity
        if self.thrust_min is None:
            self.thrust_min = self.mass * self.gravity * 0.5
        if self.thrust_max is None:
            self.thrust_max = self.mass * self.gravity * 1.5

    @property
    def sim_steps_per_mellinger(self) -> int:
        """Number of sim steps per Mellinger update."""
        return self.sim_freq // self.mellinger_freq

    @property
    def sim_steps_per_control(self) -> int:
        """Number of sim steps per control update (env step)."""
        return self.sim_freq // self.control_freq

    @property
    def max_episode_steps(self) -> int:
        """Maximum number of environment steps per episode."""
        return int(self.episode_length_s * self.control_freq)

    @property
    def dt(self) -> float:
        """Environment timestep [s]."""
        return 1.0 / self.control_freq
