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

        mass: Drone mass [kg] (loaded from drone_model).
        gravity: Gravity magnitude [m/s^2] (loaded from drone_model).
        min_thrust: Minimum collective thrust [N] (loaded from drone_model).
        max_thrust: Maximum collective thrust [N] (loaded from drone_model).

        bb_crash_tolerance: Blue-blue collision distance [m].
        rr_crash_tolerance: Red-red collision distance [m].
        br_crash_tolerance: Blue-red capture distance [m].

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
        reward_escape: Reward when blue survives episode (positive).
        reward_red_crash: Reward when red crashes (positive for blue).
        reward_blue_crash: Reward for blue-blue collision (negative).
        reward_boundary: Reward for boundary violation (negative).
        reward_alive: Per-step survival bonus.
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

    # Physical parameters (computed from drone_model in __post_init__)
    mass: float = field(init=False)
    gravity: float = field(init=False)
    min_thrust: float = field(init=False)
    max_thrust: float = field(init=False)

    # Collision tolerances
    bb_crash_tolerance: float = 0.2
    rr_crash_tolerance: float = 0.2
    br_crash_tolerance: float = 0.2

    # Boundary settings
    boundary_size: float = 3.0
    min_altitude: float = 0.1
    max_altitude: float = 3.0

    # ProNav parameters
    N_pronav_fb: float = 5.0
    N_pronav_ff: float = 1.0
    velocity_closure_threshold: float = 0.5

    # Pure pursuit gains
    pp_k_pxy: float = 6.1624
    pp_k_vxy: float = 3.39
    pp_k_pz: float = 20.0
    pp_k_vz: float = 10.0

    # Attitude limits
    max_roll_pitch: float = 0.5  # rad (~28 degrees)
    max_yaw: float = 0.1

    # Reward scales
    reward_capture: float = -30.0
    reward_escape: float = 20.0
    reward_red_crash: float = 20.0
    reward_blue_crash: float = -20.0
    reward_boundary: float = -5.0
    reward_alive: float = 0.1
    reward_pursuer_proximity: float = 0.5  # Reward evaders when pursuers are close to each other
    reward_pursuer_proximity_decay: float = 2.0  # Exponential decay rate for pursuer proximity reward

    # Target assignment
    random_target_assignment: bool = False  # Randomize red->blue target assignment at reset

    # Device
    device: str = "cpu"

    def __post_init__(self):
        """Compute derived values and load physical parameters."""
        self.n_blue = self.n_pairs
        self.n_red = self.n_pairs
        self.n_drones = self.n_blue + self.n_red

        # Load physical parameters from drone-models
        drone_params = load_params("so_rpy", self.drone_model)
        self.mass = float(drone_params["mass"])
        self.gravity = float(np.abs(drone_params["gravity_vec"][2]))
        self.min_thrust = float(drone_params["thrust_min"]) * 4  # Per motor -> collective
        self.max_thrust = float(drone_params["thrust_max"]) * 4

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
