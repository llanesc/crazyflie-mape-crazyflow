# Multi-Agent Pursuit-Evasion ROS2 Package

ROS2 package for running multi-agent pursuit-evasion experiments on Crazyflie quadrotors with learned evader policies (FFN or ACMPC).

## Overview

This package implements a hardware testbed for multi-agent pursuit-evasion games where:
- **Blue team (evaders)**: Controlled by learned neural network policies trained in simulation
- **Red team (pursuers)**: Controlled by scripted guidance laws (Pure Pursuit, ProNav, or Augmented ProNav)

The implementation matches the simulation environment (`red_vs_blue_env.py`) to ensure sim-to-real transfer.

## Package Structure

```
multiagent_pursuit_evasion/
├── multiagent_pursuit_evasion/
│   ├── __init__.py
│   ├── multiagent_pursuit_evasion.py  # Main entry point
│   ├── server.py                       # Game state coordinator
│   ├── pursuer_evader.py               # Team control nodes
│   └── policy_loader.py                # Policy checkpoint loading
├── models/
│   ├── acmpc/
│   │   ├── environment_config.json    # Training config
│   │   ├── learning_config.json       # Hyperparameters
│   │   └── best_agent.pt              # Policy checkpoint (user provides)
│   └── ffn/
│       ├── environment_config.json    # Training config
│       ├── learning_config.json       # Hyperparameters
│       └── best_agent.pt              # Policy checkpoint (user provides)
├── package.xml
├── setup.py
└── README.md
```

## Dependencies

### ROS2 Packages
- `rclpy`
- `crazyflie_py`
- `crazyflie_interfaces`
- `multiagent_pursuit_evasion_interfaces`
- `geometry_msgs`
- `tf_transformations`

### Python Packages
- `numpy`
- `scipy`
- `torch`
- `drone-models` (for physical parameters)
- `skrl` (for policy classes)

For ACMPC policies additionally:
- `leap_c`
- `acados`

## Installation

1. Build the interfaces package first:
```bash
cd ~/ros2_ws
colcon build --packages-select multiagent_pursuit_evasion_interfaces
source install/setup.bash
```

2. Build this package:
```bash
colcon build --packages-select multiagent_pursuit_evasion
source install/setup.bash
```

## Setup

Before running, copy your trained policy files to the appropriate models directory:

### For FFN Policy
```bash
cp /path/to/your/best_agent.pt models/ffn/
cp /path/to/your/environment_config.json models/ffn/
```

### For ACMPC Policy
```bash
cp /path/to/your/best_agent.pt models/acmpc/
cp /path/to/your/environment_config.json models/acmpc/
```

The package expects:
- `models/{policy_type}/best_agent.pt` - Policy checkpoint
- `models/{policy_type}/environment_config.json` - Training configuration

## Usage

### Basic Usage

```bash
# Run with FFN policy (default)
ros2 run multiagent_pursuit_evasion main_executor

# Run with ACMPC policy
ros2 run multiagent_pursuit_evasion main_executor -p acmpc
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--policy_type` | `-p` | ffn | Policy type: `ffn` or `acmpc` |

### Examples

```bash
# Run FFN policy
ros2 run multiagent_pursuit_evasion main_executor -p ffn

# Run ACMPC policy
ros2 run multiagent_pursuit_evasion main_executor -p acmpc
```

### Launch Files

```bash
# Launch Crazyflie sim + FFN policy
ros2 launch multiagent_pursuit_evasion launch_sim_ffn.py

# Launch Crazyflie sim + ACMPC policy
ros2 launch multiagent_pursuit_evasion launch_sim_acmpc.py
```

## Architecture

### Nodes

#### MultiAgentPursuitEvasionServer
Central coordinator that:
- Subscribes to pose, velocity, and angular velocity for all agents
- Publishes `Status` messages at 100Hz with full game state
- Handles collision detection (BB, RR, BR) and boundary violations
- Manages target reassignment when blue agents are captured
- Tracks blue velocity for ProNav feedforward acceleration

#### PursuerTeam
Red team controller implementing:
- **Pure Pursuit (PP)**: Linear position-velocity feedback
- **ProNav**: Proportional navigation with LOS rate feedback
- **AugProNav**: Augmented ProNav with speed floor for low-closure scenarios

Control gains are loaded from `pursuit_gains` in the config.

#### EvaderTeam
Blue team controller that:
- Builds 46D observations matching simulation environment
- Runs policy inference (deterministic, mean action)
- Denormalizes actions to physical units [roll, pitch, yaw, thrust]

### Message Flow

```
Crazyflie → pose, velocity, angular_velocity → Server → Status → Teams → cmd_attitude_setpoint → Crazyflie
```

### Observation Format (46D for n_pairs=2)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Own state | 12 | pos(3) + vel(3) + rpy(3) + rpy_rates(3) |
| Own one-hot | 2 | Agent identity (masked by alive) |
| All blue states | 14 | n_blue × (pos(3) + vel(3) + alive(1)) |
| All red states | 14 | n_red × (pos(3) + vel(3) + alive(1)) |
| Target assignments | 4 | n_red × n_blue one-hot (masked by red alive) |

## Configuration

The `environment_config.json` contains:

```json
{
  "policy_type": "ffn",
  "n_pairs": 2,
  "drone_model": "cf2x_T350",
  "pursuer_strategy": "ProNav",
  "roll_pitch_max": 0.5,
  "yaw_max": 0.1,
  "bb_collision_tolerance": 0.5,
  "rr_collision_tolerance": 0.5,
  "rb_collision_tolerance": 0.5,
  "boundary_size": 3.0,
  "min_altitude": 0.1,
  "max_altitude": 2.0,
  "pursuit_gains": {
    "N_pronav_fb": 5.0,
    "N_pronav_ff": 1.0,
    "velocity_closure_threshold": 0.1,
    "pp_k_pxy": 6.1624,
    "pp_k_vxy": 3.39,
    "pp_k_pz": 20.0,
    "pp_k_vz": 10.0
  },
  "observation_space": {
    "per_agent_obs_dim": 46
  }
}
```

## Hardware Setup

### Crazyflie Naming Convention
- Blue (evaders): `blue_1`, `blue_2`, ...
- Red (pursuers): `red_1`, `red_2`, ...

### Required Logging Variables
Configure the Crazyflies to stream:
- Pose: via motion capture system
- Velocity: `stateEstimate.vx`, `stateEstimate.vy`, `stateEstimate.vz`
- Angular velocity: `stateEstimateZ.rateRoll`, `stateEstimateZ.ratePitch`, `stateEstimateZ.rateYaw` (in millirad/s)

### Command Service
Use the `/command` service to control the experiment:
```bash
# Initialize (hover at starting positions)
ros2 service call /command multiagent_pursuit_evasion_interfaces/srv/Command "{command: 1}"

# Run (start pursuit-evasion)
ros2 service call /command multiagent_pursuit_evasion_interfaces/srv/Command "{command: 2}"

# Stop
ros2 service call /command multiagent_pursuit_evasion_interfaces/srv/Command "{command: 0}"
```

## Collision and Boundary Rules

Matching the simulation environment:
- **BB collision**: Both blue agents die if within `bb_collision_tolerance`
- **RR collision**: Both red agents die if within `rr_collision_tolerance`
- **BR capture**: Blue dies, capturing red also dies if within `rb_collision_tolerance`
- **Boundary**: Blue dies if outside `[-boundary_size, boundary_size]` in x/y or outside `[min_altitude, max_altitude]` in z

When a blue dies, any red pursuing it is reassigned to the closest remaining alive blue.

## Policy Types

### FFN (Feedforward Neural Network)
- Simple MLP policy with tanh output
- Fast inference (~0.1ms per batch)
- Architecture: configurable hidden sizes (default: [256, 256])

### ACMPC (Actor-Critic MPC)
- Neural network outputs MPC cost parameters
- MPC solver computes optimal control
- Requires `leap_c` and `acados` installation
- Slower inference but potentially better tracking

## Troubleshooting

### "Checkpoint not found"
Copy your `best_agent.pt` to `models/{policy_type}/`:
```bash
cp /path/to/best_agent.pt models/ffn/
```

### "Config not found"
Copy your `environment_config.json` to `models/{policy_type}/`:
```bash
cp /path/to/environment_config.json models/ffn/
```

### "Agents not ready"
Ensure all Crazyflies are streaming pose, velocity, and angular_velocity data.

### Policy loading fails
- Verify config matches training configuration
- For ACMPC, ensure leap_c and acados are installed

### Observation dimension mismatch
The observation dimension depends on `n_pairs`. Default is 46D for n_pairs=2:
- 12 (own) + 2 (one-hot) + 14 (blues) + 14 (reds) + 4 (targets) = 46

## License

MIT
