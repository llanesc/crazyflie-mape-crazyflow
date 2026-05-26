# Crazyflie MAPE Crazyflow

Multi-agent pursuit-evasion experiments for Crazyflie quadrotors using Crazyflow simulation, MAPPO training, and two evader policy families:

- **FFN / MLP**: a shared feedforward Gaussian policy that directly outputs normalized attitude/thrust commands.
- **ACMPC**: a shared LEAP-C/acados model predictive control policy whose neural network learns MPC cost weights and references.

Blue agents are evaders controlled by learned policies. Red agents are pursuers controlled by scripted guidance laws such as Pure Pursuit, ProNav, Augmented ProNav, or TPN, depending on the experiment configuration.

## Repository Layout

```text
.
├── crazyflie_mape_crazyflow/        # Main Python package
│   ├── agents/                      # Custom SKRL/MAPPO agent integrations
│   ├── envs/                        # Crazyflow red-vs-blue environment and spawn logic
│   ├── leap_c/                      # Quadrotor MPC dynamics, OCP builders, and planners
│   ├── policies/                    # FFN, ACMPC, and shared critic policy modules
│   ├── pursuit/                     # Scripted pursuer guidance controllers
│   ├── utils/                       # Experiment config, curriculum, and math helpers
│   └── preprocessors.py             # Observation/state preprocessing helpers
├── scripts/                         # Training, evaluation, sweeps, plotting, and debugging
├── results/                         # Experiment configs and run outputs
│   ├── ffn/                         # MLP/FFN experiments
│   └── acmpc/                       # ACMPC experiments
├── tests/                           # Unit, integration, profiling, and debugging tests
├── docs/                            # Notes on MPC and parameter changes
├── external/                        # Vendored/local dependencies
│   ├── crazyflow/                   # JAX/MuJoCo Crazyflie simulator
│   ├── leap-c/                      # Differentiable MPC tooling
│   └── skrl/                        # Local SKRL checkout, if used
├── hardware/                        # ROS2 hardware/simulation stack and Crazyflie support
├── pyproject.toml                   # Package metadata, dependencies, pixi config, lint/test config
└── README.md
```

## Main Package

`crazyflie_mape_crazyflow/envs/red_vs_blue_env.py` implements the vectorized Crazyflow environment. It advances the drone simulation, computes observations/rewards, checks collisions and boundary violations, assigns pursuer targets, and exposes a multi-agent API for SKRL.

`crazyflie_mape_crazyflow/envs/red_vs_blue_config.py` contains the `RedVsBlueEnvConfig` dataclass. It defines agent counts, simulation/control rates, drone model parameters, collision tolerances, arena limits, reward weights, domain randomization, disturbances, and MPC state representation.

`crazyflie_mape_crazyflow/envs/spawn.py` builds deterministic, box-random, and curriculum spawn functions. These are driven by the `environment.spawn` blocks in the YAML experiment files.

`crazyflie_mape_crazyflow/envs/wrappers.py` contains environment wrappers such as action rescaling between normalized policy outputs and physical control commands.

`crazyflie_mape_crazyflow/policies/ffn_shared_policy.py` defines the shared MLP/FFN Gaussian policy. It maps each blue agent observation to `[roll, pitch, yaw, thrust]` actions normalized to `[-1, 1]`.

`crazyflie_mape_crazyflow/policies/leap_c_shared_policy_linear_ls.py` defines the ACMPC policy. It wraps a differentiable LEAP-C MPC planner and uses neural networks to output LINEAR_LS cost weights and references.

`crazyflie_mape_crazyflow/policies/shared_critic.py` defines the shared value function used by MAPPO.

`crazyflie_mape_crazyflow/agents/mappo_mpc.py` contains the MAPPO variant used for ACMPC policies, including handling for MPC state passed through environment info.

`crazyflie_mape_crazyflow/leap_c/` contains quadrotor MPC internals:

- `quadrotor_planner.py`: planner wrapper used by the ACMPC policy.
- `quadrotor_ocp_linear_ls.py`: LINEAR_LS acados OCP formulation.
- `quadrotor_ocp_qp.py`: QP-oriented OCP variant.
- `so_rpy_dynamics_sx.py`: quadrotor dynamics in roll/pitch/yaw state form.

`crazyflie_mape_crazyflow/pursuit/` contains scripted red-team pursuit controllers such as pure pursuit and proportional navigation.

`crazyflie_mape_crazyflow/utils/experiment_config.py` loads YAML experiment files and converts them into environment, policy, and training configuration dictionaries. `utils/curriculum.py` handles curriculum levels and advancement.

## Scripts

Training entry points:

- `scripts/train_mappo_ffn.py`: trains FFN/MLP blue evaders with MAPPO.
- `scripts/train_mappo_acmpc.py`: trains ACMPC blue evaders with MAPPO and LEAP-C MPC.

Evaluation entry points:

- `scripts/eval_mappo_ffn.py`: evaluates trained FFN/MLP checkpoints from `results/ffn`.
- `scripts/eval_mappo_acmpc.py`: evaluates trained ACMPC checkpoints from `results/acmpc`.

Analysis and debugging:

- `scripts/sweep_override_mass*.py`: run mass robustness sweeps.
- `scripts/mass_sweep_plotter.py`: plot mass sweep results.
- `scripts/plot_training_comparison.py`: compare training curves.
- `scripts/compare_crazysim_crazyflow.py`: compare simulator behavior.
- `scripts/compare_obs_trajectories.py` and `scripts/plotting/compare_obs_plots.py`: inspect observation and trajectory differences.
- `scripts/debug_*.py`: targeted debugging utilities for training/evaluation/disturbances.
- `scripts/sysid_from_sim_csv.py`: system identification helper from simulation CSV data.
- `scripts/replay_crazysim_in_crazyflow.py`: replay CrazySim logs in Crazyflow.

## Experiments and Results

Experiment directories live under `results/{policy_type}/{experiment}/`.

```text
results/
├── ffn/
│   ├── action_penalty/config.yaml
│   ├── bounded_thrust/config.yaml
│   └── curriculum_training/config.yaml
└── acmpc/
    ├── action_penalty/config.yaml
    ├── box_random_spawn/config.yaml
    ├── curriculum_training/config.yaml
    └── deterministic_spawn/config.yaml
```

Each `config.yaml` describes:

- `environment`: agent count, pursuit strategy, drone model, spawn distribution, arena, randomization, disturbances, and physical limits.
- `policy`: FFN or ACMPC architecture and action/MPC settings.
- `training`: MAPPO rollout length, learning rate, scheduler, device, number of worlds, and total timesteps.
- `rewards`: capture, crash, boundary, smoothness, energy, angle, velocity, and proximity reward terms.
- `curriculum`: optional staged spawn and parameter schedule.

Training creates run folders under:

```text
results/ffn/<experiment>/results/run_<timestamp>/
results/acmpc/<experiment>/results/run_<timestamp>/
```

Run folders contain checkpoints plus exported config JSON files used by the evaluation scripts. Checkpoints can be selected by run directory name, full path, or `--step`.

## Setup

This project is packaged with `pyproject.toml` and includes a pixi workspace. Python `>=3.11,<3.14` is expected.

With pixi:

```bash
pixi install
pixi shell
```

Or with pip in an existing environment:

```bash
pip install -e ".[learning]"
pip install -e external/crazyflow
pip install -e external/leap-c
```

For GPU JAX support, use the `gpu` extra or pixi GPU environment defined in `pyproject.toml`.

## Train

Train an FFN/MLP policy:

```bash
python scripts/train_mappo_ffn.py --experiment action_penalty
```

Train an ACMPC policy:

```bash
python scripts/train_mappo_acmpc.py --experiment action_penalty
```

Resume a run:

```bash
python scripts/train_mappo_ffn.py --experiment action_penalty --resume-run run_YYYYMMDDHHMMSS
```

Start from a curriculum level:

```bash
python scripts/train_mappo_acmpc.py --experiment action_penalty --curriculum-level 4
```

## Evaluate

Evaluate the MLP/FFN policy on curriculum level 9:

```bash
python scripts/eval_mappo_ffn.py \
  --experiment action_penalty \
  --checkpoint run_20260223130705 \
  --n-episodes 1000 \
  --level 9 \
  --n-worlds 1 \
  --render \
  --no-domain-rand
```

Evaluate the ACMPC policy with deterministic actions, no disturbances, and an overridden blue drone mass:

```bash
python scripts/eval_mappo_acmpc.py \
  --experiment action_penalty \
  --n-episodes 15 \
  --n-worlds 1 \
  --no-domain-rand \
  --no-disturbance \
  --deterministic \
  --checkpoint run_20260219203108 \
  --level 9 \
  --override-mass 0.0406
```

Useful evaluation flags:

- `--checkpoint`: run folder name such as `run_20260223130705`, or a full checkpoint path.
- `--step`: load a specific periodic checkpoint such as `500k`, `1m`, or `2m`.
- `--level`: evaluate with a curriculum level's spawn and parameters.
- `--n-worlds`: number of parallel environments.
- `--n-episodes`: number of completed episodes to evaluate.
- `--deterministic`: use the policy mean action.
- `--render`: show MuJoCo rendering.
- `--record`: save a video, requires `--n-worlds 1`.
- `--no-domain-rand`: disable mass/inertia randomization.
- `--no-disturbance`: disable force/torque disturbances.
- `--override-mass`: override blue/evader drone mass. Pass one value for all blue drones or multiple values for per-drone masses.
- `--save-obs`: save per-episode observation CSV files.
- `--trajectory`: draw drone trajectory traces while rendering.

## Hardware and ROS2

`hardware/ros2_ws/src/multiagent_pursuit_evasion/` is the ROS2 package for running trained policies in Crazyflie simulation or on hardware. It contains:

- `multiagent_pursuit_evasion/server.py`: game-state coordinator, collision logic, and status publisher.
- `multiagent_pursuit_evasion/pursuer_evader.py`: red and blue team control nodes.
- `multiagent_pursuit_evasion/policy_loader.py`: loads FFN or ACMPC checkpoints for deployment.
- `config/`: Crazyflie, motion-capture, and pursuit-evasion configs.
- `launch/`: launch files for sim and hardware FFN/ACMPC runs.
- `models/`: exported model checkpoints and config JSON files for ROS2 deployment.

See `hardware/ros2_ws/src/multiagent_pursuit_evasion/README.md` for ROS2-specific build and launch commands.

## Tests

Run the main test suite:

```bash
pytest -v tests
```

With pixi:

```bash
pixi run -e tests test
```

The `tests/` directory includes environment behavior tests, dynamics and MPC checks, hardware-adjacent mapping tests, and profiling scripts.
