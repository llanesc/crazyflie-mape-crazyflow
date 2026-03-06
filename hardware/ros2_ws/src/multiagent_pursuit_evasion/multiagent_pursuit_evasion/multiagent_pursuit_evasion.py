"""Main entry point for multi-agent pursuit-evasion hardware experiments.

Launches the server, pursuer team, and evader team nodes with configurable
policy type (FFN or ACMPC). Checkpoints and configs are loaded from the
models/{policy_type}/ directory.
"""

import argparse
import multiprocessing
import os
from pathlib import Path

# =============================================================================
# CURRICULUM LEVEL - Set this to use level-specific collision tolerances
# =============================================================================
# Set to None to use base config values, or an integer (0, 1, 2, ...) to use
# parameters from that curriculum level (e.g., collision tolerances)
CURRICULUM_LEVEL = 9  # e.g., 0, 1, 2, or None for base config
# =============================================================================

# =============================================================================
# COLLISION TOLERANCES - Manual overrides (set to None to use config/curriculum)
# =============================================================================
BB_COLLISION_TOLERANCE = 0.2   # Blue-Blue collision distance (meters), or None
RR_COLLISION_TOLERANCE = 0.2   # Red-Red collision distance (meters), or None
RB_COLLISION_TOLERANCE = 0.2   # Red-Blue (capture) distance (meters), or None
# =============================================================================

from ament_index_python.packages import get_package_share_directory

# Package name for looking up source directory
_PACKAGE_NAME = 'multiagent_pursuit_evasion'

from multiagent_pursuit_evasion.policy_loader import load_config


def run_server_process(n_blue: int, n_red: int, config: dict, require_accel: bool = True):
    """Run the server node in a separate process with its own GIL."""
    # Import here to avoid issues with multiprocessing
    import rclpy
    from rclpy import executors
    from multiagent_pursuit_evasion.server import MultiAgentPursuitEvasionServer

    rclpy.init()

    server_node = MultiAgentPursuitEvasionServer(
        n_blue=n_blue,
        n_red=n_red,
        config=config,
        require_accel=require_accel,
    )

    executor = executors.SingleThreadedExecutor()
    executor.add_node(server_node)

    try:
        server_node.get_logger().info(
            f'Server process started: {n_blue} blue vs {n_red} red'
        )
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        server_node.running = False
        executor.shutdown()
        server_node.destroy_node()
        rclpy.shutdown()


def run_evader_process(config: dict, checkpoint_path: str, policy_type: str):
    """Run the evader team in a separate process with its own GIL."""
    import rclpy
    import rclpy.logging
    from rclpy import executors
    from multiagent_pursuit_evasion.pursuer_evader import EvaderTeam
    from multiagent_pursuit_evasion.policy_loader import load_policy

    rclpy.init()
    logger = rclpy.logging.get_logger('evader_process')

    # Load policy in this process
    try:
        policy, obs_preprocessor = load_policy(
            policy_type=policy_type,
            checkpoint_path=checkpoint_path,
            config=config,
            device='cpu',
        )
        logger.info(f"Evader policy loaded: {policy_type}")
        if obs_preprocessor is not None:
            logger.info("Observation preprocessor loaded from checkpoint")
        else:
            logger.warn("No observation preprocessor found in checkpoint")
    except Exception as e:
        logger.error(f"Failed to load evader policy: {e}")
        rclpy.shutdown()
        return

    evader_node = EvaderTeam(
        config=config,
        policy=policy,
        policy_type=policy_type,
        obs_preprocessor=obs_preprocessor,
    )

    executor = executors.SingleThreadedExecutor()
    executor.add_node(evader_node)

    try:
        logger.info('Evader process started')
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        evader_node.destroy_node()
        rclpy.shutdown()


def run_pursuer_process(config: dict):
    """Run the pursuer team in a separate process with its own GIL."""
    import rclpy
    import rclpy.logging
    from rclpy import executors
    from multiagent_pursuit_evasion.pursuer_evader import PursuerTeam

    rclpy.init()
    logger = rclpy.logging.get_logger('pursuer_process')

    pursuer_node = PursuerTeam(config=config)

    executor = executors.SingleThreadedExecutor()
    executor.add_node(pursuer_node)

    try:
        logger.info('Pursuer process started')
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        pursuer_node.destroy_node()
        rclpy.shutdown()


def get_models_dir() -> Path:
    """Get the models directory path.

    Prefers source directory over install to avoid rebuilds during development.

    Order of preference:
    1. MAPE_MODELS_DIR environment variable (if set)
    2. Source directory (auto-detected from install path)
    3. Relative to __file__ (for direct execution)
    4. Installed share directory (fallback)

    Returns:
        Path to the models directory.
    """
    # 1. Environment variable override (for maximum flexibility)
    if 'MAPE_MODELS_DIR' in os.environ:
        models_dir = Path(os.environ['MAPE_MODELS_DIR'])
        if models_dir.exists():
            return models_dir
        print(f"Warning: MAPE_MODELS_DIR={models_dir} does not exist, trying auto-detection")

    # 2. Auto-detect source directory from install path
    # When running via ROS2, __file__ is in:
    #   .../ros2_ws/install/{package}/lib/pythonX.X/site-packages/{package}/...
    # We want to find:
    #   .../ros2_ws/src/{package}/models/
    file_path = Path(__file__).resolve()
    path_str = str(file_path)

    if '/install/' in path_str:
        # Extract workspace root (everything before /install/)
        ws_root = Path(path_str.split('/install/')[0])
        src_models_dir = ws_root / 'src' / _PACKAGE_NAME / 'models'
        if src_models_dir.exists():
            return src_models_dir

    # 3. Try relative to __file__ (for direct python execution or symlink install)
    package_dir = file_path.parent.parent
    src_models_dir = package_dir / 'models'
    if src_models_dir.exists():
        return src_models_dir

    # 4. Fall back to installed share directory
    try:
        share_dir = get_package_share_directory(_PACKAGE_NAME)
        models_dir = Path(share_dir) / 'models'
        if models_dir.exists():
            return models_dir
    except Exception:
        pass

    raise FileNotFoundError(
        "Models directory not found. Set MAPE_MODELS_DIR environment variable or "
        f"ensure models/ exists in src/{_PACKAGE_NAME}/"
    )


def find_checkpoint(policy_type: str) -> Path:
    """Find the checkpoint file for a policy type.

    Looks for best_agent.pt, best_agent_*.pt, or any .pt file in models/{policy_type}/.

    Args:
        policy_type: "ffn" or "acmpc".

    Returns:
        Path to checkpoint file.

    Raises:
        FileNotFoundError: If checkpoint not found.
    """
    models_dir = get_models_dir()
    policy_dir = models_dir / policy_type.lower()

    # Try best_agent.pt first (exact match)
    best_agent = policy_dir / 'best_agent.pt'
    if best_agent.exists():
        return best_agent

    # Try best_agent_*.pt (sorted by step number, highest first)
    best_agents = list(policy_dir.glob("best_agent_*.pt"))
    if best_agents:
        def get_step(p: Path) -> int:
            try:
                return int(p.stem.split("_")[-1])
            except (IndexError, ValueError):
                return 0
        best_agents.sort(key=get_step, reverse=True)
        return best_agents[0]

    # Try final_checkpoint.pt
    final_checkpoint = policy_dir / 'final_checkpoint.pt'
    if final_checkpoint.exists():
        return final_checkpoint

    # Fall back to any .pt file
    pt_files = list(policy_dir.glob("*.pt"))
    if pt_files:
        return pt_files[0]

    raise FileNotFoundError(
        f"Checkpoint not found in {policy_dir}. "
        f"Please copy a .pt checkpoint file to models/{policy_type.lower()}/"
    )


def find_config(policy_type: str) -> Path:
    """Find the config file for a policy type.

    Looks for environment_config.json in models/{policy_type}/.

    Args:
        policy_type: "ffn" or "acmpc".

    Returns:
        Path to config file.

    Raises:
        FileNotFoundError: If config not found.
    """
    models_dir = get_models_dir()
    config_path = models_dir / policy_type.lower() / 'environment_config.json'

    if config_path.exists():
        return config_path

    raise FileNotFoundError(
        f"Config not found at {config_path}. "
        f"Please copy your environment_config.json to models/{policy_type.lower()}/"
    )


# Backwards compatibility: old param names -> new param names
PARAM_NAME_ALIASES = {
    "bb_crash_tolerance": "bb_collision_tolerance",
    "rr_crash_tolerance": "rr_collision_tolerance",
    "br_crash_tolerance": "rb_collision_tolerance",
}


def apply_curriculum_level(config: dict, level: int) -> dict:
    """Apply curriculum level parameters to config.

    Args:
        config: Base configuration dictionary.
        level: Curriculum level index (0, 1, 2, ...).

    Returns:
        Updated config with level parameters applied.
    """
    curriculum_levels = config.get('curriculum_levels')
    if curriculum_levels is None:
        print(f"Warning: CURRICULUM_LEVEL={level} but no curriculum_levels in config")
        return config

    if level < 0 or level >= len(curriculum_levels):
        print(f"Warning: CURRICULUM_LEVEL={level} out of range (0-{len(curriculum_levels)-1})")
        return config

    level_config = curriculum_levels[level]
    level_name = level_config.get("name", f"Level {level}")
    print(f"Applying curriculum level {level}: {level_name}")

    # Get level params (handle both JSON and YAML formats)
    if "params" in level_config and isinstance(level_config["params"], dict):
        level_params = level_config["params"]
    else:
        level_params = {k: v for k, v in level_config.items() if k not in ("name", "level", "spawn")}

    # Apply level params to config
    for param_name, param_value in level_params.items():
        # Translate old param names to new ones
        translated_name = PARAM_NAME_ALIASES.get(param_name, param_name)
        config[translated_name] = param_value
        print(f"  {translated_name}: {param_value}")

    return config


def main():
    """Main entry point."""
    # Use spawn to get fresh Python interpreter in child process
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description='Multi-agent pursuit-evasion hardware experiment'
    )
    parser.add_argument(
        '-p', '--policy_type',
        type=str,
        choices=['ffn', 'acmpc'],
        default='ffn',
        help='Policy type: ffn or acmpc (default: ffn)'
    )
    parser.add_argument(
        '--require-accel',
        action='store_true',
        default=False,
        help='Require acceleration data from blue agents (default: False)'
    )

    args, _ = parser.parse_known_args()

    # Find config and checkpoint BEFORE rclpy.init() (use print for early logging)
    try:
        config_path = find_config(args.policy_type)
        checkpoint_path = find_checkpoint(args.policy_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Loading config from: {config_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    config = load_config(str(config_path))

    # Apply curriculum level if specified
    if CURRICULUM_LEVEL is not None:
        config = apply_curriculum_level(config, CURRICULUM_LEVEL)

    # Apply manual collision tolerance overrides (take priority over curriculum)
    for key, value in [
        ('bb_collision_tolerance', BB_COLLISION_TOLERANCE),
        ('rr_collision_tolerance', RR_COLLISION_TOLERANCE),
        ('rb_collision_tolerance', RB_COLLISION_TOLERANCE),
    ]:
        if value is not None:
            config[key] = value
            print(f"  Override {key}: {value}")

    n_blue = config['n_pairs']
    n_red = config['n_pairs']

    # Start all nodes in separate processes to avoid GIL contention
    # Each process has its own Python interpreter and GIL

    # Server process
    server_process = multiprocessing.Process(
        target=run_server_process,
        args=(n_blue, n_red, config, args.require_accel),
    )
    server_process.start()
    print(f"Server process started (require_accel={args.require_accel})")

    # Evader process (with policy)
    evader_process = multiprocessing.Process(
        target=run_evader_process,
        args=(config, str(checkpoint_path), args.policy_type),
    )
    evader_process.start()
    print(f"Evader process started with {args.policy_type.upper()} policy")

    # Pursuer process
    pursuer_process = multiprocessing.Process(
        target=run_pursuer_process,
        args=(config,),
    )
    pursuer_process.start()
    print("Pursuer process started")

    print(
        f'\nMAPE experiment running: {n_blue} blue vs {n_red} red, '
        f'policy={args.policy_type.upper()}'
    )
    print('Press CTRL-C to shutdown\n')

    # Wait for processes
    try:
        server_process.join()
    except KeyboardInterrupt:
        print('\nShutting down...')

    # Cleanup all processes
    for proc, name in [
        (server_process, 'server'),
        (evader_process, 'evader'),
        (pursuer_process, 'pursuer'),
    ]:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                print(f"Force killed {name} process")


if __name__ == '__main__':
    main()
