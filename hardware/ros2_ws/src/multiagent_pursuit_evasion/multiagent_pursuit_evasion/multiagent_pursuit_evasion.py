"""Main entry point for multi-agent pursuit-evasion hardware experiments.

Launches the server, pursuer team, and evader team nodes with configurable
policy type (FFN or ACMPC). Checkpoints and configs are loaded from the
models/{policy_type}/ directory.
"""

import argparse
import multiprocessing
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from multiagent_pursuit_evasion.policy_loader import load_config


def run_server_process(n_blue: int, n_red: int, config: dict):
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
        policy = load_policy(
            policy_type=policy_type,
            checkpoint_path=checkpoint_path,
            config=config,
            device='cpu',
        )
        logger.info(f"Evader policy loaded: {policy_type}")
    except Exception as e:
        logger.error(f"Failed to load evader policy: {e}")
        rclpy.shutdown()
        return

    evader_node = EvaderTeam(
        config=config,
        policy=policy,
        policy_type=policy_type,
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

    Returns:
        Path to the models directory.
    """
    # First try ROS2 share directory
    try:
        share_dir = get_package_share_directory('multiagent_pursuit_evasion')
        models_dir = Path(share_dir) / 'models'
        if models_dir.exists():
            return models_dir
    except Exception:
        pass

    # Fallback to relative to this file's location (for development)
    package_dir = Path(__file__).parent.parent
    models_dir = package_dir / 'models'

    if models_dir.exists():
        return models_dir

    raise FileNotFoundError(
        f"Models directory not found. "
        "Please ensure the models/ directory exists in the package."
    )


def find_checkpoint(policy_type: str) -> Path:
    """Find the checkpoint file for a policy type.

    Looks for best_agent.pt or final_checkpoint.pt in models/{policy_type}/.

    Args:
        policy_type: "ffn" or "acmpc".

    Returns:
        Path to checkpoint file.

    Raises:
        FileNotFoundError: If checkpoint not found.
    """
    models_dir = get_models_dir()
    policy_dir = models_dir / policy_type.lower()

    # Try best_agent.pt first, then final_checkpoint.pt
    for checkpoint_name in ['best_agent.pt', 'final_checkpoint.pt']:
        checkpoint_path = policy_dir / checkpoint_name
        if checkpoint_path.exists():
            return checkpoint_path

    raise FileNotFoundError(
        f"Checkpoint not found in {policy_dir}. "
        f"Please copy best_agent.pt or final_checkpoint.pt to models/{policy_type.lower()}/"
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

    args, _ = parser.parse_known_args()

    # Find config and checkpoint BEFORE rclpy.init() (use print for early logging)
    try:
        config_path = find_config(args.policy_type)
        checkpoint_path = find_checkpoint(args.policy_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Loading config from: {config_path}")
    config = load_config(str(config_path))

    n_blue = config['n_pairs']
    n_red = config['n_pairs']

    # Start all nodes in separate processes to avoid GIL contention
    # Each process has its own Python interpreter and GIL

    # Server process
    server_process = multiprocessing.Process(
        target=run_server_process,
        args=(n_blue, n_red, config),
    )
    server_process.start()
    print("Server process started")

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
