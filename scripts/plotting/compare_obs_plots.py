#!/usr/bin/env python3
"""Compare observations from hardware (rosbag) and simulation (eval) CSVs.

Creates time series plots comparing position, velocity, RPY, and RPY rates
for both blue agents.

Usage:
    python compare_obs_plots.py <rosbag_csv> <eval_csv> [--output <output_dir>]

Example:
    python compare_obs_plots.py logs/rosbag_obs.csv eval_obs.csv --output plots/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Observation indices within each agent's observation vector (46 dims total)
# Own state [12]: pos(3), vel(3), rpy(3), rpy_rates(3)
OBS_INDICES = {
    'pos_x': 0,
    'pos_y': 1,
    'pos_z': 2,
    'vel_x': 3,
    'vel_y': 4,
    'vel_z': 5,
    'roll': 6,
    'pitch': 7,
    'yaw': 8,
    'roll_rate': 9,
    'pitch_rate': 10,
    'yaw_rate': 11,
}


def load_obs_csv(csv_path: Path) -> pd.DataFrame:
    """Load observation CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with time and observation columns.
    """
    df = pd.read_csv(csv_path)
    return df


def get_agent_data(df: pd.DataFrame, agent_idx: int, obs_name: str) -> np.ndarray:
    """Extract specific observation data for an agent.

    Args:
        df: DataFrame with observation data.
        agent_idx: Agent index (0 or 1).
        obs_name: Observation name (e.g., 'pos_x', 'vel_y', 'roll').

    Returns:
        Array of observation values.
    """
    obs_idx = OBS_INDICES[obs_name]
    col_name = f'agent{agent_idx}_obs{obs_idx}'
    return df[col_name].values


def compute_max_deviation_from_start(data_list: list) -> float:
    """Compute the maximum deviation from starting value across all data arrays.

    This ensures that when we center on the starting value, all data fits within
    the y-limits. The range will be 2 * max_deviation.

    Args:
        data_list: List of numpy arrays (each array's first element is its start value).

    Returns:
        Maximum absolute deviation from start value, with 10% margin.
    """
    max_deviation = 0.0
    for data in data_list:
        start_val = data[0]
        # Find max deviation above and below start
        max_above = np.max(data) - start_val
        max_below = start_val - np.min(data)
        max_deviation = max(max_deviation, max_above, max_below)
    # Add 10% margin
    return max_deviation * 1.1 if max_deviation > 1e-6 else 0.5


def plot_comparison(
    time_hw: np.ndarray,
    data_hw: np.ndarray,
    time_sim: np.ndarray,
    data_sim: np.ndarray,
    ax: plt.Axes,
    ylabel: str,
    title: str = None,
    ylim: tuple = None,
):
    """Plot hardware vs simulation comparison on a single axis.

    Args:
        time_hw: Hardware time array.
        data_hw: Hardware data array.
        time_sim: Simulation time array.
        data_sim: Simulation data array.
        ax: Matplotlib axis to plot on.
        ylabel: Y-axis label.
        title: Optional title for the subplot.
        ylim: Optional (ymin, ymax) tuple for y-axis limits.
    """
    ax.plot(time_hw, data_hw, 'b-', label='Hardware', linewidth=1.5, alpha=0.8)
    ax.plot(time_sim, data_sim, 'r--', label='Simulation', linewidth=1.5, alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)


def create_agent_plots(
    df_hw: pd.DataFrame,
    df_sim: pd.DataFrame,
    agent_idx: int,
    output_dir: Path = None,
    show: bool = True,
):
    """Create all comparison plots for a single agent.

    Args:
        df_hw: Hardware observation DataFrame.
        df_sim: Simulation observation DataFrame.
        agent_idx: Agent index (0 or 1).
        output_dir: Directory to save plots (None to skip saving).
        show: Whether to display plots interactively.
    """
    time_hw = df_hw['time'].values
    time_sim = df_sim['time'].values

    agent_name = f"Blue {agent_idx}"

    # Define plot groups: (title, dimensions with labels, unit, filename)
    plot_groups = [
        ('Position', [('pos_x', 'X [m]'), ('pos_y', 'Y [m]'), ('pos_z', 'Z [m]')], 'position'),
        ('Velocity', [('vel_x', 'Vx [m/s]'), ('vel_y', 'Vy [m/s]'), ('vel_z', 'Vz [m/s]')], 'velocity'),
        ('RPY Angles', [('roll', 'Roll [rad]'), ('pitch', 'Pitch [rad]'), ('yaw', 'Yaw [rad]')], 'rpy'),
        ('RPY Rates', [('roll_rate', 'Roll Rate [rad/s]'), ('pitch_rate', 'Pitch Rate [rad/s]'), ('yaw_rate', 'Yaw Rate [rad/s]')], 'rpy_rates'),
    ]

    for group_title, dims, filename in plot_groups:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f'{agent_name} - {group_title} Comparison', fontsize=12, fontweight='bold')

        # Collect all data for this group to compute shared range
        all_data = []
        dim_data = []
        for dim, label in dims:
            data_hw = get_agent_data(df_hw, agent_idx, dim)
            data_sim = get_agent_data(df_sim, agent_idx, dim)
            all_data.extend([data_hw, data_sim])
            dim_data.append((data_hw, data_sim, label))

        # Compute max deviation from start across all dimensions for consistent plot sizes
        max_dev = compute_max_deviation_from_start(all_data)

        # Plot each dimension centered on its starting value
        for i, (data_hw, data_sim, label) in enumerate(dim_data):
            # Center on hardware starting value, use shared deviation for consistent size
            center = data_hw[0]
            ylim = (center - max_dev, center + max_dev)
            plot_comparison(time_hw, data_hw, time_sim, data_sim, axes[i], label, ylim=ylim)

        axes[-1].set_xlabel('Time [s]')
        fig.tight_layout()

        if output_dir:
            fig.savefig(output_dir / f'agent{agent_idx}_{filename}.png', dpi=150)

        if not show:
            plt.close(fig)

    if show:
        plt.show()


def create_combined_plot(
    df_hw: pd.DataFrame,
    df_sim: pd.DataFrame,
    output_dir: Path = None,
    show: bool = True,
):
    """Create a single combined figure with all agents and state variables.

    Args:
        df_hw: Hardware observation DataFrame.
        df_sim: Simulation observation DataFrame.
        output_dir: Directory to save plot (None to skip saving).
        show: Whether to display plot interactively.
    """
    time_hw = df_hw['time'].values
    time_sim = df_sim['time'].values

    # Create figure with 4 rows (pos, vel, rpy, rpy_rates) x 2 cols (agent0, agent1)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    fig.suptitle('Hardware vs Simulation Observation Comparison', fontsize=14, fontweight='bold')

    state_groups = [
        ('Position', [('pos_x', 'X'), ('pos_y', 'Y'), ('pos_z', 'Z')], '[m]'),
        ('Velocity', [('vel_x', 'Vx'), ('vel_y', 'Vy'), ('vel_z', 'Vz')], '[m/s]'),
        ('RPY', [('roll', 'R'), ('pitch', 'P'), ('yaw', 'Y')], '[rad]'),
        ('RPY Rates', [('roll_rate', 'dR'), ('pitch_rate', 'dP'), ('yaw_rate', 'dY')], '[rad/s]'),
    ]

    for row, (group_name, dims, unit) in enumerate(state_groups):
        # Compute max range across all dimensions AND both agents for this state group
        all_data = []
        for agent_idx in [0, 1]:
            for dim_name, _ in dims:
                all_data.append(get_agent_data(df_hw, agent_idx, dim_name))
                all_data.append(get_agent_data(df_sim, agent_idx, dim_name))
        max_dev = compute_max_deviation_from_start(all_data)

        for col, agent_idx in enumerate([0, 1]):
            ax = axes[row, col]

            # Collect data for this agent to find center point
            agent_data_hw = []
            agent_data_sim = []
            for dim_name, _ in dims:
                agent_data_hw.append(get_agent_data(df_hw, agent_idx, dim_name))
                agent_data_sim.append(get_agent_data(df_sim, agent_idx, dim_name))

            # Center on the mean of starting values across all dimensions
            center = np.mean([d[0] for d in agent_data_hw])
            ylim = (center - max_dev, center + max_dev)

            # Plot all 3 dimensions with different colors
            colors = ['tab:blue', 'tab:orange', 'tab:green']
            for dim_idx, (dim_name, dim_label) in enumerate(dims):
                data_hw = agent_data_hw[dim_idx]
                data_sim = agent_data_sim[dim_idx]

                ax.plot(time_hw, data_hw, '-', color=colors[dim_idx],
                       label=f'{dim_label} HW', linewidth=1.2, alpha=0.8)
                ax.plot(time_sim, data_sim, '--', color=colors[dim_idx],
                       label=f'{dim_label} Sim', linewidth=1.2, alpha=0.8)

            ax.set_ylabel(f'{group_name} {unit}', fontsize=9)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=7, ncol=2)

            if row == 0:
                ax.set_title(f'Blue {agent_idx}', fontsize=11, fontweight='bold')

    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')

    fig.tight_layout()

    if output_dir:
        fig.savefig(output_dir / 'combined_comparison.png', dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Compare hardware and simulation observation CSVs'
    )
    parser.add_argument(
        'rosbag_csv',
        type=str,
        help='Path to rosbag observation CSV (hardware)'
    )
    parser.add_argument(
        'eval_csv',
        type=str,
        help='Path to eval observation CSV (simulation)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for saved plots (default: display only)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots interactively (useful for batch processing)'
    )
    parser.add_argument(
        '--combined-only',
        action='store_true',
        help='Only create the combined comparison plot'
    )

    args = parser.parse_args()

    # Load CSVs
    rosbag_csv = Path(args.rosbag_csv)
    eval_csv = Path(args.eval_csv)

    if not rosbag_csv.exists():
        print(f"Error: Rosbag CSV not found: {rosbag_csv}")
        return 1

    if not eval_csv.exists():
        print(f"Error: Eval CSV not found: {eval_csv}")
        return 1

    print(f"Loading hardware data: {rosbag_csv}")
    df_hw = load_obs_csv(rosbag_csv)
    print(f"  Samples: {len(df_hw)}, Duration: {df_hw['time'].iloc[-1]:.2f}s")

    print(f"Loading simulation data: {eval_csv}")
    df_sim = load_obs_csv(eval_csv)
    print(f"  Samples: {len(df_sim)}, Duration: {df_sim['time'].iloc[-1]:.2f}s")

    # Setup output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving plots to: {output_dir}")

    show = not args.no_show

    # Create plots
    if args.combined_only:
        print("Creating combined comparison plot...")
        create_combined_plot(df_hw, df_sim, output_dir, show)
    else:
        print("Creating individual agent plots...")
        for agent_idx in [0, 1]:
            print(f"  Agent {agent_idx}...")
            create_agent_plots(df_hw, df_sim, agent_idx, output_dir, show=False)

        print("Creating combined comparison plot...")
        create_combined_plot(df_hw, df_sim, output_dir, show)

    if output_dir:
        print(f"Plots saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
