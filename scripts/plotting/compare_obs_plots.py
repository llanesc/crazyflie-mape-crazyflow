#!/usr/bin/env python3
"""Plot or compare observations from hardware (rosbag) and/or simulation (eval) CSVs.

Supports two modes:
  - Single CSV:  plot position, velocity, RPY, and body rates for both blue agents.
  - Comparison:  overlay hardware vs simulation traces on each subplot.

Auto-detects the observation format:
  - 46-dim (legacy): own state uses RPY (3) directly.
  - 52-dim (current): own state uses a flattened rotation matrix (9);
    RPY is derived via ZYX Euler extraction.

Usage:
    # Single CSV (simulation only)
    python compare_obs_plots.py ep003_sim.csv --output plots/

    # Compare hardware vs simulation
    python compare_obs_plots.py rosbag_obs.csv --eval eval_sim.csv --output plots/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Observation format definitions ──────────────────────────────────────────

# Legacy 46-dim format: own state = pos(3)+vel(3)+rpy(3)+rpy_rates(3) = 12
OBS_INDICES_46 = {
    'pos_x': 0, 'pos_y': 1, 'pos_z': 2,
    'vel_x': 3, 'vel_y': 4, 'vel_z': 5,
    'roll': 6, 'pitch': 7, 'yaw': 8,
    'roll_rate': 9, 'pitch_rate': 10, 'yaw_rate': 11,
}

# Current 52-dim format: own state = pos(3)+vel(3)+rotmat(9)+body_rates(3) = 18
OBS_INDICES_52 = {
    'pos_x': 0, 'pos_y': 1, 'pos_z': 2,
    'vel_x': 3, 'vel_y': 4, 'vel_z': 5,
    # Rotation matrix (flattened row-major 3x3) at indices 6-14
    'rotmat_00': 6, 'rotmat_01': 7, 'rotmat_02': 8,
    'rotmat_10': 9, 'rotmat_11': 10, 'rotmat_12': 11,
    'rotmat_20': 12, 'rotmat_21': 13, 'rotmat_22': 14,
    'roll_rate': 15, 'pitch_rate': 16, 'yaw_rate': 17,
}


def detect_red_agents(df: pd.DataFrame) -> int:
    """Detect the number of red (pursuer) agents from column names.

    Returns:
        Number of red agents found (0 if none).
    """
    n_red = 0
    while f'red{n_red}_pos_x' in df.columns:
        n_red += 1
    return n_red


def get_red_agent_data(df: pd.DataFrame, red_idx: int, obs_name: str) -> np.ndarray:
    """Extract state data for a red (pursuer) agent.

    Args:
        df: DataFrame with red agent columns.
        red_idx: Red agent index (0, 1, ...).
        obs_name: State name (e.g., 'pos_x', 'vel_y', 'roll').

    Returns:
        Array of state values.
    """
    col_name = f'red{red_idx}_{obs_name}'
    return df[col_name].values


def detect_obs_format(df: pd.DataFrame) -> int:
    """Detect observation format from the number of agent0 columns.

    Returns:
        46 or 52 indicating the observation dimensionality.
    """
    agent0_cols = [c for c in df.columns if c.startswith('agent0_obs')]
    n_dims = len(agent0_cols)
    if n_dims <= 46:
        return 46
    return 52


def load_obs_csv(csv_path: Path) -> pd.DataFrame:
    """Load observation CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with time and observation columns.
    """
    df = pd.read_csv(csv_path)
    return df


def _rotmat_to_rpy(df: pd.DataFrame, agent_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert flattened rotation matrix columns to roll, pitch, yaw arrays.

    Uses ZYX Euler angle extraction:
        roll  = atan2(R[2,1], R[2,2])
        pitch = -asin(clamp(R[2,0]))
        yaw   = atan2(R[1,0], R[0,0])

    Args:
        df: DataFrame with observation data (52-dim format).
        agent_idx: Agent index.

    Returns:
        Tuple of (roll, pitch, yaw) arrays in radians.
    """
    idx = OBS_INDICES_52
    prefix = f'agent{agent_idx}_obs'
    r00 = df[f'{prefix}{idx["rotmat_00"]}'].values
    r10 = df[f'{prefix}{idx["rotmat_10"]}'].values
    r20 = df[f'{prefix}{idx["rotmat_20"]}'].values
    r21 = df[f'{prefix}{idx["rotmat_21"]}'].values
    r22 = df[f'{prefix}{idx["rotmat_22"]}'].values

    roll = np.arctan2(r21, r22)
    pitch = -np.arcsin(np.clip(r20, -1.0, 1.0))
    yaw = np.arctan2(r10, r00)
    return roll, pitch, yaw


def get_agent_data(df: pd.DataFrame, agent_idx: int, obs_name: str,
                   obs_format: int = 46) -> np.ndarray:
    """Extract specific observation data for an agent.

    Args:
        df: DataFrame with observation data.
        agent_idx: Agent index (0 or 1).
        obs_name: Observation name (e.g., 'pos_x', 'vel_y', 'roll').
        obs_format: 46 (legacy RPY) or 52 (rotation matrix).

    Returns:
        Array of observation values.
    """
    if obs_format == 52:
        # RPY must be derived from rotation matrix columns
        if obs_name in ('roll', 'pitch', 'yaw'):
            roll, pitch, yaw = _rotmat_to_rpy(df, agent_idx)
            return {'roll': roll, 'pitch': pitch, 'yaw': yaw}[obs_name]
        idx = OBS_INDICES_52
    else:
        idx = OBS_INDICES_46

    obs_idx = idx[obs_name]
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


def plot_traces(
    ax: plt.Axes,
    ylabel: str,
    time_a: np.ndarray,
    data_a: np.ndarray,
    label_a: str = 'Hardware',
    time_b: np.ndarray = None,
    data_b: np.ndarray = None,
    label_b: str = 'Simulation',
    title: str = None,
    ylim: tuple = None,
):
    """Plot one or two traces on a single axis.

    When *time_b* / *data_b* are ``None`` only the first trace is drawn.
    """
    ax.plot(time_a, data_a, 'b-', label=label_a, linewidth=1.5, alpha=0.8)
    if time_b is not None and data_b is not None:
        ax.plot(time_b, data_b, 'r--', label=label_b, linewidth=1.5, alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)


def create_agent_plots(
    df_a: pd.DataFrame,
    agent_idx: int,
    df_b: pd.DataFrame = None,
    label_a: str = 'Hardware',
    label_b: str = 'Simulation',
    output_dir: Path = None,
    show: bool = True,
):
    """Create all observation plots for a single agent.

    When *df_b* is ``None`` only *df_a* is plotted (single-CSV mode).

    Args:
        df_a: First (or only) observation DataFrame.
        agent_idx: Agent index (0 or 1).
        df_b: Optional second DataFrame for comparison.
        label_a: Legend label for df_a traces.
        label_b: Legend label for df_b traces.
        output_dir: Directory to save plots (None to skip saving).
        show: Whether to display plots interactively.
    """
    fmt_a = detect_obs_format(df_a)
    fmt_b = detect_obs_format(df_b) if df_b is not None else None

    time_a = df_a['time'].values
    time_b = df_b['time'].values if df_b is not None else None

    agent_name = f"Blue {agent_idx}"
    mode_str = 'Comparison' if df_b is not None else ''

    # Define plot groups: (title, dimensions with labels, filename)
    plot_groups = [
        ('Position', [('pos_x', 'X [m]'), ('pos_y', 'Y [m]'), ('pos_z', 'Z [m]')], 'position'),
        ('Velocity', [('vel_x', 'Vx [m/s]'), ('vel_y', 'Vy [m/s]'), ('vel_z', 'Vz [m/s]')], 'velocity'),
        ('RPY Angles', [('roll', 'Roll [rad]'), ('pitch', 'Pitch [rad]'), ('yaw', 'Yaw [rad]')], 'rpy'),
        ('Body Rates', [('roll_rate', 'Roll Rate [rad/s]'), ('pitch_rate', 'Pitch Rate [rad/s]'), ('yaw_rate', 'Yaw Rate [rad/s]')], 'body_rates'),
    ]

    for group_title, dims, filename in plot_groups:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        title = f'{agent_name} - {group_title} {mode_str}'.rstrip()
        fig.suptitle(title, fontsize=12, fontweight='bold')

        all_data = []
        dim_data = []
        for dim, label in dims:
            da = get_agent_data(df_a, agent_idx, dim, fmt_a)
            db = get_agent_data(df_b, agent_idx, dim, fmt_b) if df_b is not None else None
            all_data.append(da)
            if db is not None:
                all_data.append(db)
            dim_data.append((da, db, label))

        max_dev = compute_max_deviation_from_start(all_data)

        for i, (da, db, label) in enumerate(dim_data):
            center = da[0]
            ylim = (center - max_dev, center + max_dev)
            plot_traces(axes[i], label,
                        time_a, da, label_a,
                        time_b, db, label_b,
                        ylim=ylim)

        axes[-1].set_xlabel('Time [s]')
        fig.tight_layout()

        if output_dir:
            fig.savefig(output_dir / f'agent{agent_idx}_{filename}.png', dpi=150)

        if not show:
            plt.close(fig)

    if show:
        plt.show()


def create_red_agent_plots(
    df_a: pd.DataFrame,
    red_idx: int,
    df_b: pd.DataFrame = None,
    label_a: str = 'Hardware',
    label_b: str = 'Simulation',
    output_dir: Path = None,
    show: bool = True,
):
    """Create observation plots for a single red (pursuer) agent.

    Red agents have raw state columns: pos(3), vel(3), rpy(3), active(1).
    When df_b is provided and also has red columns, comparison traces are drawn.

    Args:
        df_a: First (or only) DataFrame with red agent columns.
        red_idx: Red agent index (0 or 1).
        df_b: Optional second DataFrame for comparison.
        label_a: Legend label for df_a traces.
        label_b: Legend label for df_b traces.
        output_dir: Directory to save plots (None to skip saving).
        show: Whether to display plots interactively.
    """
    time_a = df_a['time'].values
    comparing = df_b is not None and detect_red_agents(df_b) > red_idx
    time_b = df_b['time'].values if comparing else None
    agent_name = f"Red {red_idx}"

    plot_groups = [
        ('Position', [('pos_x', 'X [m]'), ('pos_y', 'Y [m]'), ('pos_z', 'Z [m]')], 'position'),
        ('Velocity', [('vel_x', 'Vx [m/s]'), ('vel_y', 'Vy [m/s]'), ('vel_z', 'Vz [m/s]')], 'velocity'),
        ('RPY Angles', [('roll', 'Roll [rad]'), ('pitch', 'Pitch [rad]'), ('yaw', 'Yaw [rad]')], 'rpy'),
    ]

    for group_title, dims, filename in plot_groups:
        fig, axes = plt.subplots(len(dims), 1, figsize=(12, 8), sharex=True)
        mode_str = 'Comparison' if comparing else ''
        fig.suptitle(f'{agent_name} - {group_title} {mode_str}'.rstrip(), fontsize=12, fontweight='bold')

        all_data = []
        dim_data = []
        for dim, ylabel in dims:
            da = get_red_agent_data(df_a, red_idx, dim)
            db = get_red_agent_data(df_b, red_idx, dim) if comparing else None
            all_data.append(da)
            if db is not None:
                all_data.append(db)
            dim_data.append((da, db, ylabel))

        max_dev = compute_max_deviation_from_start(all_data)

        for i, (da, db, ylabel) in enumerate(dim_data):
            center = da[0]
            ylim = (center - max_dev, center + max_dev)
            plot_traces(axes[i], ylabel,
                        time_a, da, label_a,
                        time_b, db, label_b,
                        ylim=ylim)

        axes[-1].set_xlabel('Time [s]')
        fig.tight_layout()

        if output_dir:
            fig.savefig(output_dir / f'red{red_idx}_{filename}.png', dpi=150)

        if not show:
            plt.close(fig)

    if show:
        plt.show()


def create_combined_plot(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame = None,
    label_a: str = 'Hardware',
    label_b: str = 'Simulation',
    output_dir: Path = None,
    show: bool = True,
):
    """Create a single combined figure with all agents and state variables.

    When *df_b* is ``None`` only *df_a* is plotted.

    Args:
        df_a: First (or only) observation DataFrame.
        df_b: Optional second DataFrame for comparison.
        label_a: Legend label for df_a traces.
        label_b: Legend label for df_b traces.
        output_dir: Directory to save plot (None to skip saving).
        show: Whether to display plot interactively.
    """
    fmt_a = detect_obs_format(df_a)
    fmt_b = detect_obs_format(df_b) if df_b is not None else None

    time_a = df_a['time'].values
    time_b = df_b['time'].values if df_b is not None else None

    comparing = df_b is not None
    n_red_a = detect_red_agents(df_a)

    # Determine grid layout: blue agents + red agents
    n_blue = 2
    n_total_cols = n_blue + n_red_a
    title = f'{label_a} vs {label_b} Observation Comparison' if comparing else 'Observation Data'

    # State groups for blue (4 rows) - red only has 3 (no body rates in raw state)
    blue_state_groups = [
        ('Position', [('pos_x', 'X'), ('pos_y', 'Y'), ('pos_z', 'Z')], '[m]'),
        ('Velocity', [('vel_x', 'Vx'), ('vel_y', 'Vy'), ('vel_z', 'Vz')], '[m/s]'),
        ('RPY', [('roll', 'R'), ('pitch', 'P'), ('yaw', 'Y')], '[rad]'),
        ('Body Rates', [('roll_rate', 'dR'), ('pitch_rate', 'dP'), ('yaw_rate', 'dY')], '[rad/s]'),
    ]

    red_state_groups = [
        ('Position', [('pos_x', 'X'), ('pos_y', 'Y'), ('pos_z', 'Z')], '[m]'),
        ('Velocity', [('vel_x', 'Vx'), ('vel_y', 'Vy'), ('vel_z', 'Vz')], '[m/s]'),
        ('RPY', [('roll', 'R'), ('pitch', 'P'), ('yaw', 'Y')], '[rad]'),
    ]

    n_rows = len(blue_state_groups)
    fig, axes = plt.subplots(n_rows, n_total_cols, figsize=(7 * n_total_cols, 12), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # --- Blue agent columns ---
    for row, (group_name, dims, unit) in enumerate(blue_state_groups):
        all_data = []
        for agent_idx in range(n_blue):
            for dim_name, _ in dims:
                all_data.append(get_agent_data(df_a, agent_idx, dim_name, fmt_a))
                if comparing:
                    all_data.append(get_agent_data(df_b, agent_idx, dim_name, fmt_b))
        max_dev = compute_max_deviation_from_start(all_data)

        for col, agent_idx in enumerate(range(n_blue)):
            ax = axes[row, col]

            agent_data_a = []
            agent_data_b = []
            for dim_name, _ in dims:
                agent_data_a.append(get_agent_data(df_a, agent_idx, dim_name, fmt_a))
                if comparing:
                    agent_data_b.append(get_agent_data(df_b, agent_idx, dim_name, fmt_b))

            center = np.mean([d[0] for d in agent_data_a])
            ylim = (center - max_dev, center + max_dev)

            colors = ['tab:blue', 'tab:orange', 'tab:green']
            for dim_idx, (dim_name, dim_label) in enumerate(dims):
                da = agent_data_a[dim_idx]
                ax.plot(time_a, da, '-', color=colors[dim_idx],
                        label=f'{dim_label} {label_a}' if comparing else dim_label,
                        linewidth=1.2, alpha=0.8)
                if comparing:
                    db = agent_data_b[dim_idx]
                    ax.plot(time_b, db, '--', color=colors[dim_idx],
                            label=f'{dim_label} {label_b}', linewidth=1.2, alpha=0.8)

            ax.set_ylabel(f'{group_name} {unit}', fontsize=9)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=7, ncol=2 if comparing else 1)

            if row == 0:
                ax.set_title(f'Blue {agent_idx}', fontsize=11, fontweight='bold')

    # --- Red agent columns ---
    n_red_b = detect_red_agents(df_b) if df_b is not None else 0
    comparing_red = n_red_b > 0

    if n_red_a > 0:
        for row, (group_name, dims, unit) in enumerate(red_state_groups):
            all_data = []
            for red_idx in range(n_red_a):
                for dim_name, _ in dims:
                    all_data.append(get_red_agent_data(df_a, red_idx, dim_name))
                    if comparing_red and red_idx < n_red_b:
                        all_data.append(get_red_agent_data(df_b, red_idx, dim_name))
            max_dev = compute_max_deviation_from_start(all_data)

            for red_idx in range(n_red_a):
                col = n_blue + red_idx
                ax = axes[row, col]

                red_data_a = [get_red_agent_data(df_a, red_idx, dim_name) for dim_name, _ in dims]
                red_data_b = ([get_red_agent_data(df_b, red_idx, dim_name) for dim_name, _ in dims]
                              if comparing_red and red_idx < n_red_b else None)

                center = np.mean([d[0] for d in red_data_a])
                ylim = (center - max_dev, center + max_dev)

                colors = ['tab:red', 'tab:purple', 'darkgreen']
                for dim_idx, (dim_name, dim_label) in enumerate(dims):
                    da = red_data_a[dim_idx]
                    ax.plot(time_a, da, '-', color=colors[dim_idx],
                            label=f'{dim_label} {label_a}' if comparing_red else dim_label,
                            linewidth=1.2, alpha=0.8)
                    if red_data_b is not None:
                        db = red_data_b[dim_idx]
                        ax.plot(time_b, db, '--', color=colors[dim_idx],
                                label=f'{dim_label} {label_b}', linewidth=1.2, alpha=0.8)

                ax.set_ylabel(f'{group_name} {unit}', fontsize=9)
                ax.set_ylim(ylim)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=7, ncol=2 if comparing_red else 1)

                if row == 0:
                    ax.set_title(f'Red {red_idx}', fontsize=11, fontweight='bold')

        # Hide unused cells (body rates row for red agents)
        for red_idx in range(n_red_a):
            col = n_blue + red_idx
            for row in range(len(red_state_groups), n_rows):
                axes[row, col].set_visible(False)

    for col in range(n_total_cols):
        # Find the last visible axis in this column
        for row in range(n_rows - 1, -1, -1):
            if axes[row, col].get_visible():
                axes[row, col].set_xlabel('Time [s]')
                break

    fig.tight_layout()

    if output_dir:
        fig.savefig(output_dir / 'combined_comparison.png', dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot or compare observation CSVs (hardware / simulation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single CSV (simulation only)
    python compare_obs_plots.py ep003_sim.csv -o plots/

    # Compare hardware vs simulation
    python compare_obs_plots.py rosbag_obs.csv --eval eval_sim.csv -o plots/
        """,
    )
    parser.add_argument(
        'csv',
        type=str,
        help='Path to observation CSV (plotted as first / only trace)',
    )
    parser.add_argument(
        '--eval',
        type=str,
        default=None,
        help='Optional second CSV for comparison overlay (e.g., simulation eval CSV)',
    )
    parser.add_argument(
        '--label-a',
        type=str,
        default=None,
        help='Legend label for the first CSV (default: "Hardware" in compare mode, omitted in single mode)',
    )
    parser.add_argument(
        '--label-b',
        type=str,
        default='Simulation',
        help='Legend label for the second CSV (default: "Simulation")',
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for saved plots (default: display only)',
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots interactively (useful for batch processing)',
    )
    parser.add_argument(
        '--combined-only',
        action='store_true',
        help='Only create the combined comparison plot',
    )

    args = parser.parse_args()

    # ── Load CSVs ───────────────────────────────────────────────────────────
    csv_a = Path(args.csv)
    if not csv_a.exists():
        print(f"Error: CSV not found: {csv_a}")
        return 1

    comparing = args.eval is not None
    label_a = args.label_a if args.label_a else ('Hardware' if comparing else 'Sim')
    label_b = args.label_b

    print(f"Loading data: {csv_a}")
    df_a = load_obs_csv(csv_a)
    fmt_a = detect_obs_format(df_a)
    print(f"  Samples: {len(df_a)}, Duration: {df_a['time'].iloc[-1]:.2f}s, Format: {fmt_a}-dim")

    df_b = None
    if comparing:
        csv_b = Path(args.eval)
        if not csv_b.exists():
            print(f"Error: Eval CSV not found: {csv_b}")
            return 1
        print(f"Loading comparison data: {csv_b}")
        df_b = load_obs_csv(csv_b)
        fmt_b = detect_obs_format(df_b)
        print(f"  Samples: {len(df_b)}, Duration: {df_b['time'].iloc[-1]:.2f}s, Format: {fmt_b}-dim")

    # ── Output directory ────────────────────────────────────────────────────
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving plots to: {output_dir}")

    show = not args.no_show

    # ── Create plots ────────────────────────────────────────────────────────
    n_red = detect_red_agents(df_a)
    if n_red > 0:
        print(f"Detected {n_red} red (pursuer) agent(s) in data")

    if args.combined_only:
        print("Creating combined plot...")
        create_combined_plot(df_a, df_b, label_a, label_b, output_dir, show)
    else:
        print("Creating individual agent plots...")
        for agent_idx in [0, 1]:
            print(f"  Blue {agent_idx}...")
            create_agent_plots(df_a, agent_idx, df_b, label_a, label_b, output_dir, show=False)

        for red_idx in range(n_red):
            print(f"  Red {red_idx}...")
            create_red_agent_plots(df_a, red_idx, df_b, label_a, label_b, output_dir, show=False)

        print("Creating combined plot...")
        create_combined_plot(df_a, df_b, label_a, label_b, output_dir, show)

    if output_dir:
        print(f"Plots saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
