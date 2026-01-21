"""Visual test for box_random spawn method.

Plots spawn positions for multiple resets to verify:
1. Blue agents spawn in the correct box region
2. Red agents spawn in the correct box region
3. Minimum distance constraint is satisfied
4. Z coordinate is randomized within specified range
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig


def plot_box_3d(ax, x_min, x_max, y_min, y_max, z_min, z_max, color, alpha=0.1):
    """Plot a 3D box wireframe with transparent faces."""
    # Define the 8 vertices of the box
    vertices = [
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max]
    ]

    # Define the 6 faces using vertex indices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
    ]

    # Add the box faces
    ax.add_collection3d(Poly3DCollection(
        faces, alpha=alpha, facecolor=color, edgecolor=color, linewidth=1
    ))


def plot_spawn_samples_3d(n_samples: int = 6, n_worlds_to_show: int = 4):
    """Plot spawn positions in 3D for multiple resets.

    Args:
        n_samples: Number of reset samples to show.
        n_worlds_to_show: Number of worlds to show per sample.
    """
    # Create environment with box_random spawn
    cfg = RedVsBlueEnvConfig(
        n_worlds=16,
        n_pairs=2,
        spawn_method="box_random",
        min_spawn_distance=0.5,
        # Box parameters
        blue_box_x_min=1.5,
        blue_box_x_max=2.5,
        blue_box_y_half=1.0,
        red_box_x_min=-0.5,
        red_box_x_max=0.5,
        red_box_y_half=1.0,
        # Z spawn range
        spawn_z_min=0.8,
        spawn_z_max=1.2,
    )
    env = RedVsBlueEnv(cfg=cfg)

    # Create figure with 3D subplots
    fig = plt.figure(figsize=(16, 10))

    for sample_idx in range(n_samples):
        ax = fig.add_subplot(2, 3, sample_idx + 1, projection='3d')

        # Reset environment
        env.reset(seed=sample_idx * 42)
        pos = np.asarray(env.sim.data.states.pos)

        # Plot spawn boxes
        plot_box_3d(
            ax,
            cfg.blue_box_x_min, cfg.blue_box_x_max,
            -cfg.blue_box_y_half, cfg.blue_box_y_half,
            cfg.spawn_z_min, cfg.spawn_z_max,
            color='blue', alpha=0.1
        )
        plot_box_3d(
            ax,
            cfg.red_box_x_min, cfg.red_box_x_max,
            -cfg.red_box_y_half, cfg.red_box_y_half,
            cfg.spawn_z_min, cfg.spawn_z_max,
            color='red', alpha=0.1
        )

        # Plot agents for multiple worlds
        for world_idx in range(n_worlds_to_show):
            alpha = 0.3 + 0.7 * (world_idx == 0)  # First world is more opaque

            # Blue agents
            blue_pos = pos[world_idx, :cfg.n_blue, :]
            ax.scatter(
                blue_pos[:, 0], blue_pos[:, 1], blue_pos[:, 2],
                c='blue', s=100, alpha=alpha, marker='o',
                edgecolors='darkblue', linewidths=1.5
            )

            # Red agents
            red_pos = pos[world_idx, cfg.n_blue:, :]
            ax.scatter(
                red_pos[:, 0], red_pos[:, 1], red_pos[:, 2],
                c='red', s=100, alpha=alpha, marker='^',
                edgecolors='darkred', linewidths=1.5
            )

        # Check minimum distances
        min_dists = []
        z_values = []
        for world_idx in range(cfg.n_worlds):
            all_pos = pos[world_idx, :, :]
            diff = all_pos[:, None, :] - all_pos[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            min_dists.append(np.min(dist))
            z_values.extend(all_pos[:, 2].tolist())

        min_dist_all = np.min(min_dists)
        violations = sum(1 for d in min_dists if d < cfg.min_spawn_distance)
        z_range = (np.min(z_values), np.max(z_values))

        ax.set_xlim(-1.5, 3.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0.5, 1.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Sample {sample_idx + 1} (seed={sample_idx * 42})\n'
                     f'Min dist: {min_dist_all:.3f}m, Violations: {violations}/{cfg.n_worlds}\n'
                     f'Z range: [{z_range[0]:.3f}, {z_range[1]:.3f}]')

    env.close()

    plt.suptitle(f'Box Random Spawn Test (3D)\n'
                 f'Blue box: x∈[{cfg.blue_box_x_min}, {cfg.blue_box_x_max}], '
                 f'y∈[{-cfg.blue_box_y_half}, {cfg.blue_box_y_half}], '
                 f'z∈[{cfg.spawn_z_min}, {cfg.spawn_z_max}]\n'
                 f'Red box: x∈[{cfg.red_box_x_min}, {cfg.red_box_x_max}], '
                 f'y∈[{-cfg.red_box_y_half}, {cfg.red_box_y_half}], '
                 f'z∈[{cfg.spawn_z_min}, {cfg.spawn_z_max}]\n'
                 f'Min spawn distance: {cfg.min_spawn_distance}m',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig('box_random_spawn_test_3d.png', dpi=150, bbox_inches='tight')
    print(f'Saved plot to box_random_spawn_test_3d.png')
    plt.show()


def plot_distance_histogram(n_resets: int = 100):
    """Plot histogram of minimum pairwise distances across many resets.

    Args:
        n_resets: Number of resets to sample.
    """
    cfg = RedVsBlueEnvConfig(
        n_worlds=64,
        n_pairs=2,
        spawn_method="box_random",
        min_spawn_distance=0.5,
        spawn_z_min=0.8,
        spawn_z_max=1.2,
    )
    env = RedVsBlueEnv(cfg=cfg)

    all_min_dists = []
    all_z_values = []

    for i in range(n_resets):
        env.reset(seed=i)
        pos = np.asarray(env.sim.data.states.pos)

        for world_idx in range(cfg.n_worlds):
            all_pos = pos[world_idx, :, :]
            diff = all_pos[:, None, :] - all_pos[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            all_min_dists.append(np.min(dist))
            all_z_values.extend(all_pos[:, 2].tolist())

    env.close()

    all_min_dists = np.array(all_min_dists)
    all_z_values = np.array(all_z_values)
    violations = np.sum(all_min_dists < cfg.min_spawn_distance)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distance histogram
    ax1 = axes[0]
    ax1.hist(all_min_dists, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(cfg.min_spawn_distance, color='red', linestyle='--', linewidth=2,
                label=f'Min distance threshold ({cfg.min_spawn_distance}m)')
    ax1.set_xlabel('Minimum Pairwise Distance (m)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Distribution of Minimum Spawn Distances\n'
                  f'{n_resets} resets × {cfg.n_worlds} worlds = {len(all_min_dists)} samples\n'
                  f'Violations: {violations} ({100*violations/len(all_min_dists):.2f}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Z distribution histogram
    ax2 = axes[1]
    ax2.hist(all_z_values, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(cfg.spawn_z_min, color='red', linestyle='--', linewidth=2,
                label=f'Z bounds [{cfg.spawn_z_min}, {cfg.spawn_z_max}]')
    ax2.axvline(cfg.spawn_z_max, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Z Coordinate (m)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Distribution of Spawn Z Coordinates\n'
                  f'Expected range: [{cfg.spawn_z_min}, {cfg.spawn_z_max}]\n'
                  f'Actual range: [{all_z_values.min():.3f}, {all_z_values.max():.3f}]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('box_random_spawn_histogram.png', dpi=150, bbox_inches='tight')
    print(f'Saved histogram to box_random_spawn_histogram.png')
    plt.show()


if __name__ == "__main__":
    print("Plotting spawn samples (3D)...")
    plot_spawn_samples_3d()

    print("\nPlotting distance histogram...")
    plot_distance_histogram()
