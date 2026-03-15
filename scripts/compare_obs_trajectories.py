#!/usr/bin/env python3
"""Compare CrazySim (HW/rosbag) vs Crazyflow (sim) observation trajectories.

Generates per-agent plots (position, velocity, RPY, body rates) and a combined
comparison, matching the style of the existing hw_pid_vs_sim plots.

Usage:
    python scripts/compare_obs_trajectories.py <hw_csv> <sim_csv> [--outdir DIR]

If --outdir is not specified, plots are saved next to the sim_csv in a
crazysim_vs_crazyflow/ subdirectory.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


# ── Color palettes ──────────────────────────────────────────────────────────
# [hw, sim, extra] per agent type
EVADER_COLORS = ['#003399', '#4488EE', '#88CCFF']   # dark→light blue
PURSUER_COLORS = ['#8B0000', '#FF6600', '#CC0077']  # deep red, orange, red-magenta

def source_colors(agent_name: str) -> list:
    """Return [hw_color, sim_color, extra_color] for the agent."""
    return EVADER_COLORS if agent_name.startswith('blue') else PURSUER_COLORS


# ── Obs layout constants ────────────────────────────────────────────────────
# HW (rosbag) obs: pos(3) vel(3) rpy(3) rpy_rates(3) [ally_onehot ...shared]
HW_POS = slice(0, 3)
HW_VEL = slice(3, 6)
HW_RPY = slice(6, 9)
HW_RPY_RATES = slice(9, 12)

# Sim obs: pos(3) vel(3) rotmat_flat(9) body_rates(3) [ally_onehot ...shared]
SIM_POS = slice(0, 3)
SIM_VEL = slice(3, 6)
SIM_ROTMAT = slice(6, 15)
SIM_BODY_RATES = slice(15, 18)


def rotmat_to_rpy(rotmat_flat: np.ndarray) -> np.ndarray:
    """Convert flattened rotation matrices (T, 9) → RPY (T, 3) via scipy."""
    T = rotmat_flat.shape[0]
    mats = rotmat_flat.reshape(T, 3, 3)
    return Rotation.from_matrix(mats).as_euler("xyz")


def ang_vel_to_rpy_rates(rotmat_flat: np.ndarray, ang_vel: np.ndarray) -> np.ndarray:
    """Convert body angular velocity to Euler rates using the W matrix.

    rpy_rates = W @ ang_vel, where W depends on roll (phi) and pitch (theta).
    """
    rpy = rotmat_to_rpy(rotmat_flat)
    phi = rpy[:, 0]
    theta = rpy[:, 1]

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    # Avoid division by zero
    cos_theta = np.where(np.abs(cos_theta) < 1e-8, 1e-8, cos_theta)
    tan_theta = np.tan(theta)

    p = ang_vel[:, 0]
    q = ang_vel[:, 1]
    r = ang_vel[:, 2]

    phi_dot = p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
    theta_dot = cos_phi * q - sin_phi * r
    psi_dot = sin_phi / cos_theta * q + cos_phi / cos_theta * r

    return np.stack([phi_dot, theta_dot, psi_dot], axis=-1)


def load_hw_csv(path: str) -> dict:
    """Load HW/rosbag obs CSV. Returns dict of agent data."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    t = np.array([float(data["time"][i]) for i in range(len(data))])

    # Detect number of blue agents
    n_blue = 0
    while f"agent{n_blue}_obs0" in data.dtype.names:
        n_blue += 1

    # Detect number of red agents
    n_red = 0
    while f"red{n_red}_pos_x" in data.dtype.names:
        n_red += 1

    agents = {}
    for i in range(n_blue):
        prefix = f"agent{i}"
        obs = np.column_stack([data[f"{prefix}_obs{j}"] for j in range(46)])
        agents[f"blue{i}"] = {
            "pos": obs[:, HW_POS],
            "vel": obs[:, HW_VEL],
            "rpy": obs[:, HW_RPY],
            "rpy_rates": obs[:, HW_RPY_RATES],
        }

    for i in range(n_red):
        prefix = f"red{i}"
        agents[f"red{i}"] = {
            "pos": np.column_stack([data[f"{prefix}_pos_{c}"] for c in "xyz"]),
            "vel": np.column_stack([data[f"{prefix}_vel_{c}"] for c in "xyz"]),
            "rpy": np.column_stack([data[f"{prefix}_{c}"] for c in ["roll", "pitch", "yaw"]]),
            "rpy_rates": np.full((len(t), 3), np.nan),  # Not available for red in HW
        }

    return {"time": t, "agents": agents, "n_blue": n_blue, "n_red": n_red}


def load_sim_csv(path: str) -> dict:
    """Load sim obs CSV. Returns dict of agent data."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    t = np.array([float(data["time"][i]) for i in range(len(data))])

    # Detect number of blue agents
    n_blue = 0
    while f"agent{n_blue}_obs0" in data.dtype.names:
        n_blue += 1

    # Detect obs dimension
    n_obs = 0
    while f"agent0_obs{n_obs}" in data.dtype.names:
        n_obs += 1

    # Detect number of red agents
    n_red = 0
    while f"red{n_red}_pos_x" in data.dtype.names:
        n_red += 1

    agents = {}
    for i in range(n_blue):
        prefix = f"agent{i}"
        obs = np.column_stack([data[f"{prefix}_obs{j}"] for j in range(n_obs)])

        if n_obs > 46:
            # Sim format: rotmat(9) + body_rates(3)
            pos = obs[:, SIM_POS]
            vel = obs[:, SIM_VEL]
            rotmat_flat = obs[:, SIM_ROTMAT]
            body_rates = obs[:, SIM_BODY_RATES]
            rpy = rotmat_to_rpy(rotmat_flat)
            rpy_rates = ang_vel_to_rpy_rates(rotmat_flat, body_rates)
        else:
            # HW format in sim CSV (fallback)
            pos = obs[:, HW_POS]
            vel = obs[:, HW_VEL]
            rpy = obs[:, HW_RPY]
            rpy_rates = obs[:, HW_RPY_RATES]
            body_rates = rpy_rates  # Approximate

        agents[f"blue{i}"] = {
            "pos": pos,
            "vel": vel,
            "rpy": rpy,
            "rpy_rates": rpy_rates,
        }

    for i in range(n_red):
        prefix = f"red{i}"
        agents[f"red{i}"] = {
            "pos": np.column_stack([data[f"{prefix}_pos_{c}"] for c in "xyz"]),
            "vel": np.column_stack([data[f"{prefix}_vel_{c}"] for c in "xyz"]),
            "rpy": np.column_stack([data[f"{prefix}_{c}"] for c in ["roll", "pitch", "yaw"]]),
            "rpy_rates": np.full((len(t), 3), np.nan),
        }

    return {"time": t, "agents": agents, "n_blue": n_blue, "n_red": n_red}


def detect_motion_start(data: dict, threshold: float = 1e-3) -> float:
    """Detect when motion starts by finding first significant pitch change in blue0."""
    agents = data["agents"]
    t = data["time"]
    if "blue0" not in agents:
        return 0.0
    # Use pitch (index 1 of rpy) as the motion indicator — it responds first to forward commands
    pitch = agents["blue0"]["rpy"][:, 1]
    for i in range(len(t)):
        if abs(pitch[i]) > threshold:
            return t[i]
    return 0.0


def align_time(hw: dict, sim: dict, align_motion: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Find overlapping time range and return index arrays for hw and sim.

    If align_motion=True, shift HW time so first motion aligns with sim first motion.
    """
    if align_motion:
        hw_start = detect_motion_start(hw)
        sim_start = detect_motion_start(sim)
        offset = hw_start - sim_start
        hw["time"] = hw["time"] - offset
        print(f"Motion alignment: HW starts at {hw_start:.3f}s, Sim at {sim_start:.3f}s, "
              f"shifting HW by {-offset:.3f}s")

    # Trim any negative times after shifting
    hw_pos_mask = hw["time"] >= 0
    sim_pos_mask = sim["time"] >= 0

    t_end = min(hw["time"][hw_pos_mask][-1], sim["time"][sim_pos_mask][-1])
    hw_mask = hw_pos_mask & (hw["time"] <= t_end)
    sim_mask = sim_pos_mask & (sim["time"] <= t_end)
    # Also return full masks (no t_end trim) for animation
    hw_mask_full = hw_pos_mask
    sim_mask_full = sim_pos_mask
    return hw_mask, sim_mask, hw_mask_full, sim_mask_full


def plot_3panel(
    t_hw, t_sim, hw_data, sim_data, labels, ylabel_unit, title, save_path,
    hw_label="CrazySim", sim_label="Crazyflow", ylims=None, shared_ylim=False,
    t_extra=None, extra_data=None, extra_label=None, agent_name="blue0",
):
    """Plot 3-panel comparison (e.g., X/Y/Z or Vx/Vy/Vz).

    Args:
        ylims: Optional list of (ymin, ymax) tuples for each panel.
        shared_ylim: If True, use the same y-axis range (max extent) for all panels.
        t_extra/extra_data/extra_label: Optional third dataset.
        agent_name: Used to pick evader (blue shades) or pursuer (red shades).
    """
    colors = source_colors(agent_name)  # [hw, sim, extra]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, lab, col in zip(axes, labels, range(3)):
        ax.plot(t_hw, hw_data[:, col], color=colors[0], linestyle="-", linewidth=1.5, label=hw_label)
        ax.plot(t_sim, sim_data[:, col], color=colors[1], linestyle="--", linewidth=1.5, label=sim_label)
        if t_extra is not None and extra_data is not None:
            ax.plot(t_extra, extra_data[:, col], color=colors[2], linestyle="-.", linewidth=1.5, label=extra_label)
        ax.set_ylabel(f"{lab} [{ylabel_unit}]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        if ylims and col < len(ylims) and ylims[col] is not None:
            ax.set_ylim(ylims[col])

    if shared_ylim and not ylims:
        # Same range size for all panels, but each centered on its own data
        max_range = 0
        panel_centers = []
        for col in range(3):
            arrays = [hw_data[:, col], sim_data[:, col]]
            if extra_data is not None:
                arrays.append(extra_data[:, col])
            vals = np.concatenate(arrays)
            vmin, vmax = vals.min(), vals.max()
            panel_centers.append((vmin + vmax) / 2)
            max_range = max(max_range, vmax - vmin)
        half = max_range * 1.1 / 2  # 10% margin
        for ax, center in zip(axes, panel_centers):
            ax.set_ylim(center - half, center + half)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined(t_hw, t_sim, hw_agents, sim_agents, agent_names, save_path,
                  hw_label="CrazySim", sim_label="Crazyflow",
                  t_extra=None, extra_agents=None, extra_label=None):
    """Plot combined comparison grid: agents × (pos, vel, rpy)."""
    n_agents = len(agent_names)
    fig, axes = plt.subplots(3, n_agents, figsize=(5 * n_agents, 10), squeeze=False)
    title = f"{hw_label} vs {sim_label}"
    if extra_label:
        title += f" vs {extra_label}"
    fig.suptitle(f"{title} — Observation Comparison", fontsize=14, fontweight="bold")

    row_configs = [
        ("pos", ["X", "Y", "Z"], "m"),
        ("vel", ["Vx", "Vy", "Vz"], "m/s"),
        ("rpy", ["Roll", "Pitch", "Yaw"], "rad"),
    ]

    # Component alphas within each source line (dim each component slightly)
    component_alphas = [1.0, 0.7, 0.45]
    linestyles = ["-", "--", "-."]

    for col_idx, name in enumerate(agent_names):
        display = name.replace("blue", "Blue ").replace("red", "Red ")
        hw_ag = hw_agents[name]
        sim_ag = sim_agents[name]
        src_colors = source_colors(name)  # [hw, sim, extra]

        for row_idx, (key, labels, unit) in enumerate(row_configs):
            ax = axes[row_idx, col_idx]
            has_extra = t_extra is not None and extra_agents is not None and name in extra_agents

            for k, lab in enumerate(labels):
                alpha = component_alphas[k]
                ax.plot(t_hw, hw_ag[key][:, k], color=src_colors[0], linestyle=linestyles[0],
                        linewidth=1.2, label=f"{lab} {hw_label}", alpha=alpha)
                ax.plot(t_sim, sim_ag[key][:, k], color=src_colors[1], linestyle=linestyles[1],
                        linewidth=1.2, label=f"{lab} {sim_label}", alpha=alpha)
                if has_extra:
                    extra_ag = extra_agents[name]
                    ax.plot(t_extra, extra_ag[key][:, k], color=src_colors[2], linestyle=linestyles[2],
                            linewidth=1.2, label=f"{lab} {extra_label}", alpha=alpha)

            if row_idx == 0:
                ax.set_title(display, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{', '.join(labels)} [{unit}]")
            ncol = 3 if has_extra else 2
            ax.legend(fontsize=6, ncol=ncol, loc="upper right")
            ax.grid(True, alpha=0.3)
            if row_idx == 2:
                ax.set_xlabel("Time [s]")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_3d_trajectories(t_hw, t_sim, hw_agents, sim_agents, agent_names,
                         save_path, hw_label="CrazySim", sim_label="Crazyflow",
                         extra_agents=None, extra_label=None):
    """Plot 3D trajectories for all agents. Evaders (blue) in blue, pursuers (red) in red."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    all_pos_arrays = []
    for name in agent_names:
        src_colors = source_colors(name)
        hw_pos = hw_agents[name]['pos']
        sim_pos = sim_agents[name]['pos']
        label_prefix = name.replace('blue', 'Blue ').replace('red', 'Red ')

        ax.plot(hw_pos[:, 0], hw_pos[:, 1], hw_pos[:, 2],
                color=src_colors[0], linewidth=1.5, label=f"{label_prefix} {hw_label}")
        ax.scatter(hw_pos[0, 0], hw_pos[0, 1], hw_pos[0, 2],
                   color=src_colors[0], marker='o', s=40)

        ax.plot(sim_pos[:, 0], sim_pos[:, 1], sim_pos[:, 2],
                color=src_colors[1], linewidth=1.5, linestyle='--',
                label=f"{label_prefix} {sim_label}")
        ax.scatter(sim_pos[0, 0], sim_pos[0, 1], sim_pos[0, 2],
                   color=src_colors[1], marker='x', s=40)

        all_pos_arrays += [hw_pos, sim_pos]

        if extra_agents is not None and name in extra_agents:
            ex_pos = extra_agents[name]['pos']
            ax.plot(ex_pos[:, 0], ex_pos[:, 1], ex_pos[:, 2],
                    color=src_colors[2], linewidth=1.5, linestyle='-.',
                    label=f"{label_prefix} {extra_label}")
            ax.scatter(ex_pos[0, 0], ex_pos[0, 1], ex_pos[0, 2],
                       color=src_colors[2], marker='^', s=40)
            all_pos_arrays.append(ex_pos)

    # Equal aspect ratio
    all_pts = np.concatenate(all_pos_arrays, axis=0)
    margin = 0.15
    mins = all_pts.min(axis=0) - margin
    maxs = all_pts.max(axis=0) + margin
    centers = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    title = f'{hw_label} vs {sim_label}'
    if extra_label:
        title += f' vs {extra_label}'
    ax.set_title(f'{title} — 3D Trajectories')

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    leg_elems = [
        Patch(color='none', label=rf'$\mathbf{{{hw_label}}}$'),
        Line2D([0], [0], color=EVADER_COLORS[0], lw=2, label='Evader'),
        Line2D([0], [0], color=PURSUER_COLORS[0], lw=2, label='Pursuer'),
        Patch(color='none', label=rf'$\mathbf{{{sim_label}}}$'),
        Line2D([0], [0], color=EVADER_COLORS[1], lw=2, ls='--', label='Evader'),
        Line2D([0], [0], color=PURSUER_COLORS[1], lw=2, ls='--', label='Pursuer'),
    ]
    if extra_label:
        leg_elems += [
            Patch(color='none', label=rf'$\mathbf{{{extra_label}}}$'),
            Line2D([0], [0], color=EVADER_COLORS[2], lw=2, ls='-.', label='Evader'),
            Line2D([0], [0], color=PURSUER_COLORS[2], lw=2, ls='-.', label='Pursuer'),
        ]
    ax.legend(handles=leg_elems, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _rpy_to_rotation_matrix(roll, pitch, yaw):
    """Convert RPY angles to a 3x3 rotation matrix (ZYX convention)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])
    return R


def animate_3d_trajectories(t_hw, t_sim, hw_agents, sim_agents, agent_names,
                             save_path, hw_label="CrazySim", sim_label="Crazyflow",
                             fps=50, arm_len=0.04, slowdown=3,
                             hw_runs=None,
                             t_extra=None, extra_agents=None, extra_label=None):
    """Create animated 3D GIF with oriented cross markers at current drone positions.

    Each drone is drawn as an X-shaped cross oriented by its RPY angles.
    Trails show past trajectory.

    If hw_runs is provided, it should be a list of (t_hw, hw_agents) tuples for
    multiple consecutive reruns. The GIF plays each run sequentially, resetting
    trails between runs, while the camera rotates 360 degrees over all runs.
    """
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.interpolate import interp1d

    def interp_agent(t_src, agent_data, t_out):
        result = {}
        for key in ["pos", "rpy"]:
            cols = []
            for k in range(3):
                f = interp1d(t_src, agent_data[key][:, k], kind="linear",
                             bounds_error=False, fill_value=(agent_data[key][0, k],
                                                             agent_data[key][-1, k]))
                cols.append(f(t_out))
            result[key] = np.column_stack(cols)
        return result

    # Build list of runs: each run = (hw_interp, sim_interp, n_frames)
    if hw_runs is None:
        hw_runs = [(t_hw, hw_agents)]

    dt = 1.0 / fps
    runs = []
    all_pos_arrays = []

    for t_hw_r, hw_agents_r in hw_runs:
        t_max = min(t_hw_r[-1], t_sim[-1])
        t_frames_r = np.arange(0, t_max + dt, dt)
        hw_interp_r = {n: interp_agent(t_hw_r, hw_agents_r[n], t_frames_r) for n in agent_names}
        sim_interp_r = {n: interp_agent(t_sim, sim_agents[n], t_frames_r) for n in agent_names}
        runs.append((t_frames_r, hw_interp_r, sim_interp_r))
        for n in agent_names:
            all_pos_arrays.append(hw_interp_r[n]["pos"])
            all_pos_arrays.append(sim_interp_r[n]["pos"])

    # Compute axis limits from all runs — equal aspect ratio
    all_pos = np.concatenate(all_pos_arrays, axis=0)
    margin = 0.15
    mins = all_pos.min(axis=0) - margin
    maxs = all_pos.max(axis=0) + margin
    centers = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2
    xlim = (centers[0] - max_range, centers[0] + max_range)
    ylim = (centers[1] - max_range, centers[1] + max_range)
    zlim = (centers[2] - max_range, centers[2] + max_range)

    # Build frame index → (run_idx, local_frame_idx) mapping
    # Add pause frames (0.5s worth) at the end of each run
    pause_frames = int(1.5 * fps / slowdown)
    frame_map = []
    for run_idx, (t_frames_r, _, _) in enumerate(runs):
        for local_idx in range(len(t_frames_r)):
            frame_map.append((run_idx, local_idx))
        last_idx = len(t_frames_r) - 1
        for _ in range(pause_frames):
            frame_map.append((run_idx, last_idx))
    total_frames = len(frame_map)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    def draw_cross(ax, pos, rpy, color, linestyle='-', alpha=1.0, lw=2.0):
        R = _rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
        arm1 = np.array([arm_len, arm_len, 0.0])
        arm2 = np.array([arm_len, -arm_len, 0.0])
        for arm in [arm1, arm2]:
            tip_a = pos + R @ arm
            tip_b = pos - R @ arm
            ax.plot([tip_a[0], tip_b[0]], [tip_a[1], tip_b[1]], [tip_a[2], tip_b[2]],
                    color=color, linestyle=linestyle, linewidth=lw, alpha=alpha)

    elev = 25
    n_runs = len(runs)

    def update(global_frame):
        ax.cla()
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')

        run_idx, local_idx = frame_map[global_frame]
        t_frames_r, hw_interp_r, sim_interp_r = runs[run_idx]
        t_now = t_frames_r[local_idx]

        # Smooth rotation: 360 degrees over all frames
        ax.view_init(elev=elev, azim=-60 + global_frame * 360 / max(total_frames - 1, 1))

        trail = slice(0, local_idx + 1)

        for name in agent_names:
            src_colors = source_colors(name)

            hw_pos = hw_interp_r[name]["pos"]
            hw_rpy = hw_interp_r[name]["rpy"]
            ax.plot(hw_pos[trail, 0], hw_pos[trail, 1], hw_pos[trail, 2],
                    color=src_colors[0], linewidth=1.2, alpha=0.8)
            draw_cross(ax, hw_pos[local_idx], hw_rpy[local_idx], src_colors[0],
                       linestyle='-', lw=2.5)

            sim_pos = sim_interp_r[name]["pos"]
            sim_rpy = sim_interp_r[name]["rpy"]
            ax.plot(sim_pos[trail, 0], sim_pos[trail, 1], sim_pos[trail, 2],
                    color=src_colors[1], linewidth=1.2, linestyle='--', alpha=0.8)
            draw_cross(ax, sim_pos[local_idx], sim_rpy[local_idx], src_colors[1],
                       linestyle='--', lw=2.0, alpha=0.9)

            if extra_agents is not None and name in extra_agents:
                ex_interp = interp_agent(t_extra, extra_agents[name], t_frames_r)
                ex_pos = ex_interp["pos"]
                ex_rpy = ex_interp["rpy"]
                ax.plot(ex_pos[trail, 0], ex_pos[trail, 1], ex_pos[trail, 2],
                        color=src_colors[2], linewidth=1.2, linestyle='-.', alpha=0.8)
                draw_cross(ax, ex_pos[local_idx], ex_rpy[local_idx], src_colors[2],
                           linestyle='-.', lw=2.0, alpha=0.9)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(color='none', label=rf'$\mathbf{{{hw_label}}}$'),
            Line2D([0], [0], color=EVADER_COLORS[0], lw=2, label='Evader'),
            Line2D([0], [0], color=PURSUER_COLORS[0], lw=2, label='Pursuer'),
            Patch(color='none', label=rf'$\mathbf{{{sim_label}}}$'),
            Line2D([0], [0], color=EVADER_COLORS[1], lw=2, ls='--', label='Evader'),
            Line2D([0], [0], color=PURSUER_COLORS[1], lw=2, ls='--', label='Pursuer'),
        ]
        if extra_label:
            legend_elements += [
                Patch(color='none', label=rf'$\mathbf{{{extra_label}}}$'),
                Line2D([0], [0], color=EVADER_COLORS[2], lw=2, ls='-.', label='Evader'),
                Line2D([0], [0], color=PURSUER_COLORS[2], lw=2, ls='-.', label='Pursuer'),
            ]
        title_str = f'{hw_label} vs {sim_label}'
        if extra_label:
            title_str += f' vs {extra_label}'
        run_label = f" (run {run_idx + 1}/{n_runs})" if n_runs > 1 else ""
        ax.set_title(f'{title_str} — t = {t_now:.2f}s{run_label}')
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

    playback_fps = fps / slowdown
    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000/playback_fps)
    ext = Path(save_path).suffix.lower()
    if ext == ".mp4":
        anim.save(str(save_path), writer='ffmpeg', fps=playback_fps, dpi=100,
                  codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    else:
        anim.save(str(save_path), writer='pillow', fps=playback_fps, dpi=100)
    plt.close(fig)
    print(f"Saved animated trajectory: {save_path} "
          f"({total_frames} frames, {playback_fps:.1f} fps playback = 1:{slowdown} speed, "
          f"{n_runs} run(s))")


def compute_rms_errors(t_hw, t_sim, hw_agents, sim_agents, agent_names):
    """Compute RMS errors between hw and sim for each agent and state."""
    # Interpolate sim onto hw time grid for fair comparison
    from scipy.interpolate import interp1d

    t_end = min(t_hw[-1], t_sim[-1])
    t_common = t_hw[t_hw <= t_end]

    print("\n=== RMS Errors (over common time window) ===")
    print(f"Time window: 0 to {t_end:.2f}s ({len(t_common)} samples)\n")
    print(f"{'Agent':<10} {'X [m]':>8} {'Y [m]':>8} {'Z [m]':>8} "
          f"{'Vx[m/s]':>8} {'Vy[m/s]':>8} {'Vz[m/s]':>8} "
          f"{'R [rad]':>8} {'P [rad]':>8} {'Y [rad]':>8}")
    print("-" * 95)

    for name in agent_names:
        hw_ag = hw_agents[name]
        sim_ag = sim_agents[name]
        errors = []
        for key in ["pos", "vel", "rpy"]:
            for k in range(3):
                hw_vals = hw_ag[key][:len(t_common), k]
                # Interpolate sim to hw timestamps
                f_interp = interp1d(t_sim, sim_ag[key][:, k], kind="linear",
                                    fill_value="extrapolate")
                sim_interp = f_interp(t_common)
                rms = np.sqrt(np.mean((hw_vals - sim_interp) ** 2))
                errors.append(rms)

        display = name.replace("blue", "Blue").replace("red", "Red")
        print(f"{display:<10} " + " ".join(f"{e:8.4f}" for e in errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hw_csv", help="Hardware/rosbag obs CSV")
    parser.add_argument("--sim-csv", required=True, help="Crazyflow sim obs CSV")
    parser.add_argument("--extra-csv", help="Optional third obs CSV (e.g. CrazySim)")
    parser.add_argument("--outdir", help="Output directory for plots")
    parser.add_argument("--hw-label", default="Hardware", help="Label for HW data")
    parser.add_argument("--sim-label", default="Crazyflow", help="Label for sim data")
    parser.add_argument("--extra-label", default="CrazySim", help="Label for extra data")
    parser.add_argument("--align-motion", action="store_true",
                        help="Align HW time to first motion (removes service startup delay)")
    args = parser.parse_args()

    hw = load_hw_csv(args.hw_csv)
    sim = load_sim_csv(args.sim_csv)
    extra = load_hw_csv(args.extra_csv) if args.extra_csv else None

    print(f"HW:  {len(hw['time'])} samples, {hw['time'][-1]:.2f}s, "
          f"{hw['n_blue']} blue + {hw['n_red']} red")
    print(f"Sim: {len(sim['time'])} samples, {sim['time'][-1]:.2f}s, "
          f"{sim['n_blue']} blue + {sim['n_red']} red")
    if extra:
        print(f"Extra ({args.extra_label}): {len(extra['time'])} samples, {extra['time'][-1]:.2f}s, "
              f"{extra['n_blue']} blue + {extra['n_red']} red")

    # Determine output directory
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path(args.sim_csv).parent / "crazysim_vs_crazyflow"
    outdir.mkdir(parents=True, exist_ok=True)

    # Align time for hw vs sim
    hw_mask, sim_mask, hw_mask_full, sim_mask_full = align_time(hw, sim, align_motion=args.align_motion)
    t_hw = hw["time"][hw_mask]
    t_sim = sim["time"][sim_mask]

    hw_agents = {k: {sk: v[hw_mask] for sk, v in ag.items()} for k, ag in hw["agents"].items()}
    sim_agents = {k: {sk: v[sim_mask] for sk, v in ag.items()} for k, ag in sim["agents"].items()}

    # Align extra (CrazySim) independently vs sim
    t_extra = None
    extra_agents = None
    if extra:
        ex_mask, _, _, _ = align_time(extra, sim, align_motion=args.align_motion)
        t_extra = extra["time"][ex_mask]
        extra_agents = {k: {sk: v[ex_mask] for sk, v in ag.items()} for k, ag in extra["agents"].items()}

    # Common agents
    agent_names = sorted(set(hw["agents"].keys()) & set(sim["agents"].keys()))

    print(f"Common agents: {agent_names}")
    print(f"Aligned time: 0 to {min(t_hw[-1], t_sim[-1]):.2f}s")

    # Per-agent plots
    for name in agent_names:
        display = name.replace("blue", "Blue ").replace("red", "Red ")
        hw_ag = hw_agents[name]
        sim_ag = sim_agents[name]
        ex_ag = extra_agents.get(name) if extra_agents else None

        # Position
        plot_3panel(t_hw, t_sim, hw_ag["pos"], sim_ag["pos"],
                    ["X", "Y", "Z"], "m", f"{display} - Position Comparison",
                    outdir / f"{name}_position.png",
                    hw_label=args.hw_label, sim_label=args.sim_label,
                    shared_ylim=True, agent_name=name,
                    t_extra=t_extra, extra_data=ex_ag["pos"] if ex_ag else None,
                    extra_label=args.extra_label if extra else None)

        # Velocity
        plot_3panel(t_hw, t_sim, hw_ag["vel"], sim_ag["vel"],
                    ["Vx", "Vy", "Vz"], "m/s", f"{display} - Velocity Comparison",
                    outdir / f"{name}_velocity.png",
                    hw_label=args.hw_label, sim_label=args.sim_label,
                    shared_ylim=True, agent_name=name,
                    t_extra=t_extra, extra_data=ex_ag["vel"] if ex_ag else None,
                    extra_label=args.extra_label if extra else None)

        # RPY
        plot_3panel(t_hw, t_sim, hw_ag["rpy"], sim_ag["rpy"],
                    ["Roll", "Pitch", "Yaw"], "rad", f"{display} - RPY Angles Comparison",
                    outdir / f"{name}_rpy.png",
                    hw_label=args.hw_label, sim_label=args.sim_label,
                    ylims=[(-0.3, 0.3), (-0.3, 0.3), (-0.15, 0.15)], agent_name=name,
                    t_extra=t_extra, extra_data=ex_ag["rpy"] if ex_ag else None,
                    extra_label=args.extra_label if extra else None)

        # Body rates (skip if NaN for red)
        if not np.all(np.isnan(hw_ag["rpy_rates"])):
            plot_3panel(t_hw, t_sim, hw_ag["rpy_rates"], sim_ag["rpy_rates"],
                        ["Roll Rate", "Pitch Rate", "Yaw Rate"], "rad/s",
                        f"{display} - Body Rates Comparison",
                        outdir / f"{name}_body_rates.png",
                        hw_label=args.hw_label, sim_label=args.sim_label,
                        agent_name=name,
                        t_extra=t_extra, extra_data=ex_ag["rpy_rates"] if ex_ag else None,
                        extra_label=args.extra_label if extra else None)

    # Combined comparison
    plot_combined(t_hw, t_sim, hw_agents, sim_agents, agent_names,
                  outdir / "combined_comparison.png",
                  hw_label=args.hw_label, sim_label=args.sim_label,
                  t_extra=t_extra, extra_agents=extra_agents,
                  extra_label=args.extra_label if extra else None)

    # 3D trajectory plot
    plot_3d_trajectories(t_hw, t_sim, hw_agents, sim_agents, agent_names,
                         outdir / "trajectories_3d.png",
                         hw_label=args.hw_label, sim_label=args.sim_label,
                         extra_agents=extra_agents,
                         extra_label=args.extra_label if extra else None)

    # Animated 3D trajectory — replay same run 10 times for full 360 rotation
    hw_runs_data = [(t_hw, hw_agents)] * 10

    animate_3d_trajectories(t_hw, t_sim, hw_agents, sim_agents,
                            agent_names, outdir / "trajectories_3d.mp4",
                            hw_label=args.hw_label, sim_label=args.sim_label,
                            hw_runs=hw_runs_data,
                            t_extra=t_extra, extra_agents=extra_agents,
                            extra_label=args.extra_label if extra else None)

    # RMS errors
    compute_rms_errors(t_hw, t_sim, hw_agents, sim_agents, agent_names)

    print(f"\nPlots saved to: {outdir}/")


if __name__ == "__main__":
    main()
