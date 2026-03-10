#!/usr/bin/env python3
"""Replay CrazySim hover data through crazyflow's first-principles physics.

Takes motor angular rates (omega) from the CrazySim pipeline NPZ and computes
the forces/accelerations that crazyflow would produce, comparing with CrazySim.
Also runs open-loop integration to show trajectory divergence.

Both simulators use the same mass (0.0406 kg) and the same motor angular rates.

Usage:
    python3 scripts/replay_crazysim_in_crazyflow.py hardware/logs/crazysim_pipeline.npz
"""

import argparse
from pathlib import Path

import numpy as np


# ── CrazySim parameters ──
CRAZYSIM_THRUST_MAX = 0.18
CRAZYSIM_MOTOR_CONSTANT = 2.3375e-8
CRAZYSIM_MAX_ROT_VEL = 2797.0
CRAZYSIM_PWM_MIN = 7000
CRAZYSIM_PWM_MAX = 65535

# ── Shared test mass ──
MASS = 0.0406  # kg (SDF body + motors)

# ── cf2x_T350 physical parameters (from drone-models params.toml) ──
L = 0.03253
PROP_INERTIA = 38.93e-9
GRAVITY_VEC = np.array([0.0, 0.0, -9.81])
J = np.array([
    [16.8e-6, 0.0, 0.0],
    [0.0, 16.8e-6, 0.0],
    [0.0, 0.0, 29.8e-6],
])
J_INV = np.linalg.inv(J)
RPM2THRUST = np.array([0.0, -7.167227176573658e-7, 2.9401303690194613e-10])
RPM2TORQUE = np.array([0.0, 5.815894847811497e-10, 1.331813874166509e-12])
MIXING_MATRIX = np.array([
    [-1.0, -1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0, 1.0],
])
DRAG_MATRIX = np.array([
    [-0.01556697, 0.0, 0.0],
    [0.0, -0.01556697, 0.0],
    [0.0, 0.0, -0.02191672],
])
ROTOR_DYN_COEF = np.array([11.374753209400291, 0.0, 0.0, 0.00037867688499079635])


def crazysim_pwm2omega(pwm):
    """CrazySim PWM2OMEGA (from CrtpUtils.h). Returns rad/s."""
    pwm = np.asarray(pwm, dtype=float)
    thrust_desired = (pwm / CRAZYSIM_PWM_MAX) * CRAZYSIM_THRUST_MAX
    omega = np.sqrt(np.maximum(thrust_desired, 0) / CRAZYSIM_MOTOR_CONSTANT)
    omega = np.minimum(omega, CRAZYSIM_MAX_ROT_VEL)
    omega = np.where(pwm < CRAZYSIM_PWM_MIN, 0.0, omega)
    return omega


def quat_to_matrix(quat):
    """Convert xyzw quaternion to 3x3 rotation matrix (body to world)."""
    quat = quat / np.linalg.norm(quat)
    x, y, z, w = quat
    x2, y2, z2, w2 = x*x, y*y, z*z, w*w
    xy, xz, xw = x*y, x*z, x*w
    yz, yw, zw = y*z, y*w, z*w
    return np.array([
        [x2-y2-z2+w2, 2*(xy-zw), 2*(xz+yw)],
        [2*(xy+zw), -x2+y2-z2+w2, 2*(yz-xw)],
        [2*(xz-yw), 2*(yz+xw), -x2-y2+z2+w2],
    ])


def ang_vel2quat_dot(quat, ang_vel):
    """Compute quaternion derivative from angular velocity (xyzw format)."""
    p, q, r = ang_vel
    xi = np.array([
        [0, -p, -q, -r],
        [p, 0, r, -q],
        [q, -r, 0, p],
        [r, q, -p, 0],
    ])
    return 0.5 * xi @ quat


def compute_forces(quat, vel, rotor_vel_rpm, mass, drag_matrix):
    """Compute all forces at a single timestep. Returns dict of force components."""
    rot_mat = quat_to_matrix(quat)

    # Motor forces
    forces_motor = RPM2THRUST[0] + RPM2THRUST[1] * rotor_vel_rpm + RPM2THRUST[2] * rotor_vel_rpm**2
    total_thrust = np.sum(forces_motor)
    force_body = np.array([0.0, 0.0, total_thrust])
    force_thrust_world = rot_mat @ force_body

    # Gravity
    force_gravity = GRAVITY_VEC * mass

    # Drag
    vel_body = rot_mat.T @ vel
    force_drag_body = drag_matrix @ vel_body
    force_drag_world = rot_mat @ force_drag_body

    # Total
    force_total = force_thrust_world + force_gravity + force_drag_world
    acc = force_total / mass

    return {
        "thrust_per_motor": forces_motor,
        "total_thrust": total_thrust,
        "force_thrust_z": force_thrust_world[2],
        "force_gravity_z": force_gravity[2],
        "force_drag_z": force_drag_world[2],
        "force_total_z": force_total[2],
        "acc_z": acc[2],
    }


def first_principles_step(pos, quat, vel, ang_vel, cmd_rpm, rotor_vel_rpm, mass, drag_matrix):
    """Single dynamics step with rotor dynamics.

    Args:
        cmd_rpm: Commanded rotor speeds (RPM) — from CrazySim omega.
        rotor_vel_rpm: Current rotor velocity state (RPM) — evolves via motor dynamics.

    Returns:
        pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot
    """
    rot_mat = quat_to_matrix(quat)

    # Rotor dynamics: rotor_vel tracks cmd with first-order dynamics
    rotor_vel_dot = np.where(
        cmd_rpm > rotor_vel_rpm,
        ROTOR_DYN_COEF[0] * (cmd_rpm - rotor_vel_rpm)
        + ROTOR_DYN_COEF[1] * (cmd_rpm**2 - rotor_vel_rpm**2),
        ROTOR_DYN_COEF[2] * (cmd_rpm - rotor_vel_rpm)
        + ROTOR_DYN_COEF[3] * (cmd_rpm**2 - rotor_vel_rpm**2),
    )

    # Forces from current rotor velocity (not commanded)
    forces_motor = RPM2THRUST[0] + RPM2THRUST[1] * rotor_vel_rpm + RPM2THRUST[2] * rotor_vel_rpm**2
    total_force = np.sum(forces_motor)
    force_body = np.array([0.0, 0.0, total_force])
    force_world = rot_mat @ force_body
    force_gravity = GRAVITY_VEC * mass
    vel_body = rot_mat.T @ vel
    force_drag_body = drag_matrix @ vel_body
    force_drag_world = rot_mat @ force_drag_body
    forces_sum = force_world + force_gravity + force_drag_world

    torques_motor = RPM2TORQUE[0] + RPM2TORQUE[1] * rotor_vel_rpm + RPM2TORQUE[2] * rotor_vel_rpm**2
    torque_thrust = (MIXING_MATRIX @ forces_motor) * np.array([L, L, 0.0])
    torque_drag = (MIXING_MATRIX @ torques_motor) * np.array([0.0, 0.0, 1.0])

    rpm_to_rad = 2 * np.pi / 60
    rotor_vel_rads = rotor_vel_rpm * rpm_to_rad
    rotor_vel_dot_rads = rotor_vel_dot * rpm_to_rad
    torque_inertia = PROP_INERTIA * np.array([
        -ang_vel[1] * np.sum(MIXING_MATRIX[-1, :] * rotor_vel_rads),
        -ang_vel[0] * np.sum(MIXING_MATRIX[-1, :] * rotor_vel_rads),
        np.sum(MIXING_MATRIX[-1, :] * rotor_vel_dot_rads),
    ])
    torque_vec = torque_thrust + torque_drag + torque_inertia

    pos_dot = vel
    vel_dot = forces_sum / mass
    quat_dot = ang_vel2quat_dot(quat, ang_vel)
    torque_net = torque_vec - np.cross(ang_vel, J @ ang_vel)
    ang_vel_dot = J_INV @ torque_net

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot


def load_and_prepare(data):
    """Load NPZ data and prepare aligned arrays."""
    t_motor = data["motor_m1_t"]
    pwm = np.stack([data[f"motor_m{i}"] for i in range(1, 5)], axis=-1).astype(float)
    omega_rads = np.stack([crazysim_pwm2omega(pwm[:, i]) for i in range(4)], axis=-1)
    omega_rpm = omega_rads * 60 / (2 * np.pi)

    t_state = data["stateEstimate_x_t"]
    cs_pos = np.stack([data[f"stateEstimate_{c}"] for c in "xyz"], axis=-1)
    cs_vel = np.stack([data[f"stateEstimate_v{c}"] for c in "xyz"], axis=-1)
    cs_quat = np.stack([data[f"stateEstimate_{c}"] for c in ["qx", "qy", "qz", "qw"]], axis=-1)

    # Interpolate motor data to state timestamps
    n = len(t_state)
    omega_rpm_interp = np.stack([
        np.interp(t_state, t_motor, omega_rpm[:, i]) for i in range(4)
    ], axis=-1)
    pwm_interp = np.stack([
        np.interp(t_state, t_motor, pwm[:, i]) for i in range(4)
    ], axis=-1)

    # Find steady hover start (z > 0.4m and |vz| < 0.1 m/s)
    hover_mask = (cs_pos[:, 2] > 0.4) & (np.abs(cs_vel[:, 2]) < 0.1)
    start_idx = np.argmax(hover_mask) if np.any(hover_mask) else 0

    return {
        "t": t_state, "n": n, "start_idx": start_idx,
        "cs_pos": cs_pos, "cs_vel": cs_vel, "cs_quat": cs_quat,
        "omega_rpm": omega_rpm_interp, "pwm": pwm_interp,
    }


def compute_force_timeseries(prep, mass, drag_matrix):
    """Compute force components at every timestep using CrazySim state + omega."""
    n = prep["n"]
    results = {k: np.zeros(n) for k in [
        "total_thrust", "force_thrust_z", "force_gravity_z",
        "force_drag_z", "force_total_z", "acc_z",
    ]}
    results["thrust_per_motor"] = np.zeros((n, 4))

    for i in range(n):
        quat = prep["cs_quat"][i] if i < len(prep["cs_quat"]) else prep["cs_quat"][-1]
        quat = quat / np.linalg.norm(quat)
        vel = prep["cs_vel"][i]
        forces = compute_forces(quat, vel, prep["omega_rpm"][i], mass, drag_matrix)
        for k in results:
            results[k][i] = forces[k]

    return results


def replay_openloop(prep, mass, drag_matrix):
    """Open-loop integration from flight start with rotor dynamics.

    The rotor velocities are initialized to the CrazySim omega at flight start
    (the hover RPM), and then the CrazySim omega at each timestep is used as the
    commanded RPM. The rotor dynamics model evolves the actual rotor velocities.
    """
    n = prep["n"]
    start = prep["start_idx"]

    pos = prep["cs_pos"][start].copy()
    quat = prep["cs_quat"][start].copy()
    quat = quat / np.linalg.norm(quat)
    vel = prep["cs_vel"][start].copy()
    ang_vel = np.zeros(3)
    # Initialize rotor velocities to CrazySim's omega at flight start
    rotor_vel = prep["omega_rpm"][start].copy()

    cf_pos = np.full((n, 3), np.nan)
    cf_vel = np.full((n, 3), np.nan)
    cf_pos[start] = pos
    cf_vel[start] = vel

    for i in range(start, n - 1):
        dt = prep["t"][i + 1] - prep["t"][i]
        if dt <= 0 or dt > 0.1:
            dt = 0.01

        # CrazySim omega as commanded RPM, rotor_vel as current state
        cmd_rpm = prep["omega_rpm"][i]
        pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = first_principles_step(
            pos, quat, vel, ang_vel, cmd_rpm, rotor_vel, mass, drag_matrix
        )

        pos = pos + pos_dot * dt
        quat = quat + quat_dot * dt
        quat = quat / np.linalg.norm(quat)
        vel = vel + vel_dot * dt
        ang_vel = ang_vel + ang_vel_dot * dt
        rotor_vel = rotor_vel + rotor_vel_dot * dt

        cf_pos[i + 1] = pos
        cf_vel[i + 1] = vel

    return cf_pos, cf_vel


def plot_all(prep, forces_drag, forces_nodrag, pos_drag, pos_nodrag, out_dir):
    """Generate all comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except ImportError:
        pass
    plt.rcParams["pdf.fonttype"] = 42

    t = prep["t"]
    start = prep["start_idx"]
    cs_pos = prep["cs_pos"]
    cs_vel = prep["cs_vel"]
    weight = MASS * 9.81

    # ── Plot 1: Force breakdown ──
    fig, axes = plt.subplots(4, 1, figsize=(5, 7), sharex=True)

    # Motor PWMs
    ax = axes[0]
    for i in range(4):
        ax.plot(t, prep["pwm"][:, i], linewidth=0.5, label=f'm{i+1}')
    ax.set_ylabel('PWM')
    ax.legend(fontsize=5, ncol=4)
    ax.grid(True, alpha=0.3)

    # Per-motor thrust
    ax = axes[1]
    for i in range(4):
        ax.plot(t, forces_nodrag["thrust_per_motor"][:, i] * 1000, linewidth=0.5, label=f'm{i+1}')
    ax.axhline(weight / 4 * 1000, color='k', linewidth=0.5, linestyle='--', label='mg/4')
    ax.set_ylabel('Thrust/motor (mN)')
    ax.legend(fontsize=5, ncol=5)
    ax.grid(True, alpha=0.3)

    # Total Z forces
    ax = axes[2]
    ax.plot(t, forces_nodrag["force_thrust_z"], linewidth=0.7, label='Thrust Z')
    ax.plot(t, forces_drag["force_drag_z"], linewidth=0.7, label='Drag Z (crazyflow)')
    ax.axhline(-weight, color='k', linewidth=0.5, linestyle='--', label='-mg')
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.set_ylabel('Force Z (N)')
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

    # Net Z acceleration
    ax = axes[3]
    ax.plot(t, forces_nodrag["acc_z"], linewidth=0.7, label='No drag (CrazySim-like)')
    ax.plot(t, forces_drag["acc_z"], linewidth=0.7, label='With drag (crazyflow)')
    ax.axhline(0, color='k', linewidth=0.3)
    ax.set_ylabel('Acc Z (m/s²)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Force Analysis: CrazySim Motor Rates → Crazyflow Physics (mass={MASS} kg)', fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "replay_force_breakdown.pdf", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'replay_force_breakdown.pdf'}")

    # ── Plot 2: Position comparison (open-loop) ──
    fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
    labels = ["X", "Y", "Z"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, cs_pos[:, i], 'k-', linewidth=0.8, label='CrazySim')
        ax.plot(t, pos_drag[:, i], 'C0--', linewidth=0.8, label='Crazyflow (with drag)')
        ax.plot(t, pos_nodrag[:, i], 'C1:', linewidth=0.8, label='Crazyflow (no drag)')
        ax.axvline(t[start], color='gray', linewidth=0.3, linestyle='--')
        ax.set_ylabel(f'{label} (m)')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=5, loc='best')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Open-Loop Replay: CrazySim vs Crazyflow (mass={MASS} kg)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "replay_position.pdf", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'replay_position.pdf'}")

    # ── Plot 3: Z-axis zoom (first 2s of flight) ──
    # Find a window where trajectories haven't diverged too far
    end_2s = np.searchsorted(t, t[start] + 2.0)
    end_5s = np.searchsorted(t, t[start] + 5.0)

    fig, axes = plt.subplots(2, 1, figsize=(5, 4))

    ax = axes[0]
    ax.plot(t[start:end_2s], cs_pos[start:end_2s, 2], 'k-', linewidth=0.8, label='CrazySim')
    ax.plot(t[start:end_2s], pos_drag[start:end_2s, 2], 'C0--', linewidth=0.8, label='Crazyflow (with drag)')
    ax.plot(t[start:end_2s], pos_nodrag[start:end_2s, 2], 'C1:', linewidth=0.8, label='Crazyflow (no drag)')
    ax.set_ylabel('Z (m)')
    ax.set_title('First 2s of flight', fontsize=8)
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t[start:end_5s], cs_pos[start:end_5s, 2], 'k-', linewidth=0.8, label='CrazySim')
    ax.plot(t[start:end_5s], pos_drag[start:end_5s, 2], 'C0--', linewidth=0.8, label='Crazyflow (with drag)')
    ax.plot(t[start:end_5s], pos_nodrag[start:end_5s, 2], 'C1:', linewidth=0.8, label='Crazyflow (no drag)')
    ax.set_ylabel('Z (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('First 5s of flight', fontsize=8)
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Z-Axis Early Divergence (mass={MASS} kg)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "replay_z_zoom.pdf", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'replay_z_zoom.pdf'}")

    # ── Plot 4: Velocity comparison ──
    fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
    vel_labels = ["Vx", "Vy", "Vz"]
    cf_vel_drag = np.gradient(pos_drag, t, axis=0)
    cf_vel_nodrag = np.gradient(pos_nodrag, t, axis=0)

    for i, (ax, label) in enumerate(zip(axes, vel_labels)):
        sl = slice(start, end_5s)
        ax.plot(t[sl], cs_vel[sl, i], 'k-', linewidth=0.8, label='CrazySim')
        ax.plot(t[sl], cf_vel_drag[sl, i], 'C0--', linewidth=0.8, label='Crazyflow (with drag)')
        ax.plot(t[sl], cf_vel_nodrag[sl, i], 'C1:', linewidth=0.8, label='Crazyflow (no drag)')
        ax.set_ylabel(f'{label} (m/s)')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=5, loc='best')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Velocity Comparison — First 5s (mass={MASS} kg)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "replay_velocity.pdf", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'replay_velocity.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="Replay CrazySim data in crazyflow physics")
    parser.add_argument("npz_path", help="Path to CrazySim pipeline NPZ")
    parser.add_argument("--out-dir", default="results/plots", help="Output directory for plots")
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    out_dir = Path(args.out_dir)

    print(f"Mass: {MASS} kg, Weight: {MASS*9.81:.4f} N")
    print(f"Drag matrix diagonal: {np.diag(DRAG_MATRIX)}")

    prep = load_and_prepare(data)
    print(f"Samples: {prep['n']}, Flight starts at idx {prep['start_idx']} "
          f"(t={prep['t'][prep['start_idx']]:.2f}s)")

    # Compute force timeseries using CrazySim state + crazyflow physics
    print("\nComputing force timeseries...")
    forces_drag = compute_force_timeseries(prep, MASS, DRAG_MATRIX)
    forces_nodrag = compute_force_timeseries(prep, MASS, np.zeros((3, 3)))

    # Print force stats during hover
    start = prep["start_idx"]
    hover_mask = prep["cs_pos"][:, 2] > 0.3  # Rough hover region
    if np.any(hover_mask):
        hover_sl = hover_mask
        weight = MASS * 9.81
        mean_thrust = np.mean(forces_nodrag["total_thrust"][hover_sl])
        mean_drag_z = np.mean(forces_drag["force_drag_z"][hover_sl])
        print(f"\n--- Hover force stats (z > 0.3m) ---")
        print(f"  Weight:           {weight:.4f} N")
        print(f"  Mean total thrust: {mean_thrust:.4f} N ({mean_thrust/weight*100:.1f}% of weight)")
        print(f"  Mean drag Z:       {mean_drag_z:.6f} N")
        print(f"  Thrust deficit:    {weight - mean_thrust:.4f} N")

    # Open-loop replay
    print("\nRunning open-loop replay WITH drag...")
    pos_drag, vel_drag = replay_openloop(prep, MASS, DRAG_MATRIX)

    print("Running open-loop replay WITHOUT drag...")
    pos_nodrag, vel_nodrag = replay_openloop(prep, MASS, np.zeros((3, 3)))

    # Stats
    end_idx = len(prep["t"]) - 1
    cs_z_end = prep["cs_pos"][end_idx, 2]
    print(f"\n--- Z at end (t={prep['t'][end_idx]:.1f}s) ---")
    print(f"  CrazySim:              {cs_z_end:.4f} m")
    print(f"  Crazyflow (with drag): {pos_drag[end_idx, 2]:.4f} m "
          f"(diff: {pos_drag[end_idx, 2]-cs_z_end:+.4f} m)")
    print(f"  Crazyflow (no drag):   {pos_nodrag[end_idx, 2]:.4f} m "
          f"(diff: {pos_nodrag[end_idx, 2]-cs_z_end:+.4f} m)")

    # Plot
    plot_all(prep, forces_drag, forces_nodrag, pos_drag, pos_nodrag, out_dir)


if __name__ == "__main__":
    main()
