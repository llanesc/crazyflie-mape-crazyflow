#!/usr/bin/env python3
"""Replay rosbag commands through crazyflow sim and compare with real flight.

Takes HW data extracted from a rosbag (.npz), replays through crazyflow's
attitude control pipeline with first_principles physics (via crazyflow Sim),
and through a pure-numpy so_rpy forward simulation, then compares predicted
trajectories to actual hardware measurements.

Usage:
    # Step 1: Extract from rosbag (env_hardware)
    env_hardware/bin/python hardware/logs/extract_rosbag_data.py <bag_path>

    # Step 2: Replay in sim (env_crazyflow)
    env_crazyflow/bin/python hardware/logs/replay_rosbag_in_sim.py <bag_path>_hw_data.npz
"""

import os
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


# ── so_rpy parameters for cf2x_T350 ──
SO_RPY_PARAMS = {
    'mass': 0.0406,
    'gravity_vec': np.array([0.0, 0.0, -9.81]),
    'acc_coef': 0.0,
    'cmd_f_coef': 0.86214912,
    'rpy_coef': np.array([-267.14, -267.14, -126.49]),
    'rpy_rates_coef': np.array([-19.48, -19.48, -14.41]),
    'cmd_rpy_coef': np.array([200.90, 200.90, 193.11]),
}


def load_hw_data(npz_path: str) -> dict:
    """Load HW data from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    drone_names = list(data['drone_names'])
    hw_data = {}
    for name in drone_names:
        hw_data[name] = {}
        for key in ['odom_t', 'pos', 'vel', 'quat', 'ang_vel',
                     'cmd_t', 'cmd_roll', 'cmd_pitch', 'cmd_yaw',
                     'cmd_thrust_pwm', 'cmd_thrust_N']:
            hw_data[name][key] = data[f"{name}/{key}"]
    return hw_data


# ── Pure numpy rotation utilities ──

def quat_dot_from_ang_vel(quat, ang_vel):
    """Compute quaternion derivative from angular velocity. quat is [x,y,z,w]."""
    wx, wy, wz = ang_vel
    qx, qy, qz, qw = quat
    return 0.5 * np.array([
        qw*wx - qz*wy + qy*wz,
        qz*wx + qw*wy - qx*wz,
        -qy*wx + qx*wy + qw*wz,
        -qx*wx - qy*wy - qz*wz,
    ])


def ang_vel_to_rpy_rates(quat, ang_vel):
    """Convert body angular velocity to Euler rate (xyz convention)."""
    rpy = R.from_quat(quat).as_euler("xyz")
    phi, theta = rpy[0], rpy[1]
    sp, cp = np.sin(phi), np.cos(phi)
    ct = np.cos(theta)
    tt = np.tan(theta)
    W = np.array([
        [1, sp*tt,  cp*tt],
        [0, cp,    -sp],
        [0, sp/ct,  cp/ct],
    ])
    return W @ ang_vel


def rpy_rates_to_ang_vel(quat, rpy_rates):
    """Convert Euler rates to body angular velocity (xyz convention)."""
    rpy = R.from_quat(quat).as_euler("xyz")
    phi, theta = rpy[0], rpy[1]
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    W_inv = np.array([
        [1,  0,   -st],
        [0,  cp,   sp*ct],
        [0, -sp,   cp*ct],
    ])
    return W_inv @ rpy_rates


def rpy_rates_deriv_to_ang_vel_deriv(quat, rpy_rates, rpy_rates_deriv):
    """Convert rpy_rates_deriv to ang_vel_deriv: w_dot = W_dot @ drpy + W_inv @ ddrpy."""
    rpy = R.from_quat(quat).as_euler("xyz")
    phi, theta = rpy[0], rpy[1]
    phi_dot, theta_dot = rpy_rates[0], rpy_rates[1]
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)

    # W_inv (rpy_rates -> ang_vel)
    W_inv = np.array([
        [1,  0,   -st],
        [0,  cp,   sp*ct],
        [0, -sp,   cp*ct],
    ])

    # W_inv_dot
    W_inv_dot = np.array([
        [0, 0,                          -ct*theta_dot],
        [0, -sp*phi_dot,                cp*phi_dot*ct - sp*st*theta_dot],
        [0, -cp*phi_dot,               -sp*phi_dot*ct - cp*st*theta_dot],
    ])

    return W_inv_dot @ rpy_rates + W_inv @ rpy_rates_deriv


def so_rpy_step(pos, vel, quat, ang_vel, cmd_rpy, cmd_f, dt, params):
    """One Euler integration step of the so_rpy dynamics.

    Args:
        pos: (3,) position
        vel: (3,) velocity
        quat: (4,) quaternion [x,y,z,w]
        ang_vel: (3,) body angular velocity
        cmd_rpy: (3,) commanded roll, pitch, yaw_rate (rad)
        cmd_f: float, commanded thrust (N)
        dt: time step
        params: dict of so_rpy parameters

    Returns:
        new_pos, new_vel, new_quat, new_ang_vel
    """
    mass = params['mass']
    gravity = params['gravity_vec']
    acc_coef = params['acc_coef']
    cmd_f_coef = params['cmd_f_coef']
    rpy_coef = params['rpy_coef']
    rpy_rates_coef = params['rpy_rates_coef']
    cmd_rpy_coef = params['cmd_rpy_coef']

    # Translation dynamics
    rot = R.from_quat(quat)
    euler = rot.as_euler("xyz")
    z_axis = rot.as_matrix()[:, 2]  # body z in world frame

    thrust = acc_coef + cmd_f_coef * cmd_f
    vel_dot = (thrust / mass) * z_axis + gravity

    # Rotation dynamics
    rpy_rates = ang_vel_to_rpy_rates(quat, ang_vel)
    rpy_rates_dot = rpy_coef * euler + rpy_rates_coef * rpy_rates + cmd_rpy_coef * cmd_rpy

    quat_dot = quat_dot_from_ang_vel(quat, ang_vel)
    ang_vel_dot = rpy_rates_deriv_to_ang_vel_deriv(quat, rpy_rates, rpy_rates_dot)

    # Euler integration
    new_pos = pos + vel * dt
    new_vel = vel + vel_dot * dt
    new_quat = quat + quat_dot * dt
    new_quat = new_quat / np.linalg.norm(new_quat)  # re-normalize
    new_ang_vel = ang_vel + ang_vel_dot * dt

    return new_pos, new_vel, new_quat, new_ang_vel


def simulate_so_rpy_numpy(hw_data: dict, sim_freq: int = 500, ctrl_freq: int = 50) -> dict:
    """Forward simulate so_rpy dynamics in pure numpy."""
    duration = hw_data['odom_t'][-1]
    dt = 1.0 / sim_freq
    n_sim_steps = int(duration * sim_freq)
    sim_steps_per_ctrl = sim_freq // ctrl_freq

    # Interpolate commands
    cmd_t = hw_data['cmd_t']
    interp_roll = interp1d(cmd_t, hw_data['cmd_roll'], fill_value="extrapolate")
    interp_pitch = interp1d(cmd_t, hw_data['cmd_pitch'], fill_value="extrapolate")
    interp_yaw = interp1d(cmd_t, hw_data['cmd_yaw'], fill_value="extrapolate")
    interp_thrust = interp1d(cmd_t, hw_data['cmd_thrust_N'], fill_value="extrapolate")

    # Initial state
    pos = hw_data['pos'][0].copy()
    vel = hw_data['vel'][0].copy()
    quat = hw_data['quat'][0].copy()
    ang_vel = hw_data['ang_vel'][0].copy()

    sim_times = []
    sim_positions = []
    sim_velocities = []
    sim_quats = []

    t = 0.0
    for step in range(n_sim_steps):
        # Record at ctrl frequency
        if step % sim_steps_per_ctrl == 0:
            sim_times.append(t)
            sim_positions.append(pos.copy())
            sim_velocities.append(vel.copy())
            sim_quats.append(quat.copy())

        # Get command
        roll = float(interp_roll(t))
        pitch = float(interp_pitch(t))
        yaw = float(interp_yaw(t))
        thrust = float(interp_thrust(t))
        cmd_rpy = np.array([roll, pitch, yaw])

        pos, vel, quat, ang_vel = so_rpy_step(
            pos, vel, quat, ang_vel, cmd_rpy, thrust, dt, SO_RPY_PARAMS
        )
        t += dt

    return {
        'sim_t': np.array(sim_times),
        'sim_pos': np.array(sim_positions),
        'sim_vel': np.array(sim_velocities),
        'sim_quat': np.array(sim_quats),
    }


def simulate_first_principles(hw_data: dict, drone_model: str = "cf2x_T350",
                                sim_freq: int = 500, ctrl_freq: int = 50,
                                mass: float = 0.0406) -> dict:
    """Simulate using crazyflow's first_principles physics."""
    import jax.numpy as jnp
    from crazyflow.sim.sim import Sim
    from crazyflow.sim.physics import Physics

    sim = Sim(
        n_worlds=1, n_drones=1, freq=sim_freq,
        drone_model=drone_model, physics=Physics.first_principles,
        control="attitude",
    )

    # Override mass
    sim.data = sim.data.replace(
        params=sim.data.params.replace(
            mass=jnp.full_like(sim.data.params.mass, mass)
        )
    )

    # Compute hover rotor velocity for the actual mass
    from drone_models.transform import motor_force2rotor_vel
    hover_force_per_motor = mass * 9.81 / 4.0
    rpm2thrust = sim.data.params.rpm2thrust  # (3,)
    hover_force_arr = jnp.full((1, 1, 4), hover_force_per_motor)
    hover_rotor_vel = motor_force2rotor_vel(hover_force_arr, rpm2thrust)

    init_pos = hw_data['pos'][0]
    init_vel = hw_data['vel'][0]
    init_quat = hw_data['quat'][0]
    init_ang_vel = hw_data['ang_vel'][0]

    sim.data = sim.data.replace(
        states=sim.data.states.replace(
            pos=jnp.array(init_pos.reshape(1, 1, 3)),
            vel=jnp.array(init_vel.reshape(1, 1, 3)),
            quat=jnp.array(init_quat.reshape(1, 1, 4)),
            ang_vel=jnp.array(init_ang_vel.reshape(1, 1, 3)),
            rotor_vel=hover_rotor_vel,
        ),
        controls=sim.data.controls.replace(
            rotor_vel=hover_rotor_vel,
        )
    )

    duration = hw_data['odom_t'][-1]
    n_steps = int(duration * ctrl_freq)
    ctrl_dt = 1.0 / ctrl_freq
    sim_steps_per_ctrl = sim_freq // ctrl_freq

    cmd_t = hw_data['cmd_t']
    interp_roll = interp1d(cmd_t, hw_data['cmd_roll'], fill_value="extrapolate")
    interp_pitch = interp1d(cmd_t, hw_data['cmd_pitch'], fill_value="extrapolate")
    interp_yaw = interp1d(cmd_t, hw_data['cmd_yaw'], fill_value="extrapolate")
    interp_thrust = interp1d(cmd_t, hw_data['cmd_thrust_N'], fill_value="extrapolate")

    sim_times = []
    sim_positions = []
    sim_velocities = []
    sim_quats = []

    t = 0.0
    for step in range(n_steps):
        sim_times.append(t)
        sim_positions.append(np.asarray(sim.data.states.pos[0, 0]))
        sim_velocities.append(np.asarray(sim.data.states.vel[0, 0]))
        sim_quats.append(np.asarray(sim.data.states.quat[0, 0]))

        roll = float(interp_roll(t))
        pitch = float(interp_pitch(t))
        yaw = float(interp_yaw(t))
        thrust = float(interp_thrust(t))

        cmd = jnp.array([[[roll, pitch, yaw, thrust]]])
        sim.attitude_control(cmd)
        sim.step(n_steps=sim_steps_per_ctrl)
        t += ctrl_dt

    return {
        'sim_t': np.array(sim_times),
        'sim_pos': np.array(sim_positions),
        'sim_vel': np.array(sim_velocities),
        'sim_quat': np.array(sim_quats),
    }


def compute_errors(hw_data: dict, sim_result: dict) -> dict:
    """Compute trajectory errors between HW and sim."""
    hw_t = hw_data['odom_t']
    sim_t = sim_result['sim_t']

    t_end = min(hw_t[-1], sim_t[-1])
    hw_mask = hw_t <= t_end
    hw_t_trim = hw_t[hw_mask]
    hw_pos = hw_data['pos'][hw_mask]
    hw_vel = hw_data['vel'][hw_mask]

    sim_pos_interp = np.array([
        interp1d(sim_t, sim_result['sim_pos'][:, i], fill_value="extrapolate")(hw_t_trim)
        for i in range(3)
    ]).T
    sim_vel_interp = np.array([
        interp1d(sim_t, sim_result['sim_vel'][:, i], fill_value="extrapolate")(hw_t_trim)
        for i in range(3)
    ]).T

    pos_err = hw_pos - sim_pos_interp
    vel_err = hw_vel - sim_vel_interp

    return {
        't': hw_t_trim,
        'hw_pos': hw_pos,
        'sim_pos': sim_pos_interp,
        'hw_vel': hw_vel,
        'sim_vel': sim_vel_interp,
        'pos_rmse': np.sqrt(np.mean(pos_err**2, axis=0)),
        'pos_rmse_3d': np.sqrt(np.mean(np.sum(pos_err**2, axis=1))),
        'vel_rmse': np.sqrt(np.mean(vel_err**2, axis=0)),
        'vel_rmse_3d': np.sqrt(np.mean(np.sum(vel_err**2, axis=1))),
        'pos_err_z_mean': np.mean(pos_err[:, 2]),
    }


def plot_comparison(hw_data: dict, fp_result: dict, sorpy_result: dict,
                    fp_errors: dict, sorpy_errors: dict, name: str, out_dir: Path):
    """Plot X, Y, Z position comparison for one drone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except ImportError:
        pass
    plt.rcParams["pdf.fonttype"] = 42

    hw_t = hw_data['odom_t']
    hw_pos = hw_data['pos']
    labels = ['X (m)', 'Y (m)', 'Z (m)']

    fig, axes = plt.subplots(3, 1, figsize=(5, 6), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(hw_t, hw_pos[:, i], 'k-', linewidth=1.2, label='Hardware')
        ax.plot(fp_result['sim_t'], fp_result['sim_pos'][:, i], '--',
                linewidth=1.0, label=f'first_principles (RMSE={fp_errors["pos_rmse"][i]:.3f})')
        ax.plot(sorpy_result['sim_t'], sorpy_result['sim_pos'][:, i], ':',
                linewidth=1.0, label=f'so_rpy (RMSE={sorpy_errors["pos_rmse"][i]:.3f})')
        ax.set_ylabel(label)
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'{name} — Position: HW vs Sim (mass={SO_RPY_PARAMS["mass"]:.4f} kg)', fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"replay_{name}.pdf"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


def plot_commands(hw_data: dict, name: str, out_dir: Path):
    """Plot commanded roll, pitch, yaw_rate, and thrust for one drone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except ImportError:
        pass
    plt.rcParams["pdf.fonttype"] = 42

    cmd_t = hw_data['cmd_t']

    fig, axes = plt.subplots(4, 1, figsize=(5, 7), sharex=True)

    axes[0].plot(cmd_t, np.degrees(hw_data['cmd_roll']), linewidth=0.8)
    axes[0].set_ylabel('Roll (deg)')

    axes[1].plot(cmd_t, np.degrees(hw_data['cmd_pitch']), linewidth=0.8)
    axes[1].set_ylabel('Pitch (deg)')

    axes[2].plot(cmd_t, np.degrees(hw_data['cmd_yaw']), linewidth=0.8)
    axes[2].set_ylabel('Yaw rate (deg/s)')

    axes[3].plot(cmd_t, hw_data['cmd_thrust_N'], linewidth=0.8, label='N')
    hover_thrust = SO_RPY_PARAMS['mass'] * 9.81
    axes[3].axhline(hover_thrust, color='r', linestyle='--', linewidth=0.6, label=f'hover={hover_thrust:.3f} N')
    axes[3].set_ylabel('Thrust (N)')
    axes[3].legend(fontsize=6)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'{name} — Commands', fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"commands_{name}.pdf"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved commands plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Replay HW commands in sim')
    parser.add_argument('npz_path', type=str, help='Path to .npz file from extract_rosbag_data.py')
    parser.add_argument('--drone-model', default='cf2x_T350', help='Drone model')
    parser.add_argument('--sim-freq', type=int, default=500)
    parser.add_argument('--ctrl-freq', type=int, default=50)
    parser.add_argument('--out-dir', default='results/plots', help='Output directory for plots')
    args = parser.parse_args()

    print(f"Loading: {args.npz_path}")
    hw_data = load_hw_data(args.npz_path)
    print(f"Loaded {len(hw_data)} drones: {list(hw_data.keys())}")
    out_dir = Path(args.out_dir)

    for name, data in hw_data.items():
        print(f"\n{'='*70}")
        print(f"  {name}: {len(data['odom_t'])} odom samples, "
              f"{len(data['cmd_t'])} cmd samples, "
              f"{data['odom_t'][-1]:.2f}s")
        print(f"{'='*70}")

        fp_result = fp_errors = None
        sorpy_result = sorpy_errors = None

        # first_principles
        print(f"\n  --- first_principles (Mellinger + full physics, mass={SO_RPY_PARAMS['mass']}) ---")
        try:
            fp_result = simulate_first_principles(
                data, drone_model=args.drone_model,
                sim_freq=args.sim_freq, ctrl_freq=args.ctrl_freq,
                mass=SO_RPY_PARAMS['mass'],
            )
            fp_errors = compute_errors(data, fp_result)
            print(f"  Position RMSE [x,y,z]: [{fp_errors['pos_rmse'][0]:.4f}, {fp_errors['pos_rmse'][1]:.4f}, {fp_errors['pos_rmse'][2]:.4f}]")
            print(f"  Position RMSE 3D: {fp_errors['pos_rmse_3d']:.4f} m")
            print(f"  Z position mean error: {fp_errors['pos_err_z_mean']:.4f} m "
                  f"({'HW below sim' if fp_errors['pos_err_z_mean'] < 0 else 'HW above sim'})")
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback; traceback.print_exc()

        # so_rpy
        print(f"\n  --- so_rpy (direct dynamics, mass={SO_RPY_PARAMS['mass']}) ---")
        try:
            sorpy_result = simulate_so_rpy_numpy(
                data, sim_freq=args.sim_freq, ctrl_freq=args.ctrl_freq
            )
            sorpy_errors = compute_errors(data, sorpy_result)
            print(f"  Position RMSE [x,y,z]: [{sorpy_errors['pos_rmse'][0]:.4f}, {sorpy_errors['pos_rmse'][1]:.4f}, {sorpy_errors['pos_rmse'][2]:.4f}]")
            print(f"  Position RMSE 3D: {sorpy_errors['pos_rmse_3d']:.4f} m")
            print(f"  Z position mean error: {sorpy_errors['pos_err_z_mean']:.4f} m "
                  f"({'HW below sim' if sorpy_errors['pos_err_z_mean'] < 0 else 'HW above sim'})")
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback; traceback.print_exc()

        # Plot
        plot_commands(data, name, out_dir)
        if fp_result is not None and sorpy_result is not None:
            plot_comparison(data, fp_result, sorpy_result, fp_errors, sorpy_errors, name, out_dir)


if __name__ == '__main__':
    main()
