#!/usr/bin/env python3
"""System identification for the so_rpy drone model using sim CSV data.

Reads the eval CSV files (with command columns) and runs the same sysid
pipeline as the rosbag version, for comparison.

Usage:
    python scripts/sysid_from_sim_csv.py <csv_path_or_dir> [--mass MASS]
"""

import os
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import bilinear, butter, lfilter, lfiltic

# Model mass default (cf2x_T350)
MASS = 0.0379


def svf_filter(y, t, f_c=6.0, N_deriv=2):
    """State Variable Filter for smooth derivatives."""
    if y.ndim == 1:
        y = y[None, :]
    batch_size, signal_length = y.shape
    N_ord = N_deriv + 2
    omega_c = 2 * np.pi * f_c
    f_s = 1 / np.mean(np.diff(t))

    b, a = butter(N=N_ord, Wn=omega_c, analog=True)
    b_dig, a_dig = bilinear(b, a, fs=f_s)
    a_flipped = np.flip(a)

    def _f(t_val, x, u):
        x_dot = []
        x_dot_last = 0
        for i in np.arange(1, N_ord):
            x_dot.append(x[i])
        for i in np.arange(0, N_ord):
            x_dot_last -= a_flipped[i] * x[i]
        x_dot_last += b[0] * u(t_val)
        x_dot.append(x_dot_last)
        return x_dot

    results = np.zeros((batch_size, N_deriv + 1, signal_length))
    for i in range(batch_size):
        pad = 100
        y_backwards = np.flip(y[i], axis=-1)
        y_backwards_padded = np.concatenate([np.ones(pad) * y_backwards[0], y_backwards])
        zi = lfiltic(b_dig, a_dig, y_backwards_padded, x=y_backwards_padded)
        y_backwards_filt, _ = lfilter(b_dig, a_dig, y_backwards_padded, axis=-1, zi=zi)
        u = interp1d(t, np.flip(y_backwards_filt[pad:], axis=-1), kind="linear",
                      fill_value="extrapolate")
        x0 = np.zeros(N_ord)
        x0[0] = y[i, 0]
        sol = solve_ivp(_f, [t[0], t[-1]], x0, t_eval=t, args=(u,))
        results[i] = sol.y[:-1]
    return results.squeeze()


def extract_drone_data_from_csv(csv_path: Path) -> dict:
    """Extract per-drone state and command data from a sim CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Detect observation dimension
    agent0_obs_cols = [c for c in df.columns if c.startswith('agent0_obs')]
    obs_dim = len(agent0_obs_cols)

    # Detect number of blue agents
    n_blue = 0
    while f'agent{n_blue}_obs0' in df.columns:
        n_blue += 1

    # Detect number of red agents
    n_red = 0
    while f'red{n_red}_pos_x' in df.columns:
        n_red += 1

    # Check for command columns
    has_cmds = 'blue0_cmd_roll' in df.columns

    if not has_cmds:
        raise ValueError("CSV does not contain command columns (blue0_cmd_roll, etc.)")

    print(f"  {n_blue} blue, {n_red} red, obs_dim={obs_dim}, {len(df)} samples")

    t = df['time'].values
    result = {}

    # Blue agents: extract state from observations
    for i in range(n_blue):
        prefix = f'agent{i}_obs'
        # Obs format (52-dim with rotation matrix):
        #   pos(3), vel(3), rot_matrix(9) = first 15 dims of own state
        # Or (46-dim with RPY):
        #   pos(3), vel(3), rpy(3), rpy_rates(3) = first 12 dims
        pos = df[[f'{prefix}{j}' for j in range(3)]].values
        vel = df[[f'{prefix}{j}' for j in range(3, 6)]].values

        if obs_dim == 52:
            # Rotation matrix format: obs[6:15] is flattened 3x3 rotation matrix
            rot_flat = df[[f'{prefix}{j}' for j in range(6, 15)]].values
            rot_matrices = rot_flat.reshape(-1, 3, 3)
            rpy = R.from_matrix(rot_matrices).as_euler('xyz')
            quat = R.from_matrix(rot_matrices).as_quat()
        else:
            # RPY format: obs[6:9]
            rpy = df[[f'{prefix}{j}' for j in range(6, 9)]].values
            quat = R.from_euler('xyz', rpy).as_quat()

        cmd_roll = df[f'blue{i}_cmd_roll'].values
        cmd_pitch = df[f'blue{i}_cmd_pitch'].values
        cmd_yaw = df[f'blue{i}_cmd_yaw'].values
        cmd_thrust = df[f'blue{i}_cmd_thrust'].values
        cmd_rpy = np.stack([cmd_roll, cmd_pitch, cmd_yaw], axis=-1)

        result[f'blue_{i+1}'] = {
            'time': t,
            'pos': pos,
            'vel': vel,
            'quat': quat,
            'rpy': rpy,
            'cmd_rpy': cmd_rpy,
            'cmd_f': cmd_thrust,
        }

    # Red agents: extract state from red columns + commands
    for i in range(n_red):
        pos = df[[f'red{i}_pos_x', f'red{i}_pos_y', f'red{i}_pos_z']].values
        vel = df[[f'red{i}_vel_x', f'red{i}_vel_y', f'red{i}_vel_z']].values
        rpy = df[[f'red{i}_roll', f'red{i}_pitch', f'red{i}_yaw']].values
        quat = R.from_euler('xyz', rpy).as_quat()
        active = df[f'red{i}_active'].values

        cmd_roll = df[f'red{i}_cmd_roll'].values
        cmd_pitch = df[f'red{i}_cmd_pitch'].values
        cmd_yaw = df[f'red{i}_cmd_yaw'].values
        cmd_thrust = df[f'red{i}_cmd_thrust'].values
        cmd_rpy = np.stack([cmd_roll, cmd_pitch, cmd_yaw], axis=-1)

        # Trim to active period
        active_mask = active > 0.5
        if not active_mask.all():
            first_inactive = np.argmin(active_mask)
            if first_inactive > 10:
                t_trim = t[:first_inactive]
                pos = pos[:first_inactive]
                vel = vel[:first_inactive]
                rpy = rpy[:first_inactive]
                quat = quat[:first_inactive]
                cmd_rpy = cmd_rpy[:first_inactive]
                cmd_thrust = cmd_thrust[:first_inactive]
            else:
                print(f"  red_{i+1}: Too few active samples ({first_inactive}), skipping")
                continue
        else:
            t_trim = t

        result[f'red_{i+1}'] = {
            'time': t_trim,
            'pos': pos,
            'vel': vel,
            'quat': quat,
            'rpy': rpy,
            'cmd_rpy': cmd_rpy,
            'cmd_f': cmd_thrust,
        }

    return result


def compute_svf_derivatives(data: dict) -> dict:
    """Compute SVF-filtered states and their derivatives."""
    t = data['time']
    svf_lin = svf_filter(data['pos'].T, t, f_c=6, N_deriv=3)
    data['SVF_pos'] = svf_lin[:, 0].T
    data['SVF_vel'] = svf_lin[:, 1].T
    data['SVF_acc'] = svf_lin[:, 2].T

    svf_rot = svf_filter(data['rpy'].T, t, f_c=8, N_deriv=3)
    data['SVF_rpy'] = svf_rot[:, 0].T
    data['SVF_drpy'] = svf_rot[:, 1].T
    data['SVF_ddrpy'] = svf_rot[:, 2].T
    data['SVF_quat'] = R.from_euler('xyz', data['SVF_rpy']).as_quat()

    svf_cmd_f = svf_filter(data['cmd_f'], t, f_c=6, N_deriv=1)
    data['SVF_cmd_f'] = svf_cmd_f[0]

    svf_cmd_rpy = svf_filter(data['cmd_rpy'].T, t, f_c=8, N_deriv=1)
    data['SVF_cmd_rpy'] = svf_cmd_rpy[:, 0].T

    return data


def sysid_translation_lsq(data: dict, mass: float, gravity: np.ndarray) -> dict:
    """Identify cmd_f_coef using linear least squares."""
    acc = data['SVF_acc']
    cmd_f = data['SVF_cmd_f']
    quat = data['SVF_quat']

    z_axis = R.from_quat(quat).as_matrix()[..., :, 2]
    acc_minus_g = acc - gravity[None, :]
    acc_z_body = np.sum(acc_minus_g * z_axis, axis=-1)

    A = cmd_f[:, None]
    b = acc_z_body
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cmd_f_coef = x[0] * mass

    A2 = np.column_stack([np.ones_like(cmd_f), cmd_f])
    x2, _, _, _ = np.linalg.lstsq(A2, acc_z_body, rcond=None)
    acc_coef_full = x2[0] * mass
    cmd_f_coef_full = x2[1] * mass

    acc_z_pred = cmd_f_coef / mass * cmd_f
    rmse = np.sqrt(np.mean((acc_z_body - acc_z_pred)**2))
    ss_res = np.sum((acc_z_body - acc_z_pred)**2)
    ss_tot = np.sum((acc_z_body - np.mean(acc_z_body))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    acc_pred_3d = (cmd_f_coef / mass * cmd_f)[:, None] * z_axis + gravity[None, :]
    rmse_3d = np.sqrt(np.mean((acc - acc_pred_3d)**2))

    print(f"  cmd_f_coef = {cmd_f_coef:.6f}")
    print(f"  (with acc_coef: acc_coef={acc_coef_full:.6f}, cmd_f_coef={cmd_f_coef_full:.6f})")
    print(f"  Z-axis RMSE = {rmse:.4f} m/s², R² = {r2:.4f}")
    print(f"  3D acc RMSE = {rmse_3d:.4f} m/s²")

    return {'cmd_f_coef': cmd_f_coef, 'acc_coef': acc_coef_full, 'cmd_f_coef_full': cmd_f_coef_full}


def sysid_rotation_lsq(data: dict) -> dict:
    """Identify rpy_coef, rpy_rates_coef, cmd_rpy_coef using linear least squares."""
    rpy = data['SVF_rpy']
    drpy = data['SVF_drpy']
    ddrpy = data['SVF_ddrpy']
    cmd_rpy = data['SVF_cmd_rpy']

    axis_names = ['roll', 'pitch', 'yaw']
    coefs = np.zeros((3, 3))

    for i, axis in enumerate(axis_names):
        A = np.column_stack([rpy[:, i], drpy[:, i], cmd_rpy[:, i]])
        b = ddrpy[:, i]
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        coefs[i] = x

        pred = A @ x
        rmse = np.sqrt(np.mean((b - pred)**2))
        ss_res = np.sum((b - pred)**2)
        ss_tot = np.sum((b - np.mean(b))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  {axis}: rpy_coef={x[0]:.2f}, rpy_rates_coef={x[1]:.2f}, "
              f"cmd_rpy_coef={x[2]:.2f}  (RMSE={rmse:.4f}, R²={r2:.4f})")

    return {
        'rpy_coef': coefs[:, 0],
        'rpy_rates_coef': coefs[:, 1],
        'cmd_rpy_coef': coefs[:, 2],
    }


def run_sysid(drone_data: dict, mass: float) -> dict:
    """Run system identification for each drone."""
    gravity = np.array([0.0, 0.0, -9.81])
    results = {}

    for name, data in drone_data.items():
        print(f"\n{'='*60}")
        print(f"  System Identification: {name}")
        print(f"{'='*60}")

        data = compute_svf_derivatives(data)

        print("\n--- Translation ---")
        try:
            trans_params = sysid_translation_lsq(data, mass, gravity)
        except Exception as e:
            print(f"  Failed: {e}")
            trans_params = {}

        print("\n--- Rotation ---")
        try:
            rot_params = sysid_rotation_lsq(data)
        except Exception as e:
            print(f"  Failed: {e}")
            rot_params = {}

        results[name] = {**trans_params, **rot_params}

    return results


def main():
    parser = argparse.ArgumentParser(description='SysID from sim CSV for so_rpy model')
    parser.add_argument('path', type=str, help='Path to sim CSV file or directory of CSVs')
    parser.add_argument('--mass', type=float, default=MASS, help=f'Drone mass in kg (default: {MASS})')
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_dir():
        csv_files = sorted(path.glob('*_sim.csv'))
    elif path.is_file():
        csv_files = [path]
    else:
        print(f"Error: Path not found: {path}")
        return 1

    if not csv_files:
        print(f"No sim CSV files found in {path}")
        return 1

    print(f"Mass: {args.mass} kg")
    print(f"Files: {len(csv_files)}")

    all_results = {}
    for csv_file in csv_files:
        print(f"\n{'#'*70}")
        print(f"  File: {csv_file.name}")
        print(f"{'#'*70}")

        try:
            drone_data = extract_drone_data_from_csv(csv_file)
        except Exception as e:
            print(f"  Error extracting data: {e}")
            continue

        results = run_sysid(drone_data, mass=args.mass)

        for name, params in results.items():
            if name not in all_results:
                all_results[name] = []
            all_results[name].append(params)

    # Aggregate summary
    print(f"\n{'='*70}")
    print(f"  AGGREGATE SUMMARY (across {len(csv_files)} episodes)")
    print(f"{'='*70}")

    for name in sorted(all_results.keys()):
        params_list = all_results[name]
        print(f"\n{name} ({len(params_list)} episodes):")

        # cmd_f_coef
        cmd_f_vals = [p['cmd_f_coef'] for p in params_list if 'cmd_f_coef' in p]
        if cmd_f_vals:
            print(f"  cmd_f_coef: {np.mean(cmd_f_vals):.6f} ± {np.std(cmd_f_vals):.6f}")

        # rpy_coef
        rpy_vals = [p['rpy_coef'] for p in params_list if 'rpy_coef' in p]
        if rpy_vals:
            rpy_arr = np.array(rpy_vals)
            print(f"  rpy_coef:       mean={np.mean(rpy_arr, axis=0)}, std={np.std(rpy_arr, axis=0)}")

        rpy_rates_vals = [p['rpy_rates_coef'] for p in params_list if 'rpy_rates_coef' in p]
        if rpy_rates_vals:
            arr = np.array(rpy_rates_vals)
            print(f"  rpy_rates_coef: mean={np.mean(arr, axis=0)}, std={np.std(arr, axis=0)}")

        cmd_rpy_vals = [p['cmd_rpy_coef'] for p in params_list if 'cmd_rpy_coef' in p]
        if cmd_rpy_vals:
            arr = np.array(cmd_rpy_vals)
            print(f"  cmd_rpy_coef:   mean={np.mean(arr, axis=0)}, std={np.std(arr, axis=0)}")

    # Reference values
    print(f"\n{'='*70}")
    print("  cf2x_T350 reference values:")
    print(f"{'='*70}")
    try:
        from crazyflow.sim.physics import load_params
        ref = load_params("so_rpy", "cf2x_T350")
        for key in ['cmd_f_coef', 'rpy_coef', 'rpy_rates_coef', 'cmd_rpy_coef']:
            if key in ref:
                print(f"  {key}: {ref[key]}")
    except Exception:
        print("  (Could not load reference params)")

    return 0


if __name__ == '__main__':
    exit(main())
