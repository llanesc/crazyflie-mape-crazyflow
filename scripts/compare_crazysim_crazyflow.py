#!/usr/bin/env python3
"""Compare CrazySim and Crazyflow thrust pipelines.

Can operate in two modes:
  1. Static force model comparison (no data needed) — plots both force models
     as a function of PWM to visualize the discrepancy.
  2. Data-driven comparison — takes CrazySim pipeline data (from
     crazysim_thrust_pipeline_logger.py) and/or HW rosbag data and compares.

Usage:
    # Static force model comparison only:
    python3 scripts/compare_crazysim_crazyflow.py --static-only --out-dir results/plots

    # With CrazySim pipeline data:
    python3 scripts/compare_crazysim_crazyflow.py hardware/logs/crazysim_pipeline.npz

    # With HW rosbag data (from extract_rosbag_data.py):
    python3 scripts/compare_crazysim_crazyflow.py --hw-data hardware/logs/rosbag_hw_data.npz
"""

import argparse
from pathlib import Path

import numpy as np


# ── CrazySim parameters (from CrtpUtils.h + SDF) ──
CRAZYSIM_THRUST_MAX = 0.18        # N per motor
CRAZYSIM_MOTOR_CONSTANT = 2.3375e-8  # F = kf * omega^2
CRAZYSIM_MAX_ROT_VEL = 2797.0     # rad/s
CRAZYSIM_PWM_MIN = 7000
CRAZYSIM_PWM_MAX = 65535
CRAZYSIM_MASS_SDF = 0.0404        # kg (from model.sdf.jinja)
CRAZYSIM_MASS_FIRMWARE = 0.0325   # kg (CF_MASS with thrust upgrade in firmware)

# ── Crazyflow cf2x_T350 parameters (from params.toml) ──
CF_THRUST_MAX = 0.18              # N per motor
CF_PWM_MAX = 65535
CF_RPM2THRUST = np.array([0.0, -7.167227176573658e-7, 2.9401303690194613e-10])
CF_MASS = 0.0379                  # kg (from drone-models data/params.toml)
CF_MASS_CTRL = 0.0325             # kg (from drone-controllers params.toml, used by Mellinger)


def crazysim_pwm2omega(pwm):
    """CrazySim PWM2OMEGA (from CrtpUtils.h)."""
    pwm = np.asarray(pwm, dtype=float)
    thrust_desired = (pwm / CRAZYSIM_PWM_MAX) * CRAZYSIM_THRUST_MAX
    omega = np.sqrt(np.maximum(thrust_desired, 0) / CRAZYSIM_MOTOR_CONSTANT)
    omega = np.minimum(omega, CRAZYSIM_MAX_ROT_VEL)
    omega = np.where(pwm < CRAZYSIM_PWM_MIN, 0.0, omega)
    return omega


def crazysim_omega2force(omega):
    """Gazebo MulticopterMotorModel: F = motorConstant * omega^2."""
    return CRAZYSIM_MOTOR_CONSTANT * np.asarray(omega) ** 2


def crazysim_pwm2force(pwm):
    """Full CrazySim pipeline: PWM -> omega -> force (per motor)."""
    return crazysim_omega2force(crazysim_pwm2omega(pwm))


def crazyflow_pwm2force_linear(pwm):
    """Crazyflow linear PWM-to-force (same as CrazySim, shared mapping)."""
    return (np.asarray(pwm, dtype=float) / CF_PWM_MAX) * CF_THRUST_MAX


def crazyflow_force2omega(force):
    """Crazyflow motor_force2rotor_vel using rpm2thrust polynomial.

    Inverts: force = c + b*rpm + a*rpm^2  (where c=0 for T350)
    """
    a = CF_RPM2THRUST[2]
    b = CF_RPM2THRUST[1]
    c = CF_RPM2THRUST[0]
    force = np.asarray(force, dtype=float)
    disc = b ** 2 - 4 * a * (c - force)
    disc = np.maximum(disc, 0)
    return (-b + np.sqrt(disc)) / (2 * a)


def crazyflow_omega2force(omega):
    """Crazyflow forward force model: F = c + b*rpm + a*rpm^2."""
    omega = np.asarray(omega, dtype=float)
    return CF_RPM2THRUST[0] + CF_RPM2THRUST[1] * omega + CF_RPM2THRUST[2] * omega ** 2


def crazyflow_pwm2force_full(pwm):
    """Full Crazyflow pipeline: PWM -> linear force -> omega (via rpm2thrust) -> actual force."""
    linear_force = crazyflow_pwm2force_linear(pwm)
    omega = crazyflow_force2omega(linear_force)
    return crazyflow_omega2force(omega)


def plot_static_force_comparison(out_dir: Path):
    """Plot both force models as f(PWM) over the full range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except ImportError:
        pass
    plt.rcParams["pdf.fonttype"] = 42

    pwm = np.linspace(CRAZYSIM_PWM_MIN, CRAZYSIM_PWM_MAX, 1000)

    # CrazySim: PWM -> omega -> F = kf*omega^2
    # By design of PWM2OMEGA, this roundtrips to the linear mapping
    force_crazysim = crazysim_pwm2force(pwm)

    # Crazyflow linear (shared): F = (pwm/65535) * 0.18
    force_cf_linear = crazyflow_pwm2force_linear(pwm)

    # Crazyflow full: PWM -> linear force -> omega (rpm2thrust inverse) -> force (rpm2thrust forward)
    force_cf_full = crazyflow_pwm2force_full(pwm)

    # Hover PWM for different masses
    hover_masses = {
        "SDF (0.0404 kg)": CRAZYSIM_MASS_SDF,
        "FW (0.0325 kg)": CRAZYSIM_MASS_FIRMWARE,
        "CF phys (0.0379 kg)": CF_MASS,
    }

    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    # Top: Force vs PWM
    ax = axes[0]
    ax.plot(pwm, force_crazysim * 1000, '-', linewidth=1.0, label='CrazySim (linear roundtrip)')
    ax.plot(pwm, force_cf_full * 1000, '--', linewidth=1.0, label='Crazyflow (rpm2thrust)')
    for name, m in hover_masses.items():
        hover_force = m * 9.81 / 4  # per motor
        ax.axhline(hover_force * 1000, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
        ax.annotate(name, xy=(CRAZYSIM_PWM_MAX * 0.55, hover_force * 1000),
                    fontsize=5, color='gray')
    ax.set_ylabel('Force per motor (mN)')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Bottom: Relative difference
    ax = axes[1]
    diff_pct = (force_cf_full - force_crazysim) / (force_crazysim + 1e-12) * 100
    ax.plot(pwm, diff_pct, 'r-', linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.set_xlabel('PWM')
    ax.set_ylabel('Difference (%)\n(Crazyflow - CrazySim)')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Static Force Model: CrazySim vs Crazyflow (cf2x\\_T350)', fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "force_model_comparison.pdf"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print key values
    print("\n--- Static Force Model Comparison ---")
    print(f"{'Mass (kg)':<18} {'Hover F/motor (mN)':<22} {'Hover PWM':<12} "
          f"{'CrazySim F (mN)':<18} {'Crazyflow F (mN)':<18} {'Diff (%)'}")
    for name, m in hover_masses.items():
        hover_f = m * 9.81 / 4
        hover_pwm = int(hover_f / CRAZYSIM_THRUST_MAX * CRAZYSIM_PWM_MAX)
        cs_f = crazysim_pwm2force(hover_pwm)
        cf_f = crazyflow_pwm2force_full(hover_pwm)
        diff = (cf_f - cs_f) / cs_f * 100
        print(f"{name:<18} {hover_f*1000:<22.3f} {hover_pwm:<12} "
              f"{cs_f*1000:<18.3f} {cf_f*1000:<18.3f} {diff:+.2f}%")


def plot_omega_comparison(out_dir: Path):
    """Plot omega (rotor velocity) vs PWM for both models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except ImportError:
        pass
    plt.rcParams["pdf.fonttype"] = 42

    pwm = np.linspace(CRAZYSIM_PWM_MIN, CRAZYSIM_PWM_MAX, 1000)

    omega_cs = crazysim_pwm2omega(pwm)
    force_linear = crazyflow_pwm2force_linear(pwm)
    omega_cf = crazyflow_force2omega(force_linear)

    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    ax = axes[0]
    ax.plot(pwm, omega_cs, '-', linewidth=1.0, label='CrazySim')
    ax.plot(pwm, omega_cf, '--', linewidth=1.0, label='Crazyflow')
    ax.set_ylabel('Rotor velocity (rad/s)')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    diff_pct = (omega_cf - omega_cs) / (omega_cs + 1e-12) * 100
    ax.plot(pwm, diff_pct, 'r-', linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.set_xlabel('PWM')
    ax.set_ylabel('Difference (%)\n(Crazyflow - CrazySim)')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Rotor Velocity: CrazySim vs Crazyflow (cf2x\\_T350)', fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "omega_comparison.pdf"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pipeline_data(npz_path: str, out_dir: Path):
    """Plot pipeline data collected from CrazySim logger."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except ImportError:
        pass
    plt.rcParams["pdf.fonttype"] = 42

    data = np.load(npz_path, allow_pickle=True)
    print(f"Loaded pipeline data: {list(data.keys())}")

    # Position
    if "stateEstimate_x" in data:
        fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)
        for i, (coord, label) in enumerate(zip(["x", "y", "z"], ["X", "Y", "Z"])):
            key = f"stateEstimate_{coord}"
            t = data[f"{key}_t"]
            axes[i].plot(t, data[key], linewidth=0.8)
            axes[i].set_ylabel(f'{label} (m)')
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('CrazySim State Estimate — Position', fontsize=9)
        fig.tight_layout()
        out_path = out_dir / "crazysim_position.pdf"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    # Motor PWMs and forces
    if "motor_m1req" in data:
        fig, axes = plt.subplots(3, 1, figsize=(5, 6), sharex=True)

        t = data["motor_m1req_t"]
        colors = ['C0', 'C1', 'C2', 'C3']

        # PWM
        ax = axes[0]
        for i, m in enumerate(["m1", "m2", "m3", "m4"]):
            pwm = np.clip(data[f"motor_{m}req"], 0, CRAZYSIM_PWM_MAX)
            ax.plot(t, pwm, linewidth=0.6, color=colors[i], label=m)
        ax.set_ylabel('PWM')
        ax.legend(fontsize=5, ncol=4)
        ax.grid(True, alpha=0.3)

        # Forces — both models
        ax = axes[1]
        total_cs = np.zeros_like(t)
        total_cf = np.zeros_like(t)
        for i, m in enumerate(["m1", "m2", "m3", "m4"]):
            pwm = np.clip(data[f"motor_{m}req"], 0, CRAZYSIM_PWM_MAX)
            f_cs = crazysim_pwm2force(pwm)
            f_cf = crazyflow_pwm2force_full(pwm)
            total_cs += f_cs
            total_cf += f_cf
            if i == 0:  # Only label once
                ax.plot(t, f_cs * 1000, '-', linewidth=0.5, color=colors[i],
                        label='CrazySim', alpha=0.7)
                ax.plot(t, f_cf * 1000, '--', linewidth=0.5, color=colors[i],
                        label='Crazyflow', alpha=0.7)
            else:
                ax.plot(t, f_cs * 1000, '-', linewidth=0.5, color=colors[i], alpha=0.7)
                ax.plot(t, f_cf * 1000, '--', linewidth=0.5, color=colors[i], alpha=0.7)
        ax.set_ylabel('Force/motor (mN)')
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.3)

        # Total thrust vs weight
        ax = axes[2]
        ax.plot(t, total_cs, '-', linewidth=0.8, label='CrazySim total')
        ax.plot(t, total_cf, '--', linewidth=0.8, label='Crazyflow total')
        for name, m in [("SDF", CRAZYSIM_MASS_SDF), ("FW", CRAZYSIM_MASS_FIRMWARE),
                         ("CF", CF_MASS)]:
            ax.axhline(m * 9.81, color='gray', linestyle=':', linewidth=0.5)
            ax.annotate(f'{name} weight', xy=(t[-1] * 0.7, m * 9.81),
                        fontsize=5, color='gray')
        ax.set_ylabel('Total thrust (N)')
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.3)

        fig.suptitle('CrazySim Pipeline — Motor PWMs & Forces', fontsize=9)
        fig.tight_layout()
        out_path = out_dir / "crazysim_motor_forces.pdf"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare CrazySim and Crazyflow thrust pipelines"
    )
    parser.add_argument("npz_path", nargs="?", default=None,
                        help="Path to CrazySim pipeline NPZ (from logger script)")
    parser.add_argument("--hw-data", default=None,
                        help="Path to HW rosbag NPZ (from extract_rosbag_data.py)")
    parser.add_argument("--static-only", action="store_true",
                        help="Only plot static force model comparison (no data needed)")
    parser.add_argument("--out-dir", default="results/plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # Always plot static comparison
    plot_static_force_comparison(out_dir)
    plot_omega_comparison(out_dir)

    # Plot pipeline data if provided
    if args.npz_path:
        plot_pipeline_data(args.npz_path, out_dir)

    # Plot HW data forces if provided
    if args.hw_data:
        data = np.load(args.hw_data, allow_pickle=True)
        drone_names = list(data['drone_names'])
        print(f"\nHW data drones: {drone_names}")
        for name in drone_names:
            pwms = data[f"{name}/cmd_thrust_pwm"]
            cmd_t = data[f"{name}/cmd_t"]
            f_cs = crazysim_pwm2force(pwms) * 4  # total (4 motors equal)
            f_cf = crazyflow_pwm2force_full(pwms / 4) * 4  # per-motor then *4
            f_linear = (pwms / CRAZYSIM_PWM_MAX) * CRAZYSIM_THRUST_MAX * 4

            print(f"\n  {name}:")
            print(f"    Mean PWM: {pwms.mean():.0f}")
            print(f"    CrazySim total thrust mean: {f_cs.mean():.4f} N")
            print(f"    Crazyflow total thrust mean: {f_cf.mean():.4f} N")
            print(f"    Linear mapping mean: {f_linear.mean():.4f} N")
            print(f"    Weight (0.0406 kg): {0.0406 * 9.81:.4f} N")


if __name__ == "__main__":
    main()
