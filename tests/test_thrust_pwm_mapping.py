"""Compare thrust-to-PWM mapping from pursuer_evader.py vs cf2x_T350 firmware model.

Testing hypothesis: The cf2x_T350 params assume the firmware vmotor mapping is used,
so the full chain PWM -> voltage -> RPM -> thrust results in approximately linear behavior.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Hardware mapping from pursuer_evader.py (lines 51-56)
    omega_scale = 6462.1  # rad/s per sqrt(N)
    pwm_scale = 24.5307
    pwm_offset = 380.8359  # rad/s

    # cf2x_T350 params from params.toml
    # rpm2thrust polynomial: thrust = c0 + c1*rpm + c2*rpm^2 (rpm in RPM)
    rpm2thrust_coeffs = np.array([0.0, -7.167227176573658e-7, 2.9401303690194613e-10])

    # vmotor2rpm: rpm = c0 + c1 * vmotor (where vmotor = voltage, 0 to ~4.2V)
    vmotor2rpm = np.array([2977.884883031915, 8101.0293594093055])

    # vmotor2thrust polynomial: thrust = c0 + c1*v + c2*v^2 + c3*v^3
    vmotor2thrust = np.array([0.006728127583707208, 0.01011557616217668,
                              0.010263198062061085, 0.0028358638322392503])

    # Supply voltage (nominal LiPo)
    v_supply = 4.2

    # Conversion factors
    RAD_S_TO_RPM = 60.0 / (2.0 * np.pi)

    # cf2x_T350 thrust limits (per motor)
    thrust_min = 0.01922636758983749  # N per motor
    thrust_max = 0.18  # N per motor

    # PWM limits
    pwm_min = 7000
    pwm_max = 65535

    # === Test the full chain: PWM -> voltage -> RPM -> thrust ===
    pwm_range = np.linspace(pwm_min, pwm_max, 200)

    # PWM to voltage (linear): voltage = pwm / pwm_max * v_supply
    voltage = pwm_range / pwm_max * v_supply

    # Voltage to RPM using vmotor2rpm: rpm = c0 + c1 * voltage
    rpm_from_voltage = vmotor2rpm[0] + vmotor2rpm[1] * voltage

    # RPM to thrust using rpm2thrust: thrust = c0 + c1*rpm + c2*rpm^2
    c0, c1, c2 = rpm2thrust_coeffs
    thrust_from_rpm = c0 + c1 * rpm_from_voltage + c2 * rpm_from_voltage**2

    # Also compute thrust directly from voltage using vmotor2thrust
    thrust_from_voltage = (vmotor2thrust[0] + vmotor2thrust[1] * voltage +
                           vmotor2thrust[2] * voltage**2 + vmotor2thrust[3] * voltage**3)

    # Linear assumption: thrust = pwm / pwm_max * thrust_max
    thrust_linear = pwm_range / pwm_max * thrust_max

    # === Reverse: Given thrust, what PWM does each model give? ===
    thrust_range = np.linspace(thrust_min, thrust_max, 200)

    # Hardware mapping (pursuer_evader.py)
    omega_hw = omega_scale * np.sqrt(thrust_range)
    pwm_hw = pwm_scale * (omega_hw - pwm_offset)
    pwm_hw = np.clip(pwm_hw, 0, 65535)

    # Linear simulation
    pwm_sim_linear = thrust_range / thrust_max * pwm_max

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: PWM to Thrust (forward direction)
    ax1 = axes[0, 0]
    ax1.plot(pwm_range, thrust_from_rpm * 1000, 'b-', linewidth=2, label='via rpm2thrust')
    ax1.plot(pwm_range, thrust_from_voltage * 1000, 'g--', linewidth=2, label='via vmotor2thrust')
    ax1.plot(pwm_range, thrust_linear * 1000, 'r:', linewidth=2, label='Linear assumption')
    ax1.axhline(thrust_min * 1000, color='gray', linestyle=':', alpha=0.7)
    ax1.axhline(thrust_max * 1000, color='gray', linestyle=':', alpha=0.7)
    ax1.axvline(pwm_min, color='orange', linestyle=':', alpha=0.7)
    ax1.set_xlabel('PWM')
    ax1.set_ylabel('Thrust per motor (mN)')
    ax1.set_title('PWM to Thrust (Forward Chain)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Thrust to PWM (reverse direction)
    ax2 = axes[0, 1]
    ax2.plot(thrust_range * 1000, pwm_hw, 'b-', linewidth=2, label='Hardware (pursuer_evader.py)')
    ax2.plot(thrust_range * 1000, pwm_sim_linear, 'r--', linewidth=2, label='Simulation (linear)')
    ax2.axvline(thrust_min * 1000, color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(thrust_max * 1000, color='gray', linestyle=':', alpha=0.7)
    ax2.axhline(pwm_min, color='orange', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Thrust per motor (mN)')
    ax2.set_ylabel('PWM')
    ax2.set_title('Thrust to PWM (Reverse)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Compare rpm2thrust vs vmotor2thrust chains
    ax3 = axes[1, 0]
    diff = (thrust_from_rpm - thrust_from_voltage) * 1000
    ax3.plot(pwm_range, diff, 'g-', linewidth=2)
    ax3.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax3.axvline(pwm_min, color='orange', linestyle=':', alpha=0.7)
    ax3.set_xlabel('PWM')
    ax3.set_ylabel('Thrust Difference (mN)')
    ax3.set_title('rpm2thrust - vmotor2thrust (should be ~0 if consistent)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Linearity check - deviation from linear
    ax4 = axes[1, 1]
    deviation_rpm = (thrust_from_rpm - thrust_linear) * 1000
    deviation_vmotor = (thrust_from_voltage - thrust_linear) * 1000
    ax4.plot(pwm_range, deviation_rpm, 'b-', linewidth=2, label='rpm2thrust - linear')
    ax4.plot(pwm_range, deviation_vmotor, 'g--', linewidth=2, label='vmotor2thrust - linear')
    ax4.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax4.axvline(pwm_min, color='orange', linestyle=':', alpha=0.7)
    ax4.set_xlabel('PWM')
    ax4.set_ylabel('Deviation from Linear (mN)')
    ax4.set_title('Deviation from Linear PWM-Thrust Assumption')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('thrust_pwm_comparison.png', dpi=150)
    plt.show()

    # Print summary
    print("\n=== PWM to Thrust Chain Analysis ===")
    print(f"PWM range: {pwm_min} - {pwm_max}")
    print(f"\nThrust via rpm2thrust chain: {thrust_from_rpm.min()*1000:.1f} - {thrust_from_rpm.max()*1000:.1f} mN")
    print(f"Thrust via vmotor2thrust: {thrust_from_voltage.min()*1000:.1f} - {thrust_from_voltage.max()*1000:.1f} mN")
    print(f"Thrust linear assumption: {thrust_linear.min()*1000:.1f} - {thrust_linear.max()*1000:.1f} mN")
    print(f"\nMax deviation from linear:")
    print(f"  rpm2thrust: {np.abs(deviation_rpm).max():.1f} mN")
    print(f"  vmotor2thrust: {np.abs(deviation_vmotor).max():.1f} mN")
    print(f"\nConsistency check (rpm2thrust vs vmotor2thrust):")
    print(f"  Max difference: {np.abs(diff).max():.2f} mN")


if __name__ == '__main__':
    main()
