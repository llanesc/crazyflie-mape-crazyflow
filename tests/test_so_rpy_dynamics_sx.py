"""Tests for SX-based so_rpy dynamics consistency with drone-models MX version.

Verifies that:
1. symbolic_dynamics_euler_sx matches drone_models.so_rpy.symbolic_dynamics_euler
2. symbolic_dynamics_sx matches drone_models.so_rpy.symbolic_dynamics
3. RK4 integrator produces consistent results
4. SX rotation utilities match MX versions
"""

import casadi as cs
import numpy as np
import pytest
from drone_models.core import load_params
from drone_models import so_rpy as dm_so_rpy
from drone_models.utils import rotation as dm_rotation
from scipy.spatial.transform import Rotation as R

from crazyflie_mape_crazyflow.leap_c.so_rpy_dynamics_sx import (
    integrate_euler_sx,
    integrate_rk4_sx,
    sx_ang_vel2rpy_rates,
    sx_quat2euler_xyz,
    sx_quat2matrix,
    sx_rpy2matrix,
    sx_rpy_rates2ang_vel,
    sx_rpy_rates_deriv2ang_vel_deriv,
    symbolic_dynamics_euler_sx,
    symbolic_dynamics_sx,
)

DRONE_MODEL = "cf2x_T350"
MPC_MODEL = "so_rpy"


@pytest.fixture
def drone_params():
    """Load drone parameters for testing."""
    return load_params(MPC_MODEL, DRONE_MODEL)


@pytest.fixture
def common_kwargs(drone_params):
    """Common kwargs for dynamics functions."""
    return dict(
        model_rotor_vel=False,
        mass=float(drone_params["mass"]),
        gravity_vec=drone_params["gravity_vec"],
        J=drone_params["J"],
        J_inv=drone_params["J_inv"],
        acc_coef=drone_params["acc_coef"],
        cmd_f_coef=drone_params["cmd_f_coef"],
        rpy_coef=drone_params["rpy_coef"],
        rpy_rates_coef=drone_params["rpy_rates_coef"],
        cmd_rpy_coef=drone_params["cmd_rpy_coef"],
    )


# Test states: various orientations and angular velocities
TEST_QUATS = [
    np.array([0, 0, 0, 1]),  # Identity
    R.from_euler("xyz", [0.1, 0.05, 0.2]).as_quat(),  # Small angles
    R.from_euler("xyz", [0.3, -0.2, 0.5]).as_quat(),  # Moderate angles
    R.from_euler("xyz", [-0.4, 0.3, -0.1]).as_quat(),  # Negative angles
]

TEST_ANG_VELS = [
    np.array([0, 0, 0]),
    np.array([0.5, -0.3, 0.1]),
    np.array([1.0, 0.5, -0.8]),
    np.array([-0.2, 0.7, 0.3]),
]

TEST_CONTROLS = [
    np.array([0, 0, 0, 0.4]),  # Hover-ish
    np.array([0.05, -0.03, 0.01, 0.45]),  # Small commands
    np.array([0.1, 0.1, -0.05, 0.35]),  # Moderate commands
]


class TestRotationUtilities:
    """Test SX rotation utilities against MX/scipy reference implementations."""

    @pytest.mark.parametrize("quat", TEST_QUATS)
    def test_quat2euler_xyz(self, quat):
        """Test SX quaternion to Euler conversion matches scipy."""
        # Reference: scipy
        expected = R.from_quat(quat).as_euler("xyz")

        # SX version
        q_sym = cs.SX.sym("q", 4)
        euler_sx = sx_quat2euler_xyz(q_sym)
        f = cs.Function("f", [q_sym], [euler_sx])
        result = np.array(f(quat)).flatten()

        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("quat", TEST_QUATS)
    def test_quat2matrix(self, quat):
        """Test SX quaternion to rotation matrix matches scipy."""
        expected = R.from_quat(quat).as_matrix()

        q_sym = cs.SX.sym("q", 4)
        rot_sx = sx_quat2matrix(q_sym)
        f = cs.Function("f", [q_sym], [rot_sx])
        result = np.array(f(quat)).reshape(3, 3)

        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("quat", TEST_QUATS)
    def test_rpy2matrix_matches_quat2matrix(self, quat):
        """Test that rpy2matrix and quat2matrix produce the same result."""
        rpy = R.from_quat(quat).as_euler("xyz")

        rpy_sym = cs.SX.sym("rpy", 3)
        rot_rpy = sx_rpy2matrix(rpy_sym)
        f_rpy = cs.Function("f", [rpy_sym], [rot_rpy])

        q_sym = cs.SX.sym("q", 4)
        rot_quat = sx_quat2matrix(q_sym)
        f_quat = cs.Function("f", [q_sym], [rot_quat])

        result_rpy = np.array(f_rpy(rpy)).reshape(3, 3)
        result_quat = np.array(f_quat(quat)).reshape(3, 3)

        np.testing.assert_allclose(result_rpy, result_quat, atol=1e-10)

    @pytest.mark.parametrize("quat", TEST_QUATS)
    @pytest.mark.parametrize("ang_vel", TEST_ANG_VELS)
    def test_ang_vel2rpy_rates(self, quat, ang_vel):
        """Test SX angular velocity to RPY rates matches drone_models."""
        # Reference: drone_models MX Function
        expected = np.array(dm_rotation.cs_ang_vel2rpy_rates(quat, ang_vel)).flatten()

        # SX version (takes rpy instead of quat)
        rpy = R.from_quat(quat).as_euler("xyz")
        rpy_sym = cs.SX.sym("rpy", 3)
        w_sym = cs.SX.sym("w", 3)
        drpy_sx = sx_ang_vel2rpy_rates(rpy_sym, w_sym)
        f = cs.Function("f", [rpy_sym, w_sym], [drpy_sx])
        result = np.array(f(rpy, ang_vel)).flatten()

        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("quat", TEST_QUATS)
    @pytest.mark.parametrize("ang_vel", TEST_ANG_VELS)
    def test_rpy_rates_roundtrip(self, quat, ang_vel):
        """Test ang_vel -> rpy_rates -> ang_vel roundtrip."""
        rpy = R.from_quat(quat).as_euler("xyz")

        rpy_sym = cs.SX.sym("rpy", 3)
        w_sym = cs.SX.sym("w", 3)

        drpy = sx_ang_vel2rpy_rates(rpy_sym, w_sym)
        w_recovered = sx_rpy_rates2ang_vel(rpy_sym, drpy)
        f = cs.Function("f", [rpy_sym, w_sym], [w_recovered])
        result = np.array(f(rpy, ang_vel)).flatten()

        np.testing.assert_allclose(result, ang_vel, atol=1e-10)

    @pytest.mark.parametrize("quat", TEST_QUATS)
    def test_rpy_rates_deriv2ang_vel_deriv(self, quat):
        """Test SX RPY rates derivative conversion matches drone_models."""
        rpy = R.from_quat(quat).as_euler("xyz")
        rpy_rates = np.array([0.5, -0.3, 0.1])
        rpy_rates_deriv = np.array([1.0, -0.5, 0.2])

        # Reference: drone_models MX Function
        expected = np.array(
            dm_rotation.cs_rpy_rates_deriv2ang_vel_deriv(quat, rpy_rates, rpy_rates_deriv)
        ).flatten()

        # SX version
        rpy_sym = cs.SX.sym("rpy", 3)
        dr_sym = cs.SX.sym("dr", 3)
        ddr_sym = cs.SX.sym("ddr", 3)
        result_sx = sx_rpy_rates_deriv2ang_vel_deriv(rpy_sym, dr_sym, ddr_sym)
        f = cs.Function("f", [rpy_sym, dr_sym, ddr_sym], [result_sx])
        result = np.array(f(rpy, rpy_rates, rpy_rates_deriv)).flatten()

        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestEulerDynamicsConsistency:
    """Test SX Euler dynamics against MX version from drone_models."""

    def test_euler_dynamics_matches_mx(self, common_kwargs):
        """Test symbolic_dynamics_euler_sx matches drone_models.so_rpy.symbolic_dynamics_euler."""
        # Build SX dynamics
        X_dot_sx, X_sx, U_sx, Y_sx = symbolic_dynamics_euler_sx(**common_kwargs)
        f_sx = cs.Function("f_sx", [X_sx, U_sx], [X_dot_sx, Y_sx])

        # Build MX dynamics
        X_dot_mx, X_mx, U_mx, Y_mx = dm_so_rpy.symbolic_dynamics_euler(**common_kwargs)
        f_mx = cs.Function("f_mx", [X_mx, U_mx], [X_dot_mx, Y_mx])

        # Test at multiple states
        for quat in TEST_QUATS:
            rpy = R.from_quat(quat).as_euler("xyz")
            for ang_vel in TEST_ANG_VELS:
                rpy_rates = np.array(dm_rotation.cs_ang_vel2rpy_rates(quat, ang_vel)).flatten()
                for ctrl in TEST_CONTROLS:
                    # Euler state: [pos, rpy, vel, drpy]
                    x_euler = np.concatenate([
                        [1.0, 0.5, 1.2],  # pos
                        rpy,
                        [0.3, -0.1, 0.2],  # vel
                        rpy_rates,
                    ])

                    xdot_sx, y_sx = f_sx(x_euler, ctrl)
                    xdot_mx, y_mx = f_mx(x_euler, ctrl)

                    np.testing.assert_allclose(
                        np.array(xdot_sx).flatten(),
                        np.array(xdot_mx).flatten(),
                        atol=1e-10,
                        err_msg=f"X_dot mismatch for rpy={rpy}, ang_vel={ang_vel}",
                    )
                    np.testing.assert_allclose(
                        np.array(y_sx).flatten(),
                        np.array(y_mx).flatten(),
                        atol=1e-10,
                        err_msg=f"Y mismatch for rpy={rpy}, ang_vel={ang_vel}",
                    )


class TestQuaternionDynamicsConsistency:
    """Test SX quaternion dynamics against MX version from drone_models."""

    def test_dynamics_matches_mx(self, common_kwargs):
        """Test symbolic_dynamics_sx matches drone_models.so_rpy.symbolic_dynamics."""
        # Build SX dynamics
        X_dot_sx, X_sx, U_sx, Y_sx = symbolic_dynamics_sx(**common_kwargs)
        f_sx = cs.Function("f_sx", [X_sx, U_sx], [X_dot_sx, Y_sx])

        # Build MX dynamics
        X_dot_mx, X_mx, U_mx, Y_mx = dm_so_rpy.symbolic_dynamics(**common_kwargs)
        f_mx = cs.Function("f_mx", [X_mx, U_mx], [X_dot_mx, Y_mx])

        for quat in TEST_QUATS:
            for ang_vel in TEST_ANG_VELS:
                for ctrl in TEST_CONTROLS:
                    # Quaternion state: [pos, quat, vel, ang_vel]
                    x_quat = np.concatenate([
                        [1.0, 0.5, 1.2],  # pos
                        quat,
                        [0.3, -0.1, 0.2],  # vel
                        ang_vel,
                    ])

                    xdot_sx, y_sx = f_sx(x_quat, ctrl)
                    xdot_mx, y_mx = f_mx(x_quat, ctrl)

                    np.testing.assert_allclose(
                        np.array(xdot_sx).flatten(),
                        np.array(xdot_mx).flatten(),
                        atol=1e-8,
                        err_msg=f"X_dot mismatch for quat={quat}, ang_vel={ang_vel}, ctrl={ctrl}",
                    )
                    np.testing.assert_allclose(
                        np.array(y_sx).flatten(),
                        np.array(y_mx).flatten(),
                        atol=1e-8,
                        err_msg=f"Y mismatch for quat={quat}, ang_vel={ang_vel}",
                    )

    def test_dimensions(self, common_kwargs):
        """Test that SX dynamics has correct dimensions."""
        X_dot, X, U, Y = symbolic_dynamics_sx(**common_kwargs)
        assert X.shape == (13, 1), f"State should be 13D, got {X.shape}"
        assert U.shape == (4, 1), f"Control should be 4D, got {U.shape}"
        assert X_dot.shape == (13, 1), f"State derivative should be 13D, got {X_dot.shape}"
        assert Y.shape == (7, 1), f"Output should be 7D, got {Y.shape}"

    def test_hover_equilibrium(self, common_kwargs, drone_params):
        """Test that hover is an equilibrium point (zero derivatives except gravity balance)."""
        X_dot, X, U, _ = symbolic_dynamics_sx(**common_kwargs)
        f = cs.Function("f", [X, U], [X_dot])

        mass = float(drone_params["mass"])
        acc_coef = float(drone_params["acc_coef"])
        cmd_f_coef = float(drone_params["cmd_f_coef"])

        # Hover thrust: acc_coef + cmd_f_coef * thrust_hover = mass * g
        g = 9.81
        thrust_hover = (mass * g - acc_coef) / cmd_f_coef

        # Identity quaternion, zero velocity, zero angular velocity
        x_hover = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        u_hover = np.array([0, 0, 0, thrust_hover])

        xdot = np.array(f(x_hover, u_hover)).flatten()

        # All derivatives should be ~0 at hover
        np.testing.assert_allclose(xdot, np.zeros(13), atol=1e-8)


class TestRK4Integrator:
    """Test RK4 integrator."""

    def test_rk4_vs_euler_convergence(self, common_kwargs):
        """Test that RK4 is more accurate than Euler for a given step size.

        Uses a fine Euler integration as reference truth.
        """
        X_dot, X, U, _ = symbolic_dynamics_sx(**common_kwargs)

        dt = 0.02
        dt_fine = 0.001
        n_fine_steps = int(dt / dt_fine)

        # RK4 single step
        X_next_rk4 = integrate_rk4_sx(X_dot, X, U, dt)
        f_rk4 = cs.Function("f_rk4", [X, U], [X_next_rk4])

        # Euler single step
        X_next_euler = integrate_euler_sx(X_dot, X, U, dt)
        f_euler = cs.Function("f_euler", [X, U], [X_next_euler])

        # Fine Euler (reference)
        X_next_fine = integrate_euler_sx(X_dot, X, U, dt_fine)
        f_fine = cs.Function("f_fine", [X, U], [X_next_fine])

        # Test state
        quat = R.from_euler("xyz", [0.1, 0.05, 0.2]).as_quat()
        x0 = np.concatenate([[0, 0, 1], quat, [0.5, -0.3, 0.1], [0.3, -0.2, 0.1]])
        u0 = np.array([0.05, -0.03, 0.01, 0.4])

        # Fine Euler integration (many small steps)
        x_ref = x0.copy()
        for _ in range(n_fine_steps):
            x_ref = np.array(f_fine(x_ref, u0)).flatten()

        # Single-step results
        x_rk4 = np.array(f_rk4(x0, u0)).flatten()
        x_euler = np.array(f_euler(x0, u0)).flatten()

        # RK4 should be closer to reference than Euler
        err_rk4 = np.linalg.norm(x_rk4 - x_ref)
        err_euler = np.linalg.norm(x_euler - x_ref)
        assert err_rk4 < err_euler, (
            f"RK4 error ({err_rk4:.2e}) should be less than Euler error ({err_euler:.2e})"
        )

    def test_rk4_euler_dynamics(self, common_kwargs):
        """Test RK4 also works with 12D Euler dynamics."""
        X_dot, X, U, _ = symbolic_dynamics_euler_sx(**common_kwargs)
        dt = 0.02

        X_next = integrate_rk4_sx(X_dot, X, U, dt)
        f = cs.Function("f", [X, U], [X_next])

        rpy = np.array([0.1, 0.05, 0.2])
        x0 = np.concatenate([[0, 0, 1], rpy, [0.3, -0.1, 0.2], [0.5, -0.3, 0.1]])
        u0 = np.array([0.05, -0.03, 0.01, 0.4])

        result = np.array(f(x0, u0)).flatten()
        assert result.shape == (12,)
        assert np.all(np.isfinite(result))
