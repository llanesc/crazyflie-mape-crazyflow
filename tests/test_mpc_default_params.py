"""Test MPC with LINEAR_LS style parameter scaling.

In LINEAR_LS, the cost is: J = 0.5 * ||Vx*x + Vu*u - y_ref||_W^2

The NN outputs:
1. W (weights) - log scaled for orders of magnitude control
2. y_ref (references) - linearly scaled to physical bounds

This maps to EXTERNAL cost (J = 0.5*x'Qx + p'x) via:
- Q = W (diagonal weights)
- p = -W * y_ref (derived from least squares expansion)
"""

import numpy as np
import torch

from crazyflie_mape_crazyflow.leap_c import QuadrotorPlanner, QuadrotorPlannerConfig
from crazyflie_mape_crazyflow.leap_c.quadrotor_ocp_external import (
    Q_STATE_SIZE,
    Q_CTRL_SIZE,
    P_X_SIZE,
    P_U_SIZE,
)


def scale_parameters_linear_ls(
    net_out: torch.Tensor,
    batch_size: int,
    planner: QuadrotorPlanner,
    x0: torch.Tensor,
) -> torch.Tensor:
    """Scale parameters using LINEAR_LS approach.

    NN outputs W (weights) and y_ref (references) separately.
    Then maps to EXTERNAL cost parameters (Q, p).
    """
    param_info = planner.get_param_structure_info()
    n_state_stages = param_info['n_state_stages']
    n_ctrl_stages = param_info['n_ctrl_stages']

    # Get physical params
    drone_params = planner.drone_params
    mass = float(drone_params["mass"])
    gravity = float(np.abs(drone_params["gravity_vec"][2]))
    cmd_f_coef = float(drone_params["cmd_f_coef"])
    hover_thrust = (mass * gravity) / cmd_f_coef

    # === Define scaling bounds ===

    # Log scaling for weights (W): W = 10^(log_w)
    # Range: [10^min_log, 10^max_log]
    w_state_min_log = torch.tensor([-1., -1., -1., -2., -2., -2., -1., -1., -1., -1., -1., -1.])  # ~0.1 to 0.01
    w_state_max_log = torch.tensor([2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1.])  # ~100 to 10
    w_ctrl_min_log = torch.tensor([-1., -1., -1., -1.])
    w_ctrl_max_log = torch.tensor([1., 1., 1., 1.])

    # Linear scaling for references (y_ref)
    # State refs: physical bounds
    state_ref_min = torch.tensor([-3., -3., 0., -0.5, -0.5, -3.14, -5., -5., -5., -10., -10., -10.])
    state_ref_max = torch.tensor([3., 3., 3., 0.5, 0.5, 3.14, 5., 5., 5., 10., 10., 10.])
    # Control refs: action bounds
    ctrl_ref_min = torch.tensor([-0.5, -0.5, -0.1, 0.18])  # roll, pitch, yaw, thrust_min
    ctrl_ref_max = torch.tensor([0.5, 0.5, 0.1, 0.72])  # thrust_max

    # === Split NN output into W and y_ref ===
    # First half: weights, Second half: references
    n_state = Q_STATE_SIZE
    n_ctrl = Q_CTRL_SIZE

    w_state_total = n_state * n_state_stages
    w_ctrl_total = n_ctrl * n_ctrl_stages
    yref_state_total = n_state * n_state_stages
    yref_ctrl_total = n_ctrl * n_ctrl_stages

    idx = 0
    w_state_raw = net_out[:, idx:idx + w_state_total]
    idx += w_state_total
    w_ctrl_raw = net_out[:, idx:idx + w_ctrl_total]
    idx += w_ctrl_total
    yref_state_raw = net_out[:, idx:idx + yref_state_total]
    idx += yref_state_total
    yref_ctrl_raw = net_out[:, idx:idx + yref_ctrl_total]

    # Reshape to per-stage
    w_state_per_stage = w_state_raw.reshape(batch_size, n_state_stages, n_state)
    w_ctrl_per_stage = w_ctrl_raw.reshape(batch_size, n_ctrl_stages, n_ctrl)
    yref_state_per_stage = yref_state_raw.reshape(batch_size, n_state_stages, n_state)
    yref_ctrl_per_stage = yref_ctrl_raw.reshape(batch_size, n_ctrl_stages, n_ctrl)

    # === Apply scalings ===

    # Log scaling for W: W = 10^(min_log + raw * (max_log - min_log))
    log_w_state = w_state_min_log + w_state_per_stage * (w_state_max_log - w_state_min_log)
    W_state = torch.pow(10., log_w_state)

    log_w_ctrl = w_ctrl_min_log + w_ctrl_per_stage * (w_ctrl_max_log - w_ctrl_min_log)
    W_ctrl = torch.pow(10., log_w_ctrl)

    # Linear scaling for y_ref
    yref_state = state_ref_min + yref_state_per_stage * (state_ref_max - state_ref_min)
    yref_ctrl = ctrl_ref_min + yref_ctrl_per_stage * (ctrl_ref_max - ctrl_ref_min)

    # Override state reference with current state (track current position)
    # This makes the MPC a tracking controller
    for stage in range(n_state_stages):
        yref_state[:, stage, :] = x0  # Track initial state

    # Override thrust reference to hover
    yref_ctrl[..., 3] = hover_thrust

    # === Map to EXTERNAL cost parameters ===
    # LINEAR_LS: J = 0.5 * ||x - y_ref||_W^2 = 0.5 * (x - y_ref)' W (x - y_ref)
    # Expanding: = 0.5 * x'Wx - x'W*y_ref + 0.5*y_ref'W*y_ref
    # EXTERNAL:  J = 0.5 * x'Qx + p'x + const
    # So: Q = W, p = -W * y_ref

    Q_state = W_state
    Q_ctrl = W_ctrl
    p_state = -W_state * yref_state
    p_ctrl = -W_ctrl * yref_ctrl

    # Flatten and concatenate (matching expected parameter order)
    q_state_flat = Q_state.reshape(batch_size, -1)
    q_ctrl_flat = Q_ctrl.reshape(batch_size, -1)
    p_state_flat = p_state.reshape(batch_size, -1)
    p_ctrl_flat = p_ctrl.reshape(batch_size, -1)

    return torch.cat([q_state_flat, q_ctrl_flat, p_state_flat, p_ctrl_flat], dim=-1)


def main():
    # Create planner with correct mass
    cfg = QuadrotorPlannerConfig(
        N_horizon=10,
        dt=0.05,
        param_interface="stagewise",
        n_batch_max=1,
        num_threads=1,
        drone_model="cf2x_T350",
        mass=0.0406,  # Actual measured mass
    )
    planner = QuadrotorPlanner(cfg=cfg)

    # Print physical params
    drone_params = planner.drone_params
    mass = float(drone_params["mass"])
    gravity = float(np.abs(drone_params["gravity_vec"][2]))
    cmd_f_coef = float(drone_params["cmd_f_coef"])
    hover_thrust = (mass * gravity) / cmd_f_coef

    print("=== Physical Parameters ===")
    print(f"mass: {mass}")
    print(f"gravity: {gravity}")
    print(f"cmd_f_coef: {cmd_f_coef}")
    print(f"hover_thrust: {hover_thrust:.4f}")

    # Get param dimensions
    param_dim = planner.get_learnable_param_dim()
    param_info = planner.get_param_structure_info()
    print(f"\n=== Parameter Info ===")
    print(f"param_dim: {param_dim}")
    print(f"n_state_stages: {param_info['n_state_stages']}")
    print(f"n_ctrl_stages: {param_info['n_ctrl_stages']}")

    # Initial state: hover at z=1m
    # State: [x, y, z, roll, pitch, yaw, vx, vy, vz, droll, dpitch, dyaw]
    x0 = torch.tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    # Create default 0.5 net_out (maps to middle of all ranges)
    batch_size = 1
    net_out = torch.full((batch_size, param_dim), 0.5)

    # Scale parameters using LINEAR_LS approach
    mpc_params = scale_parameters_linear_ls(net_out, batch_size, planner, x0)

    print(f"\n=== Scaled Parameters (LINEAR_LS style from 0.5 input) ===")
    print(f"mpc_params shape: {mpc_params.shape}")

    # Show W and y_ref values for first stage
    n_state = Q_STATE_SIZE
    n_ctrl = Q_CTRL_SIZE
    n_state_stages = param_info['n_state_stages']
    n_ctrl_stages = param_info['n_ctrl_stages']

    q_state = mpc_params[0, :n_state].numpy()
    q_ctrl = mpc_params[0, n_state*n_state_stages:n_state*n_state_stages+n_ctrl].numpy()

    print(f"\nW_state (stage 0): {q_state}")
    print(f"W_ctrl (stage 0): {q_ctrl}")

    print(f"\n=== Initial State ===")
    print(f"x0: {x0}")

    # Solve MPC
    print(f"\n=== Solving MPC ===")
    _, u0, x_traj, u_traj, value = planner(obs=x0, param=mpc_params)

    print(f"\n=== Results ===")
    print(f"u0 (first control): {u0}")
    print(f"value: {value}")

    print(f"\n=== State Trajectory ===")
    print(f"x_traj shape: {x_traj.shape}")
    for i in range(x_traj.shape[1]):
        print(f"  t={i}: pos=[{x_traj[0,i,0]:.4f}, {x_traj[0,i,1]:.4f}, {x_traj[0,i,2]:.4f}], "
              f"rpy=[{x_traj[0,i,3]:.4f}, {x_traj[0,i,4]:.4f}, {x_traj[0,i,5]:.4f}], "
              f"vel=[{x_traj[0,i,6]:.4f}, {x_traj[0,i,7]:.4f}, {x_traj[0,i,8]:.4f}]")

    print(f"\n=== Control Trajectory ===")
    print(f"u_traj shape: {u_traj.shape}")
    for i in range(u_traj.shape[1]):
        print(f"  t={i}: [roll={u_traj[0,i,0]:.4f}, pitch={u_traj[0,i,1]:.4f}, "
              f"yaw={u_traj[0,i,2]:.4f}, thrust={u_traj[0,i,3]:.4f}]")

    print(f"\n=== Expected vs Actual ===")
    print(f"Expected hover thrust: {hover_thrust:.4f}")
    print(f"Actual thrust from MPC: {u0[0, 3]:.4f}")
    print(f"Match: {abs(u0[0, 3].item() - hover_thrust) < 0.01}")


if __name__ == "__main__":
    main()
