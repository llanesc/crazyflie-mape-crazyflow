"""LEAP-C MPC integration with so_rpy dynamics (quaternion state, 13D).

Supports two cost formulations:
- QP: J = 0.5 * x'Qx + p'x (default, backward compatible)
- LINEAR_LS: J = 0.5 * ||Vx*x + Vu*u - y_ref||_W^2 (decoupled W and y_ref)
"""

# QP cost (default, backward compatible)
from crazyflie_mape_crazyflow.leap_c.quadrotor_ocp_qp import (
    NX,
    NU,
    Q_STATE_SIZE,
    Q_CTRL_SIZE,
    P_X_SIZE,
    P_U_SIZE,
    create_quadrotor_params_qp,
    export_parametric_ocp_qp,
    get_learnable_param_dim_qp,
)

# LINEAR_LS cost (alternative)
from crazyflie_mape_crazyflow.leap_c.quadrotor_ocp_linear_ls import (
    create_quadrotor_params_linear_ls,
    export_parametric_ocp_linear_ls,
    get_learnable_param_dim_linear_ls,
    # Euler (12D) variants for backward compatibility
    create_quadrotor_params_linear_ls_euler,
    export_parametric_ocp_linear_ls_euler,
    get_learnable_param_dim_linear_ls_euler,
    NX_EULER,
)

from crazyflie_mape_crazyflow.leap_c.quadrotor_planner import (
    QuadrotorPlanner,
    QuadrotorPlannerConfig,
)

__all__ = [
    # Constants
    "NX",
    "NU",
    "Q_STATE_SIZE",
    "Q_CTRL_SIZE",
    "P_X_SIZE",
    "P_U_SIZE",
    # QP cost (default)
    "create_quadrotor_params_qp",
    "export_parametric_ocp_qp",
    "get_learnable_param_dim_qp",
    # LINEAR_LS cost
    "create_quadrotor_params_linear_ls",
    "export_parametric_ocp_linear_ls",
    "get_learnable_param_dim_linear_ls",
    # LINEAR_LS Euler (12D) variants
    "create_quadrotor_params_linear_ls_euler",
    "export_parametric_ocp_linear_ls_euler",
    "get_learnable_param_dim_linear_ls_euler",
    "NX_EULER",
    # Planner
    "QuadrotorPlanner",
    "QuadrotorPlannerConfig",
]
