"""SKRL policy implementations with LEAP-C MPC and FFN.

Supports two LEAP-C MPC cost formulations:
- QP: J = 0.5 * x'Qx + p'x (default, backward compatible)
- LINEAR_LS: J = 0.5 * ||Vx*x + Vu*u - y_ref||_W^2 (decoupled W and y_ref)
"""

from crazyflie_mape_crazyflow.policies.ffn_shared_policy import FFNSharedGaussianPolicy
from crazyflie_mape_crazyflow.policies.shared_critic import SharedCritic

# Lazy imports for LEAP-C policies (requires acados)
def __getattr__(name):
    # QP cost policies (default, backward compatible)
    if name in ("LeapCMPCLayerQP", "LeapCSharedGaussianPolicyQP"):
        from crazyflie_mape_crazyflow.policies.leap_c_shared_policy_qp import (
            LeapCMPCLayerQP,
            LeapCSharedGaussianPolicyQP,
        )
        if name == "LeapCMPCLayerQP":
            return LeapCMPCLayerQP
        return LeapCSharedGaussianPolicyQP

    # LINEAR_LS cost policies
    if name in ("LeapCMPCLayerLinearLS", "LeapCSharedGaussianPolicyLinearLS"):
        from crazyflie_mape_crazyflow.policies.leap_c_shared_policy_linear_ls import (
            LeapCMPCLayerLinearLS,
            LeapCSharedGaussianPolicyLinearLS,
        )
        if name == "LeapCMPCLayerLinearLS":
            return LeapCMPCLayerLinearLS
        return LeapCSharedGaussianPolicyLinearLS

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # QP cost policies (default)
    "LeapCMPCLayerQP",
    "LeapCSharedGaussianPolicyQP",
    # LINEAR_LS cost policies
    "LeapCMPCLayerLinearLS",
    "LeapCSharedGaussianPolicyLinearLS",
    # Common
    "FFNSharedGaussianPolicy",
    "SharedCritic",
]
