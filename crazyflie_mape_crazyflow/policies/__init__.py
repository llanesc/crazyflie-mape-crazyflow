"""SKRL policy implementations with LEAP-C MPC (LINEAR_LS cost) and FFN."""

from crazyflie_mape_crazyflow.policies.ffn_shared_policy import FFNSharedGaussianPolicy
from crazyflie_mape_crazyflow.policies.shared_critic import SharedCritic

# Lazy imports for LEAP-C policies (requires acados)
def __getattr__(name):
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
    "LeapCMPCLayerLinearLS",
    "LeapCSharedGaussianPolicyLinearLS",
    "FFNSharedGaussianPolicy",
    "SharedCritic",
]
