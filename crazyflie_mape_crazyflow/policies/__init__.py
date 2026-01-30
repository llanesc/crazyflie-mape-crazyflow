"""SKRL policy implementations with LEAP-C MPC and FFN."""

from crazyflie_mape_crazyflow.policies.ffn_shared_policy import FFNSharedGaussianPolicy
from crazyflie_mape_crazyflow.policies.shared_critic import SharedCritic

# Lazy imports for LEAP-C policies (requires acados)
def __getattr__(name):
    if name in ("LeapCMPCLayer", "LeapCSharedGaussianPolicy"):
        from crazyflie_mape_crazyflow.policies.leap_c_shared_policy import (
            LeapCMPCLayer,
            LeapCSharedGaussianPolicy,
        )
        if name == "LeapCMPCLayer":
            return LeapCMPCLayer
        return LeapCSharedGaussianPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LeapCMPCLayer",
    "LeapCSharedGaussianPolicy",
    "FFNSharedGaussianPolicy",
    "SharedCritic",
]
