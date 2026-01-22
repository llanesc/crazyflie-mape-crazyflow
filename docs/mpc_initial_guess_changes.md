# MPC Initial Control Guess Support

This document describes changes made to support initial control guesses and warmstarting in the MPC solver.

## Date: 2026-01-22

## Files Modified

### 1. `crazyflie_mape_crazyflow/policies/leap_c_shared_policy.py`

#### MpcLayer.forward() - BEFORE:
```python
def forward(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Forward pass through MPC layer.

    Args:
        obs: Observations with shape (B, obs_dim).
        state: MPC state with shape (B, 12) [pos, rpy, vel, drpy].

    Returns:
        Normalized control action with shape (B, 4).
    """
    batch_size = obs.shape[0]

    # Get cost parameters from network
    cost_net_out = self.cost_net(obs)

    # Scale parameters
    mpc_params = self._scale_parameters(cost_net_out, batch_size)

    # Solve MPC
    ctx, u0, x_traj, u_traj, value = self.planner(
        obs=state,
        param=mpc_params,
        ctx=None,
    )

    # Normalize action to [-1, 1]
    action_normalized = (u0 - self.action_mean) / self.action_scale

    return action_normalized
```

#### MpcLayer.forward() - AFTER:
```python
def forward(
    self,
    obs: torch.Tensor,
    state: torch.Tensor,
    u0_guess: torch.Tensor | None = None,
    ctx: "AcadosDiffMpcCtx | None" = None,
) -> tuple[torch.Tensor, "AcadosDiffMpcCtx"]:
    """Forward pass through MPC layer.

    Args:
        obs: Observations with shape (B, obs_dim).
        state: MPC state with shape (B, 12) [pos, rpy, vel, drpy].
        u0_guess: Initial control guess with shape (B, 4) [roll, pitch, yaw, thrust].
            If None, solver uses its default initialization.
        ctx: Context from previous solve for warmstarting. If provided, the solver
            will use the previous solution as initial guess for faster convergence.

    Returns:
        Tuple of:
            - action_normalized: Normalized control action with shape (B, 4).
            - ctx: Context object for warmstarting subsequent solves.
    """
    batch_size = obs.shape[0]

    # Get cost parameters from network
    cost_net_out = self.cost_net(obs)

    # Scale parameters
    mpc_params = self._scale_parameters(cost_net_out, batch_size)

    # Solve MPC with optional initial guess and warmstart
    ctx, u0, x_traj, u_traj, value = self.planner(
        obs=state,
        action=u0_guess,
        param=mpc_params,
        ctx=ctx,
    )

    # Normalize action to [-1, 1]
    action_normalized = (u0 - self.action_mean) / self.action_scale

    return action_normalized, ctx
```

#### Imports - BEFORE:
```python
from typing import Mapping, Optional, Sequence, Tuple, Union
```

#### Imports - AFTER:
```python
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple, Union

# ... other imports ...

if TYPE_CHECKING:
    from leap_c.ocp.acados.diff_mpc import AcadosDiffMpcCtx
```

#### LeapCSharedGaussianPolicy.compute() - BEFORE:
```python
# Get mean action from MPC
mean_actions = self.mpc_layer(obs, state)
```

#### LeapCSharedGaussianPolicy.compute() - AFTER:
```python
# Get mean action from MPC (ignore context for now, could be used for warmstarting)
mean_actions, _ = self.mpc_layer(obs, state)
```

## How to Revert

To revert these changes:

1. Remove `TYPE_CHECKING` from imports and remove the `if TYPE_CHECKING:` block

2. Change the `forward` method signature back to:
   ```python
   def forward(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
   ```

3. Remove `u0_guess` and `ctx` parameters from the planner call:
   ```python
   ctx, u0, x_traj, u_traj, value = self.planner(
       obs=state,
       param=mpc_params,
       ctx=None,
   )
   ```

4. Change return to just the action:
   ```python
   return action_normalized
   ```

5. Update the caller in `compute()`:
   ```python
   mean_actions = self.mpc_layer(obs, state)
   ```

## Usage Example

```python
# Without warmstart (default behavior)
action, ctx = mpc_layer(obs, state)

# With warmstart from previous solve
action, ctx = mpc_layer(obs, state, ctx=prev_ctx)

# With explicit initial control guess
hover_thrust = mass * gravity
u0_guess = torch.tensor([[0, 0, 0, hover_thrust]])  # hover
action, ctx = mpc_layer(obs, state, u0_guess=u0_guess)

# With both
action, ctx = mpc_layer(obs, state, u0_guess=u0_guess, ctx=prev_ctx)
```
