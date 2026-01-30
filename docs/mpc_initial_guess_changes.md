# MPC Initial Control Guess - CRITICAL BUG DOCUMENTATION

## Date: 2026-01-22

## Critical Bug: DO NOT Pass `action`/`u0` to the Planner

### The Bug

When passing `action` (u0_guess) to the leap-c planner, the framework **incorrectly treats it as a hard constraint** instead of a warm-start initialization.

**Location:** `external/leap-c/leap_c/ocp/acados/utils/prepare_solver.py` lines 84-87:

```python
if u0 is not None:
    solver.set(0, "u", u0[idx])
    solver.constraints_set(0, "lbu", u0[idx])  # BUG: Forces lower bound = u0
    solver.constraints_set(0, "ubu", u0[idx])  # BUG: Forces upper bound = u0
```

### Impact

When you pass `action=u0_guess` to the planner:
- The first control input is **forced** to be exactly `u0_guess`
- The MPC solver **cannot** compute optimal controls that deviate from this value
- This defeats the purpose of optimization - the first control is locked

### Correct Usage

**DO NOT** pass `action` to the planner. Let the solver use its default initialization:

```python
# WRONG - causes hard constraint bug
_, u0, x_traj, u_traj, value = self.planner(
    obs=state,
    action=u0_guess,  # DO NOT DO THIS
    param=mpc_params,
)

# CORRECT - no action parameter
_, u0, x_traj, u_traj, value = self.planner(
    obs=state,
    param=mpc_params,
)
```

### Context/Warmstart via `ctx` Parameter

The `ctx` parameter for warmstarting **does work correctly**. When passing a context from a previous solve:
- The solver uses the previous solution as initialization
- The constraints remain as defined in the OCP
- The solver can compute optimal controls

```python
# Warmstart with context - this is correct
ctx, u0, x_traj, u_traj, value = self.planner(
    obs=state,
    param=mpc_params,
    ctx=prev_ctx,  # This is fine - uses previous solution as warmstart
)
```

---

## Correct Approach: AcadosDiffMpcInitializer

The proper way to provide initial guesses is through `AcadosDiffMpcInitializer`, which is
passed to `AcadosDiffMpcTorch.__init__`. This initializes the solver iterate without
modifying constraints.

### `crazyflie_mape_crazyflow/leap_c/quadrotor_planner.py`

A `QuadrotorHoverInitializer` is implemented that:
- Sets state trajectory to `x0` broadcast across all `(N+1)` stages
- Sets control trajectory to hover `[0, 0, 0, mass*gravity]` across all `N` stages
- Dual variables (`pi`, `lam`, `sl`, `su`) remain zero (from default iterate)

```python
class QuadrotorHoverInitializer(AcadosDiffMpcInitializer):
    def __init__(self, ocp: AcadosOcp, mass: float, gravity: float):
        self.default_iterate = ocp.create_default_initial_iterate().flatten()
        self.N = ocp.solver_options.N_horizon
        self.nx = ocp.dims.nx
        self.nu = ocp.dims.nu
        self.hover_thrust = mass * gravity
        self.hover_u = np.zeros(self.nu)
        self.hover_u[-1] = self.hover_thrust

    def single_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedIterate:
        iterate = deepcopy(self.default_iterate)
        x0 = solver_input.x0.flatten()
        iterate.x = np.tile(x0, self.N + 1)
        iterate.u = np.tile(self.hover_u, self.N)
        return iterate
```

The initializer is passed to `AcadosDiffMpcTorch` in `QuadrotorPlanner.__init__`:

```python
initializer = QuadrotorHoverInitializer(ocp, mass=mass, gravity=gravity)
diff_mpc = AcadosDiffMpcTorch(
    ocp,
    initializer=initializer,
    ...
)
```

### `crazyflie_mape_crazyflow/policies/leap_c_shared_policy.py`

The policy simply calls the planner without `action` - initialization is handled
by the initializer:

```python
def forward(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    batch_size = obs.shape[0]
    cost_net_out = self.cost_net(obs)
    mpc_params = self._scale_parameters(cost_net_out, batch_size)

    # Solve MPC (initialization handled by QuadrotorHoverInitializer)
    _, u0, x_traj, u_traj, value = self.planner(
        obs=state,
        param=mpc_params,
    )

    action_normalized = (u0 - self.action_mean) / self.action_scale
    return action_normalized
```

### `tests/mappo_leapc_ctx.py`

Reference implementation showing how to add MPC context management to SKRL's MAPPO:
- Uses `ctx` parameter for warmstarting (correct)
- Passes context via `inputs["mpc_ctx"]` / `outputs["mpc_ctx"]`
- Resets context for terminated/truncated episodes
- Does NOT use `action` parameter

---

## How `AcadosDiffMpcInitializer` Works in the Solve Loop

In `solve_with_retry()`:

1. **First solve (ctx=None):** `initializer.batch_iterate(solver_input)` generates batch of
   initial guesses. The `single_iterate()` method is called per sample in the batch.

2. **Subsequent solves (ctx provided):** Previous solution from `ctx.iterate` is used as
   warmstart. The initializer is only used as fallback for failed solvers.

3. **Retry on failure:** If any solver fails with warmstart, `initializer.single_iterate()`
   is called for each failed solver to reset them, then a second solve attempt is made.

---

## Context Warmstarting (for reference)

The reference implementation in `tests/mappo_leapc_ctx.py` shows how per-agent MPC context
management could be added to SKRL's MAPPO. Context warmstarting (`ctx` parameter) works correctly
and can be combined with the initializer (initializer is used on first call, context on subsequent).
