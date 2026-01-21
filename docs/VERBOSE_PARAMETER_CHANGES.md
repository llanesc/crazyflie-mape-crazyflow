# Verbose Parameter Changes for Acados Build Output

**Date**: 2026-01-20
**Purpose**: Added `verbose` parameter to suppress acados build output during MPC solver creation.

## Files Modified

### 1. `external/leap-c/leap_c/ocp/acados/utils/create_solver.py`
- Added `verbose: bool = True` parameter to `create_batch_solver()`
- Added `verbose: bool = True` parameter to `create_forward_backward_batch_solvers()`
- Pass `verbose` to `AcadosOcpBatchSolver` constructor calls

### 2. `external/leap-c/leap_c/ocp/acados/diff_mpc.py`
- Added `verbose: bool = True` parameter to `AcadosDiffMpc.__init__()`
- Pass `verbose` to `create_forward_backward_batch_solvers()`

### 3. `external/leap-c/leap_c/ocp/acados/torch.py`
- Added `verbose: bool = True` parameter to `AcadosDiffMpcTorch.__init__()`
- Pass `verbose` to `AcadosDiffMpcFunction()` constructor

### 4. `crazyflie_mape_crazyflow/leap_c/quadrotor_planner.py`
- Added `verbose: bool = True` field to `QuadrotorPlannerConfig` dataclass
- Pass `self.cfg.verbose` to `AcadosDiffMpcTorch()` constructor

### 5. `crazyflie_mape_crazyflow/policies/leap_c_shared_policy.py`
- Added `verbose: bool = True` parameter to `LeapCMPCLayer.__init__()` (aka `MpcLayer`)
- Added `verbose: bool = True` parameter to `LeapCSharedGaussianPolicy.__init__()`
- Pass `verbose` through to `QuadrotorPlannerConfig` and `LeapCMPCLayer`

### 6. `scripts/eval_mappo_acmpc.py`
- Set `verbose=False` when creating `LeapCSharedGaussianPolicy`
- Also fixed parameter name typos: `max_roll_pitch` -> `roll_pitch_max`, `max_yaw` -> `yaw_max`

## How to Revert

To revert these changes, remove the `verbose` parameter from all the above locations and remove the `verbose=...` arguments from all constructor calls. The acados library natively supports `verbose` in `AcadosOcpBatchSolver`, so that doesn't need reverting.

Alternatively, use git:
```bash
git diff HEAD -- external/leap-c/leap_c/ocp/acados/utils/create_solver.py \
                 external/leap-c/leap_c/ocp/acados/diff_mpc.py \
                 external/leap-c/leap_c/ocp/acados/torch.py \
                 crazyflie_mape_crazyflow/leap_c/quadrotor_planner.py \
                 crazyflie_mape_crazyflow/policies/leap_c_shared_policy.py \
                 scripts/eval_mappo_acmpc.py
```

## Reference
Based on acados PR #928: https://github.com/acados/acados/pull/928
