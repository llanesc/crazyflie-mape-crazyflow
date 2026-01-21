"""Spawn functions for multi-agent environments.

This module provides JIT-compiled spawn functions that can be used
with functools.partial to create configurable spawn callables.
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp


SpawnFn = Callable[[jax.Array, int, int, int], tuple[jnp.ndarray, jnp.ndarray]]
"""Type alias for spawn functions.

Args:
    key: JAX random key.
    N: Number of worlds.
    B: Number of blue agents.
    R: Number of red agents.

Returns:
    Tuple of (blue_pos, red_pos) arrays of shape (N, B, 3) and (N, R, 3).
"""


TeamSpawnFn = Callable[[jax.Array, int, int], jnp.ndarray]
"""Type alias for single-team spawn functions.

Args:
    key: JAX random key.
    N: Number of worlds.
    A: Number of agents for this team.

Returns:
    Array of shape (N, A, 3) with spawn positions.
"""


# =============================================================================
# Pure (non-JIT) spawn implementations - used inside JIT'd hybrid functions
# =============================================================================

def _deterministic_spawn_impl(
    key: jax.Array,
    N: int, A: int,
    x: float,
    teammate_spacing: float,
    initial_height: float,
) -> jnp.ndarray:
    """Pure deterministic spawn implementation (no JIT decorator)."""
    x_arr = jnp.full((A,), x)
    y_arr = jnp.linspace(-(A - 1) / 2, (A - 1) / 2, A) * teammate_spacing
    z_arr = jnp.full((A,), initial_height)
    pos_single = jnp.stack([x_arr, y_arr, z_arr], axis=-1)
    return jnp.broadcast_to(pos_single, (N, A, 3))


def _box_random_spawn_impl(
    key: jax.Array,
    N: int, A: int,
    x_min: float,
    x_max: float,
    y_half: float,
    z_min: float,
    z_max: float,
) -> jnp.ndarray:
    """Pure box random spawn implementation (no JIT decorator)."""
    key, x_key, y_key, z_key = jax.random.split(key, 4)
    x = jax.random.uniform(x_key, (N, A), minval=x_min, maxval=x_max)
    y = jax.random.uniform(y_key, (N, A), minval=-y_half, maxval=y_half)
    z = jax.random.uniform(z_key, (N, A), minval=z_min, maxval=z_max)
    return jnp.stack([x, y, z], axis=-1)


def _nominal_box_spawn_impl(
    key: jax.Array,
    N: int, A: int,
    x: float,
    teammate_spacing: float,
    initial_height: float,
    x_half: float,
    y_half: float,
    z_half: float,
) -> jnp.ndarray:
    """Pure nominal box spawn implementation (no JIT decorator)."""
    # Compute nominal positions
    nominal_x = jnp.full((A,), x)
    nominal_y = jnp.linspace(-(A - 1) / 2, (A - 1) / 2, A) * teammate_spacing
    nominal_z = jnp.full((A,), initial_height)

    # Sample random offsets
    key, dx_key, dy_key, dz_key = jax.random.split(key, 4)
    dx = jax.random.uniform(dx_key, (N, A), minval=-x_half, maxval=x_half)
    dy = jax.random.uniform(dy_key, (N, A), minval=-y_half, maxval=y_half)
    dz = jax.random.uniform(dz_key, (N, A), minval=-z_half, maxval=z_half)

    # Add offsets to nominal positions
    pos_x = nominal_x[None, :] + dx
    pos_y = nominal_y[None, :] + dy
    pos_z = nominal_z[None, :] + dz

    return jnp.stack([pos_x, pos_y, pos_z], axis=-1)


def _ring_random_spawn_impl(
    key: jax.Array,
    N: int, A: int,
    radius_min: float,
    radius_max: float,
    initial_height: float,
) -> jnp.ndarray:
    """Pure ring random spawn implementation (no JIT decorator)."""
    key, theta_key, radius_key = jax.random.split(key, 3)
    theta = jax.random.uniform(theta_key, (N, A)) * 2 * jnp.pi
    radius = jax.random.uniform(radius_key, (N, A), minval=radius_min, maxval=radius_max)
    x = radius * jnp.cos(theta)
    y = radius * jnp.sin(theta)
    z = jnp.full((N, A), initial_height)
    return jnp.stack([x, y, z], axis=-1)


# =============================================================================
# JIT'd single-team spawn functions (for standalone use)
# =============================================================================

@partial(jax.jit, static_argnames=["N", "A"])
def deterministic_spawn_jit(
    key: jax.Array,
    N: int, A: int,
    x: float = 0.0,
    teammate_spacing: float = 0.5,
    initial_height: float = 1.0,
) -> jnp.ndarray:
    """JIT-compiled deterministic spawn for a single team."""
    return _deterministic_spawn_impl(key, N, A, x, teammate_spacing, initial_height)


@partial(jax.jit, static_argnames=["N", "A"])
def box_random_spawn_jit(
    key: jax.Array,
    N: int, A: int,
    x_min: float = -0.5,
    x_max: float = 0.5,
    y_half: float = 1.0,
    z_min: float = 0.8,
    z_max: float = 1.2,
) -> jnp.ndarray:
    """JIT-compiled box random spawn for a single team."""
    return _box_random_spawn_impl(key, N, A, x_min, x_max, y_half, z_min, z_max)


@partial(jax.jit, static_argnames=["N", "A"])
def nominal_box_spawn_jit(
    key: jax.Array,
    N: int, A: int,
    x: float = 0.0,
    teammate_spacing: float = 0.5,
    initial_height: float = 1.0,
    x_half: float = 0.2,
    y_half: float = 0.2,
    z_half: float = 0.2,
) -> jnp.ndarray:
    """JIT-compiled nominal box spawn for a single team."""
    return _nominal_box_spawn_impl(key, N, A, x, teammate_spacing, initial_height, x_half, y_half, z_half)


@partial(jax.jit, static_argnames=["N", "A"])
def ring_random_spawn_jit(
    key: jax.Array,
    N: int, A: int,
    radius_min: float = 2.5,
    radius_max: float = 6.0,
    initial_height: float = 1.0,
) -> jnp.ndarray:
    """JIT-compiled ring random spawn for a single team."""
    return _ring_random_spawn_impl(key, N, A, radius_min, radius_max, initial_height)


# =============================================================================
# Factory functions for single-team spawners
# =============================================================================

def create_deterministic_spawn_fn(
    x: float = 0.0,
    teammate_spacing: float = 0.5,
    initial_height: float = 1.0,
) -> TeamSpawnFn:
    """Create a deterministic spawn function."""
    return partial(
        deterministic_spawn_jit,
        x=x,
        teammate_spacing=teammate_spacing,
        initial_height=initial_height,
    )


def create_box_random_spawn_fn(
    x_min: float = -0.5,
    x_max: float = 0.5,
    y_half: float = 1.0,
    z_min: float = 0.8,
    z_max: float = 1.2,
) -> TeamSpawnFn:
    """Create a box random spawn function."""
    return partial(
        box_random_spawn_jit,
        x_min=x_min,
        x_max=x_max,
        y_half=y_half,
        z_min=z_min,
        z_max=z_max,
    )


def create_nominal_box_spawn_fn(
    x: float = 0.0,
    teammate_spacing: float = 0.5,
    initial_height: float = 1.0,
    x_half: float = 0.2,
    y_half: float = 0.2,
    z_half: float = 0.2,
) -> TeamSpawnFn:
    """Create a nominal box spawn function."""
    return partial(
        nominal_box_spawn_jit,
        x=x,
        teammate_spacing=teammate_spacing,
        initial_height=initial_height,
        x_half=x_half,
        y_half=y_half,
        z_half=z_half,
    )


def create_ring_random_spawn_fn(
    radius_min: float = 2.5,
    radius_max: float = 6.0,
    initial_height: float = 1.0,
) -> TeamSpawnFn:
    """Create a ring random spawn function."""
    return partial(
        ring_random_spawn_jit,
        radius_min=radius_min,
        radius_max=radius_max,
        initial_height=initial_height,
    )


# =============================================================================
# Combined spawn function creation (JIT'd with all params baked in)
# =============================================================================

def _create_team_spawn_fn_from_config(
    team_config: dict,
    default_x: float = 0.0,
    default_x_min: float = -0.5,
    default_x_max: float = 0.5,
) -> tuple[str, dict]:
    """Parse team config and return method name + params dict.

    Returns:
        Tuple of (method_name, params_dict) for use in combined spawn function.
    """
    method = team_config.get("method", "deterministic")

    if method == "deterministic":
        params = {
            "x": team_config.get("x", default_x),
            "teammate_spacing": team_config.get("teammate_spacing", 0.5),
            "initial_height": team_config.get("initial_height", 1.0),
        }
    elif method == "box_random":
        params = {
            "x_min": team_config.get("x_min", default_x_min),
            "x_max": team_config.get("x_max", default_x_max),
            "y_half": team_config.get("y_half", 1.0),
            "z_min": team_config.get("z_min", 0.8),
            "z_max": team_config.get("z_max", 1.2),
        }
    elif method == "nominal_box":
        params = {
            "x": team_config.get("x", default_x),
            "teammate_spacing": team_config.get("teammate_spacing", 0.5),
            "initial_height": team_config.get("initial_height", 1.0),
            "x_half": team_config.get("x_half", 0.2),
            "y_half": team_config.get("y_half", 0.2),
            "z_half": team_config.get("z_half", 0.2),
        }
    elif method == "ring_random":
        params = {
            "radius_min": team_config.get("radius_min", 2.5),
            "radius_max": team_config.get("radius_max", 6.0),
            "initial_height": team_config.get("initial_height", 1.0),
        }
    else:
        raise ValueError(f"Unknown spawn method: {method}")

    return method, params


def create_spawn_fn_from_config(spawn_config: dict) -> SpawnFn:
    """Create a spawn function from a configuration dictionary.

    Creates a single JIT'd function with all spawn parameters baked in as literals.
    This ensures JAX traces the function only once.

    Args:
        spawn_config: Dictionary containing spawn configuration.
            Must have nested "blue" and "red" sub-dicts specifying each team's spawn.
            Supported methods: "deterministic", "box_random", "nominal_box", "ring_random".

    Returns:
        Spawn function with signature (key, N, B, R) -> (blue_pos, red_pos).
    """
    blue_config = spawn_config.get("blue", {"method": "deterministic", "x": 2.0})
    red_config = spawn_config.get("red", {"method": "deterministic", "x": 0.0})

    blue_method, blue_params = _create_team_spawn_fn_from_config(
        blue_config, default_x=2.0, default_x_min=1.5, default_x_max=2.5
    )
    red_method, red_params = _create_team_spawn_fn_from_config(
        red_config, default_x=0.0, default_x_min=-0.5, default_x_max=0.5
    )

    # Extract parameters as explicit variables to avoid dict unpacking in JIT
    # Blue params
    if blue_method == "deterministic":
        b_x = blue_params["x"]
        b_ts = blue_params["teammate_spacing"]
        b_ih = blue_params["initial_height"]
    elif blue_method == "box_random":
        b_xmin = blue_params["x_min"]
        b_xmax = blue_params["x_max"]
        b_yhalf = blue_params["y_half"]
        b_zmin = blue_params["z_min"]
        b_zmax = blue_params["z_max"]
    elif blue_method == "nominal_box":
        b_x = blue_params["x"]
        b_ts = blue_params["teammate_spacing"]
        b_ih = blue_params["initial_height"]
        b_xhalf = blue_params["x_half"]
        b_yhalf = blue_params["y_half"]
        b_zhalf = blue_params["z_half"]
    elif blue_method == "ring_random":
        b_rmin = blue_params["radius_min"]
        b_rmax = blue_params["radius_max"]
        b_ih = blue_params["initial_height"]

    # Red params
    if red_method == "deterministic":
        r_x = red_params["x"]
        r_ts = red_params["teammate_spacing"]
        r_ih = red_params["initial_height"]
    elif red_method == "box_random":
        r_xmin = red_params["x_min"]
        r_xmax = red_params["x_max"]
        r_yhalf = red_params["y_half"]
        r_zmin = red_params["z_min"]
        r_zmax = red_params["z_max"]
    elif red_method == "nominal_box":
        r_x = red_params["x"]
        r_ts = red_params["teammate_spacing"]
        r_ih = red_params["initial_height"]
        r_xhalf = red_params["x_half"]
        r_yhalf = red_params["y_half"]
        r_zhalf = red_params["z_half"]
    elif red_method == "ring_random":
        r_rmin = red_params["radius_min"]
        r_rmax = red_params["radius_max"]
        r_ih = red_params["initial_height"]

    # Create specialized JIT function based on method combination
    # All parameters are captured as closure variables (Python floats -> JAX constants)

    if blue_method == "deterministic" and red_method == "deterministic":
        @partial(jax.jit, static_argnames=["N", "B", "R"])
        def spawn_fn(key: jax.Array, N: int, B: int, R: int):
            key, bk, rk = jax.random.split(key, 3)
            blue_pos = _deterministic_spawn_impl(bk, N, B, b_x, b_ts, b_ih)
            red_pos = _deterministic_spawn_impl(rk, N, R, r_x, r_ts, r_ih)
            return blue_pos, red_pos

    elif blue_method == "deterministic" and red_method == "box_random":
        @partial(jax.jit, static_argnames=["N", "B", "R"])
        def spawn_fn(key: jax.Array, N: int, B: int, R: int):
            key, bk, rk = jax.random.split(key, 3)
            blue_pos = _deterministic_spawn_impl(bk, N, B, b_x, b_ts, b_ih)
            red_pos = _box_random_spawn_impl(rk, N, R, r_xmin, r_xmax, r_yhalf, r_zmin, r_zmax)
            return blue_pos, red_pos

    elif blue_method == "box_random" and red_method == "deterministic":
        @partial(jax.jit, static_argnames=["N", "B", "R"])
        def spawn_fn(key: jax.Array, N: int, B: int, R: int):
            key, bk, rk = jax.random.split(key, 3)
            blue_pos = _box_random_spawn_impl(bk, N, B, b_xmin, b_xmax, b_yhalf, b_zmin, b_zmax)
            red_pos = _deterministic_spawn_impl(rk, N, R, r_x, r_ts, r_ih)
            return blue_pos, red_pos

    elif blue_method == "box_random" and red_method == "box_random":
        @partial(jax.jit, static_argnames=["N", "B", "R"])
        def spawn_fn(key: jax.Array, N: int, B: int, R: int):
            key, bk, rk = jax.random.split(key, 3)
            blue_pos = _box_random_spawn_impl(bk, N, B, b_xmin, b_xmax, b_yhalf, b_zmin, b_zmax)
            red_pos = _box_random_spawn_impl(rk, N, R, r_xmin, r_xmax, r_yhalf, r_zmin, r_zmax)
            return blue_pos, red_pos

    elif blue_method == "nominal_box" and red_method == "nominal_box":
        @partial(jax.jit, static_argnames=["N", "B", "R"])
        def spawn_fn(key: jax.Array, N: int, B: int, R: int):
            key, bk, rk = jax.random.split(key, 3)
            blue_pos = _nominal_box_spawn_impl(bk, N, B, b_x, b_ts, b_ih, b_xhalf, b_yhalf, b_zhalf)
            red_pos = _nominal_box_spawn_impl(rk, N, R, r_x, r_ts, r_ih, r_xhalf, r_yhalf, r_zhalf)
            return blue_pos, red_pos

    elif blue_method == "deterministic" and red_method == "nominal_box":
        @partial(jax.jit, static_argnames=["N", "B", "R"])
        def spawn_fn(key: jax.Array, N: int, B: int, R: int):
            key, bk, rk = jax.random.split(key, 3)
            blue_pos = _deterministic_spawn_impl(bk, N, B, b_x, b_ts, b_ih)
            red_pos = _nominal_box_spawn_impl(rk, N, R, r_x, r_ts, r_ih, r_xhalf, r_yhalf, r_zhalf)
            return blue_pos, red_pos

    elif blue_method == "nominal_box" and red_method == "deterministic":
        @partial(jax.jit, static_argnames=["N", "B", "R"])
        def spawn_fn(key: jax.Array, N: int, B: int, R: int):
            key, bk, rk = jax.random.split(key, 3)
            blue_pos = _nominal_box_spawn_impl(bk, N, B, b_x, b_ts, b_ih, b_xhalf, b_yhalf, b_zhalf)
            red_pos = _deterministic_spawn_impl(rk, N, R, r_x, r_ts, r_ih)
            return blue_pos, red_pos

    elif blue_method == "box_random" and red_method == "nominal_box":
        @partial(jax.jit, static_argnames=["N", "B", "R"])
        def spawn_fn(key: jax.Array, N: int, B: int, R: int):
            key, bk, rk = jax.random.split(key, 3)
            blue_pos = _box_random_spawn_impl(bk, N, B, b_xmin, b_xmax, b_yhalf, b_zmin, b_zmax)
            red_pos = _nominal_box_spawn_impl(rk, N, R, r_x, r_ts, r_ih, r_xhalf, r_yhalf, r_zhalf)
            return blue_pos, red_pos

    elif blue_method == "nominal_box" and red_method == "box_random":
        @partial(jax.jit, static_argnames=["N", "B", "R"])
        def spawn_fn(key: jax.Array, N: int, B: int, R: int):
            key, bk, rk = jax.random.split(key, 3)
            blue_pos = _nominal_box_spawn_impl(bk, N, B, b_x, b_ts, b_ih, b_xhalf, b_yhalf, b_zhalf)
            red_pos = _box_random_spawn_impl(rk, N, R, r_xmin, r_xmax, r_yhalf, r_zmin, r_zmax)
            return blue_pos, red_pos

    else:
        raise ValueError(f"Unsupported spawn method combination: blue={blue_method}, red={red_method}")

    return spawn_fn
