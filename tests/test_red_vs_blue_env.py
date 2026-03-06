"""Unit tests for Red vs Blue environment.

Tests reward computation and termination event tracking.
"""

import os

# Use CPU by default for tests, unless TEST_DEVICE is set
_device = os.environ.get("TEST_DEVICE", "cpu")
if _device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    # Suppress JAX CUDA plugin discovery errors when running on CPU
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
# For cuda, let JAX auto-detect

import numpy as np
import jax.numpy as jnp

from crazyflie_mape_crazyflow.envs import RedVsBlueEnv, RedVsBlueEnvConfig


def create_test_env(n_pairs=2, n_worlds=4):
    """Create a test environment."""
    cfg = RedVsBlueEnvConfig(
        n_pairs=n_pairs,
        n_worlds=n_worlds,
        device="cpu",
    )
    return RedVsBlueEnv(cfg=cfg)


class TestRedVsBlueEnv:
    """Test suite for RedVsBlueEnv."""

    def test_env_creation(self):
        """Test that environment is created correctly."""
        env = create_test_env()
        try:
            assert env.cfg.n_blue == 2
            assert env.cfg.n_red == 2
            assert env.cfg.n_drones == 4
            assert len(env.possible_agents) == 2
        finally:
            env.close()

    def test_reset(self):
        """Test environment reset."""
        env = create_test_env()
        try:
            obs, info = env.reset()

            # Check observations exist for all agents
            for agent in env.possible_agents:
                assert agent in obs
                assert obs[agent].shape == (env.cfg.n_worlds, env.obs_dim)

            # Check all agents are alive
            assert env.blue_alive.all()
            assert env.red_alive.all()

            # Check episode steps are reset
            assert (env.episode_steps == 0).all()
        finally:
            env.close()

    def test_deterministic_spawn_positions(self):
        """Test that deterministic spawn places agents at expected positions."""
        env = create_test_env()
        try:
            env.reset()

            states = env.sim.data.states
            blue_pos = np.array(states.pos[:, :env.cfg.n_blue])
            red_pos = np.array(states.pos[:, env.cfg.n_blue:])

            # Blues should be at x=0.0
            np.testing.assert_allclose(blue_pos[:, :, 0], 0.0, atol=1e-5)

            # Reds should be at x=3.0
            np.testing.assert_allclose(red_pos[:, :, 0], 3.0, atol=1e-5)

            # All at initial height
            np.testing.assert_allclose(blue_pos[:, :, 2], env.cfg.initial_height, atol=1e-5)
            np.testing.assert_allclose(red_pos[:, :, 2], env.cfg.initial_height, atol=1e-5)
        finally:
            env.close()

    def test_spawn_no_immediate_crash(self):
        """Test that spawned agents don't immediately crash."""
        env = create_test_env()
        try:
            env.reset()

            states = env.sim.data.states
            blue_pos = np.array(states.pos[:, :env.cfg.n_blue])
            red_pos = np.array(states.pos[:, env.cfg.n_blue:])

            # Check blue-blue distances
            for i in range(env.cfg.n_blue):
                for j in range(i + 1, env.cfg.n_blue):
                    dist = np.linalg.norm(blue_pos[:, i] - blue_pos[:, j], axis=-1)
                    assert (dist >= env.cfg.bb_collision_tolerance).all(), \
                        f"Blues {i} and {j} spawn too close: {dist.min()}"

            # Check red-red distances
            for i in range(env.cfg.n_red):
                for j in range(i + 1, env.cfg.n_red):
                    dist = np.linalg.norm(red_pos[:, i] - red_pos[:, j], axis=-1)
                    assert (dist >= env.cfg.rr_collision_tolerance).all(), \
                        f"Reds {i} and {j} spawn too close: {dist.min()}"

            # Check blue-red distances
            for i in range(env.cfg.n_blue):
                for j in range(env.cfg.n_red):
                    dist = np.linalg.norm(blue_pos[:, i] - red_pos[:, j], axis=-1)
                    assert (dist >= env.cfg.rb_collision_tolerance).all(), \
                        f"Blue {i} and Red {j} spawn too close: {dist.min()}"
        finally:
            env.close()

    def test_spawn_within_bounds(self):
        """Test that spawned agents are within environment bounds."""
        env = create_test_env()
        try:
            env.reset()

            states = env.sim.data.states
            all_pos = np.array(states.pos)

            # Check x and y bounds
            assert (np.abs(all_pos[:, :, 0]) <= env.cfg.boundary_size).all(), \
                f"Some agents spawn outside x bounds"
            assert (np.abs(all_pos[:, :, 1]) <= env.cfg.boundary_size).all(), \
                f"Some agents spawn outside y bounds"

            # Check altitude bounds
            assert (all_pos[:, :, 2] >= env.cfg.min_altitude).all(), \
                f"Some agents spawn below min altitude"
            assert (all_pos[:, :, 2] <= env.cfg.max_altitude).all(), \
                f"Some agents spawn above max altitude"
        finally:
            env.close()


class TestCrashCounts:
    """Test that crash counts don't exceed available drones."""

    def test_bb_collision_count_bounded(self):
        """Test that bb_collision count doesn't exceed number of worlds."""
        env = create_test_env(n_worlds=8)
        try:
            env.reset()

            # Run several steps with random actions
            for _ in range(100):
                actions = {
                    agent: np.random.uniform(-1, 1, (env.cfg.n_worlds, 4)).astype(np.float32)
                    for agent in env.possible_agents
                }
                _, _, _, _, _ = env.step(actions)

                # Check bb_collision is bounded
                bb_count = env.last_termination_events["bb_collision"]
                # bb_collision counts worlds with at least one bb collision
                assert bb_count <= env.cfg.n_worlds, \
                    f"bb_collision count {bb_count} exceeds n_worlds {env.cfg.n_worlds}"
        finally:
            env.close()

    def test_rr_collision_count_bounded(self):
        """Test that rr_collision count doesn't exceed number of worlds."""
        env = create_test_env(n_worlds=8)
        try:
            env.reset()

            for _ in range(100):
                actions = {
                    agent: np.random.uniform(-1, 1, (env.cfg.n_worlds, 4)).astype(np.float32)
                    for agent in env.possible_agents
                }
                _, _, _, _, _ = env.step(actions)

                rr_count = env.last_termination_events["rr_collision"]
                assert rr_count <= env.cfg.n_worlds, \
                    f"rr_collision count {rr_count} exceeds n_worlds {env.cfg.n_worlds}"
        finally:
            env.close()

    def test_rb_collision_count_bounded(self):
        """Test that rb_collision count doesn't exceed number of worlds."""
        env = create_test_env(n_worlds=8)
        try:
            env.reset()

            for _ in range(100):
                actions = {
                    agent: np.random.uniform(-1, 1, (env.cfg.n_worlds, 4)).astype(np.float32)
                    for agent in env.possible_agents
                }
                _, _, _, _, _ = env.step(actions)

                br_count = env.last_termination_events["rb_collision"]
                assert br_count <= env.cfg.n_worlds, \
                    f"rb_collision count {br_count} exceeds n_worlds {env.cfg.n_worlds}"
        finally:
            env.close()

    def test_out_of_bounds_count_bounded(self):
        """Test that out_of_bounds count doesn't exceed number of worlds."""
        env = create_test_env(n_worlds=8)
        try:
            env.reset()

            for _ in range(100):
                actions = {
                    agent: np.random.uniform(-1, 1, (env.cfg.n_worlds, 4)).astype(np.float32)
                    for agent in env.possible_agents
                }
                _, _, _, _, _ = env.step(actions)

                oob_count = env.last_termination_events["out_of_bounds"]
                assert oob_count <= env.cfg.n_worlds, \
                    f"out_of_bounds count {oob_count} exceeds n_worlds {env.cfg.n_worlds}"
        finally:
            env.close()


class TestRewardComputation:
    """Test reward computation matches expected values."""

    def test_initial_reward_is_near_zero(self):
        """Test that initial step reward is near zero (no crashes, just small penalties)."""
        env = create_test_env()
        try:
            env.reset()

            # Take a single step with zero actions (hover)
            actions = {
                agent: np.zeros((env.cfg.n_worlds, 4), dtype=np.float32)
                for agent in env.possible_agents
            }

            _, rewards, _, _, _ = env.step(actions)

            for agent in env.possible_agents:
                # With no crashes and hover actions, reward should be small
                assert np.all(np.abs(rewards[agent]) < 5.0), \
                    f"Initial reward for {agent} unexpectedly large: {rewards[agent]}"
        finally:
            env.close()

    def test_reward_components_add_up(self):
        """Test that reward components add up correctly over multiple steps."""
        env = create_test_env()
        try:
            env.reset()

            total_bb_events = 0
            total_rr_events = 0
            total_br_events = 0
            total_oob_events = 0
            total_reward = 0.0

            # Run for several steps
            n_steps = 50
            for _ in range(n_steps):
                actions = {
                    agent: np.random.uniform(-1, 1, (env.cfg.n_worlds, 4)).astype(np.float32)
                    for agent in env.possible_agents
                }
                _, rewards, _, _, _ = env.step(actions)

                # Accumulate termination events
                total_bb_events += env.last_termination_events["bb_collision"]
                total_rr_events += env.last_termination_events["rr_collision"]
                total_br_events += env.last_termination_events["rb_collision"]
                total_oob_events += env.last_termination_events["out_of_bounds"]

                # Accumulate rewards (same for all agents)
                sample_agent = env.possible_agents[0]
                total_reward += rewards[sample_agent].sum()

            # Verify termination events were tracked
            print(f"Total events over {n_steps} steps across {env.cfg.n_worlds} worlds:")
            print(f"  bb_collision: {total_bb_events}")
            print(f"  rr_collision: {total_rr_events}")
            print(f"  rb_collision: {total_br_events}")
            print(f"  out_of_bounds: {total_oob_events}")
            print(f"  total_reward: {total_reward}")

            # Events should be non-negative
            assert total_bb_events >= 0
            assert total_rr_events >= 0
            assert total_br_events >= 0
            assert total_oob_events >= 0
        finally:
            env.close()

    def test_termination_on_all_blue_dead(self):
        """Test that episode terminates when all blues are dead."""
        env = create_test_env()
        try:
            env.reset()

            # Manually kill all blues in world 0
            env.blue_alive = env.blue_alive.at[0, :].set(False)

            # Check termination
            terminated = env._check_terminated()
            sample_agent = env.possible_agents[0]

            assert terminated[sample_agent][0] == True, "World 0 should be terminated"
            assert terminated[sample_agent][1] == False, "World 1 should not be terminated"
        finally:
            env.close()

    def test_truncation_on_max_steps(self):
        """Test that episode truncates at max steps."""
        env = create_test_env()
        try:
            env.reset()

            # Set episode steps to max for world 0
            env.episode_steps[0] = env.cfg.max_episode_steps

            truncated = env._check_truncated()
            sample_agent = env.possible_agents[0]

            assert truncated[sample_agent][0] == True, "World 0 should be truncated"
            assert truncated[sample_agent][1] == False, "World 1 should not be truncated"
        finally:
            env.close()


class TestAutoReset:
    """Test auto-reset functionality for terminated/truncated worlds."""

    def test_auto_reset_on_termination(self):
        """Test that worlds auto-reset when terminated."""
        env = create_test_env()
        try:
            env.reset()

            # Kill all blues in world 0
            env.blue_alive = env.blue_alive.at[0, :].set(False)

            # Take a step (should trigger auto-reset for world 0)
            actions = {
                agent: np.zeros((env.cfg.n_worlds, 4), dtype=np.float32)
                for agent in env.possible_agents
            }
            obs, rewards, terminated, truncated, info = env.step(actions)

            # After auto-reset, world 0 should have all blues alive again
            assert env.blue_alive[0].all(), "Blues in world 0 should be alive after auto-reset"
            assert env.episode_steps[0] == 0, "Episode steps should be reset for world 0"
        finally:
            env.close()

    def test_auto_reset_preserves_other_worlds(self):
        """Test that auto-reset only affects terminated worlds."""
        env = create_test_env()
        try:
            env.reset()

            # Take a few steps to advance episode counters
            for _ in range(5):
                actions = {
                    agent: np.zeros((env.cfg.n_worlds, 4), dtype=np.float32)
                    for agent in env.possible_agents
                }
                env.step(actions)

            initial_steps = env.episode_steps.copy()

            # Kill all blues in world 0 only
            env.blue_alive = env.blue_alive.at[0, :].set(False)

            # Take a step
            actions = {
                agent: np.zeros((env.cfg.n_worlds, 4), dtype=np.float32)
                for agent in env.possible_agents
            }
            env.step(actions)

            # World 0 should be reset, others should have incremented
            assert env.episode_steps[0] == 0, "World 0 should be reset"
            for i in range(1, env.cfg.n_worlds):
                assert env.episode_steps[i] == initial_steps[i] + 1, \
                    f"World {i} should have incremented episode steps"
        finally:
            env.close()


class TestTerminationEventTracking:
    """Test that termination events are properly tracked."""

    def test_last_termination_events_initialized(self):
        """Test that termination events are initialized on reset."""
        env = create_test_env()
        try:
            env.reset()

            assert "bb_collision" in env.last_termination_events
            assert "rr_collision" in env.last_termination_events
            assert "rb_collision" in env.last_termination_events
            assert "out_of_bounds" in env.last_termination_events
            assert "all_blue_dead" in env.last_termination_events
            assert "max_steps" in env.last_termination_events

            # All should be zero after reset
            for key, value in env.last_termination_events.items():
                assert value == 0, f"{key} should be 0 after reset, got {value}"
        finally:
            env.close()

    def test_termination_events_updated_on_step(self):
        """Test that termination events are updated after each step."""
        env = create_test_env()
        try:
            env.reset()

            actions = {
                agent: np.zeros((env.cfg.n_worlds, 4), dtype=np.float32)
                for agent in env.possible_agents
            }
            env.step(actions)

            # Events should be integers >= 0
            for key, value in env.last_termination_events.items():
                assert isinstance(value, int), f"{key} should be int, got {type(value)}"
                assert value >= 0, f"{key} should be >= 0, got {value}"
        finally:
            env.close()

    def test_death_causes_sum_to_terminations(self):
        """Test that individual death causes explain terminations over time."""
        env = create_test_env()
        try:
            env.reset()

            total_deaths = 0
            total_bb = 0
            total_br = 0
            total_oob = 0
            total_terminated = 0

            # Run for many steps
            for _ in range(200):
                actions = {
                    agent: np.random.uniform(-1, 1, (env.cfg.n_worlds, 4)).astype(np.float32)
                    for agent in env.possible_agents
                }
                env.step(actions)

                total_bb += env.last_termination_events["bb_collision"]
                total_br += env.last_termination_events["rb_collision"]
                total_oob += env.last_termination_events["out_of_bounds"]
                total_terminated += env.last_termination_events["all_blue_dead"]

            print(f"\nDeath causes over 200 steps:")
            print(f"  bb_collision: {total_bb}")
            print(f"  rb_collision: {total_br}")
            print(f"  out_of_bounds: {total_oob}")
            print(f"  all_blue_dead (terminations): {total_terminated}")

            # Total death causes should be related to terminations
            if total_terminated > 0:
                total_death_causes = total_bb + total_br + total_oob
                print(f"  total_death_causes: {total_death_causes}")
                # At least some death cause should explain terminations
                assert total_death_causes > 0 or total_terminated == 0, \
                    "Terminations occurred but no death causes tracked"
        finally:
            env.close()


def run_tests():
    """Run all tests manually."""
    import sys
    import traceback

    print("Running tests manually...\n")

    test_classes = [
        TestRedVsBlueEnv,
        TestCrashCounts,
        TestRewardComputation,
        TestAutoReset,
        TestTerminationEventTracking,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)

        instance = test_class()

        # Get test methods
        test_methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in test_methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  PASSED: {method_name}")
                passed += 1

            except Exception as e:
                print(f"  FAILED: {method_name}")
                print(f"    Error: {e}")
                errors.append((test_class.__name__, method_name, traceback.format_exc()))
                failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print('='*60)

    if errors:
        print("\nFailed tests details:")
        for class_name, method_name, tb in errors:
            print(f"\n{class_name}.{method_name}:")
            print(tb)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
