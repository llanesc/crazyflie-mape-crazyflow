"""Test collision detection logic for MAPE server.

Verifies that the vectorized meshgrid-based collision detection:
1. Does NOT trigger self-collisions (the original bug)
2. Correctly detects actual collisions between different agents
3. Handles edge cases properly
"""

import numpy as np
from numpy.linalg import norm


def create_meshgrid_and_permutations(n: int):
    """Create meshgrid and permutation arrays for collision detection."""
    meshgrid = np.arange(n)[None, :].repeat(n, axis=0)
    permutations = np.array([np.roll(np.arange(n), -k) for k in np.arange(n)])
    return meshgrid, permutations


def check_same_team_collisions(positions: np.ndarray, meshgrid: np.ndarray,
                                permutations: np.ndarray, tolerance: float) -> np.ndarray:
    """Check for collisions within the same team using vectorized meshgrid approach."""
    n = positions.shape[0]
    dist = norm(positions[meshgrid, :] - positions[:, None, :], axis=2, keepdims=True)
    collision = dist < tolerance
    crash = np.any(collision[meshgrid[:-1, :].T, permutations[:, 1:]], axis=1).flatten()
    return crash


def test_meshgrid_permutation_values():
    """Test that meshgrid and permutation arrays are generated correctly."""
    print("=" * 60)
    print("Test: Meshgrid and permutation values")
    print("=" * 60)

    # Test n=2
    meshgrid, perms = create_meshgrid_and_permutations(2)
    print(f"\nn=2:")
    print(f"  meshgrid:\n{meshgrid}")
    print(f"  permutations:\n{perms}")

    expected_meshgrid_2 = np.array([[0, 1], [0, 1]])
    expected_perms_2 = np.array([[0, 1], [1, 0]])

    assert np.array_equal(meshgrid, expected_meshgrid_2), f"meshgrid mismatch: {meshgrid} != {expected_meshgrid_2}"
    assert np.array_equal(perms, expected_perms_2), f"permutations mismatch: {perms} != {expected_perms_2}"

    # Test n=3
    meshgrid, perms = create_meshgrid_and_permutations(3)
    print(f"\nn=3:")
    print(f"  meshgrid:\n{meshgrid}")
    print(f"  permutations:\n{perms}")

    expected_meshgrid_3 = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    expected_perms_3 = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    assert np.array_equal(meshgrid, expected_meshgrid_3), f"meshgrid mismatch"
    assert np.array_equal(perms, expected_perms_3), f"permutations mismatch"

    print("\n[PASS] Meshgrid and permutation values are correct")


def test_no_self_collision():
    """Test that agents do NOT collide with themselves."""
    print("\n" + "=" * 60)
    print("Test: No self-collision (the original bug)")
    print("=" * 60)

    for n in [2, 3, 4]:
        meshgrid, perms = create_meshgrid_and_permutations(n)

        # Agents far apart - no collisions should occur
        positions = np.array([[i * 10.0, 0.0, 1.0] for i in range(n)])
        tolerance = 0.2

        crash = check_same_team_collisions(positions, meshgrid, perms, tolerance)

        print(f"\nn={n}, positions spread 10m apart:")
        print(f"  positions: {positions[:, 0].tolist()}")
        print(f"  crash result: {crash.tolist()}")

        assert not np.any(crash), f"False collision detected for n={n}! crash={crash}"

    print("\n[PASS] No self-collisions detected")


def test_actual_collision_detected():
    """Test that actual collisions between different agents are detected."""
    print("\n" + "=" * 60)
    print("Test: Actual collisions are detected")
    print("=" * 60)

    tolerance = 0.2

    # n=2: agents close together
    meshgrid, perms = create_meshgrid_and_permutations(2)
    positions = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])  # 0.1m apart
    crash = check_same_team_collisions(positions, meshgrid, perms, tolerance)
    print(f"\nn=2, agents 0.1m apart (tolerance={tolerance}):")
    print(f"  crash result: {crash.tolist()}")
    assert np.all(crash), f"Collision not detected! crash={crash}"

    # n=3: only agents 0 and 1 collide
    meshgrid, perms = create_meshgrid_and_permutations(3)
    positions = np.array([
        [0.0, 0.0, 1.0],   # Agent 0
        [0.1, 0.0, 1.0],   # Agent 1 - close to 0
        [10.0, 0.0, 1.0],  # Agent 2 - far away
    ])
    crash = check_same_team_collisions(positions, meshgrid, perms, tolerance)
    print(f"\nn=3, agents 0 and 1 close (0.1m), agent 2 far (10m):")
    print(f"  crash result: {crash.tolist()}")
    assert crash[0] and crash[1] and not crash[2], f"Unexpected crash pattern: {crash}"

    print("\n[PASS] Actual collisions detected correctly")


def test_edge_at_tolerance():
    """Test behavior at exactly the tolerance boundary."""
    print("\n" + "=" * 60)
    print("Test: Edge cases at tolerance boundary")
    print("=" * 60)

    tolerance = 0.2
    meshgrid, perms = create_meshgrid_and_permutations(2)

    # Exactly at tolerance - should NOT crash (< not <=)
    positions = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]])
    crash = check_same_team_collisions(positions, meshgrid, perms, tolerance)
    print(f"\nAgents exactly 0.2m apart (tolerance=0.2):")
    print(f"  crash result: {crash.tolist()}")
    assert not np.any(crash), f"Should not crash at exactly tolerance: {crash}"

    # Just under tolerance - should crash
    positions = np.array([[0.0, 0.0, 1.0], [0.199, 0.0, 1.0]])
    crash = check_same_team_collisions(positions, meshgrid, perms, tolerance)
    print(f"\nAgents 0.199m apart (tolerance=0.2):")
    print(f"  crash result: {crash.tolist()}")
    assert np.all(crash), f"Should crash just under tolerance: {crash}"

    print("\n[PASS] Edge cases handled correctly")


def test_compare_with_loop_implementation():
    """Compare vectorized implementation with simple loop implementation."""
    print("\n" + "=" * 60)
    print("Test: Compare vectorized vs loop implementation")
    print("=" * 60)

    def loop_collision_check(positions, tolerance):
        """Reference loop-based implementation."""
        n = len(positions)
        crash = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if norm(positions[i] - positions[j]) < tolerance:
                    crash[i] = True
                    crash[j] = True
        return crash

    tolerance = 0.2
    np.random.seed(42)

    for n in [2, 3, 4, 5]:
        meshgrid, perms = create_meshgrid_and_permutations(n)

        # Random positions
        positions = np.random.randn(n, 3) * 0.5  # Some will collide

        crash_vectorized = check_same_team_collisions(positions, meshgrid, perms, tolerance)
        crash_loop = loop_collision_check(positions, tolerance)

        print(f"\nn={n}:")
        print(f"  vectorized: {crash_vectorized.tolist()}")
        print(f"  loop:       {crash_loop.tolist()}")

        assert np.array_equal(crash_vectorized, crash_loop), \
            f"Mismatch for n={n}! vectorized={crash_vectorized}, loop={crash_loop}"

    print("\n[PASS] Vectorized matches loop implementation")


def test_original_bug_scenario():
    """Reproduce the exact scenario that caused red_2 to die instantly."""
    print("\n" + "=" * 60)
    print("Test: Original bug scenario (red_2 instant death)")
    print("=" * 60)

    # The bug: np.argsort produces [[0,1], [0,1]] instead of [[0,1], [1,0]]
    # This causes permutations[:,1:] = [[1], [1]] instead of [[1], [0]]
    # Result: both agents check distance to agent 1, not to each other

    n = 2
    tolerance = 0.2

    # Buggy implementation
    buggy_meshgrid = np.meshgrid(np.arange(n), np.arange(n))[0]
    buggy_perms = np.argsort(buggy_meshgrid, axis=1)

    # Correct implementation
    correct_meshgrid, correct_perms = create_meshgrid_and_permutations(n)

    print(f"\nBuggy permutations:\n{buggy_perms}")
    print(f"Correct permutations:\n{correct_perms}")
    print(f"\nBuggy permutations[:,1:]: {buggy_perms[:,1:].tolist()}")
    print(f"Correct permutations[:,1:]: {correct_perms[:,1:].tolist()}")

    # With buggy implementation, both rows index column 1
    # This means agent 1 checks distance to itself (index 1)!

    # Agents far apart - should have NO collisions
    positions = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]])

    # Check with buggy implementation
    dist = norm(positions[buggy_meshgrid, :] - positions[:, None, :], axis=2, keepdims=True)
    buggy_crash = np.any(dist[buggy_meshgrid[:-1, :].T, buggy_perms[:, 1:]] < tolerance, axis=1).flatten()

    # Check with correct implementation
    dist = norm(positions[correct_meshgrid, :] - positions[:, None, :], axis=2, keepdims=True)
    correct_crash = np.any(dist[correct_meshgrid[:-1, :].T, correct_perms[:, 1:]] < tolerance, axis=1).flatten()

    print(f"\nAgents 10m apart:")
    print(f"  Buggy crash result: {buggy_crash.tolist()}")
    print(f"  Correct crash result: {correct_crash.tolist()}")

    # The bug causes agent 1 to "crash" because it checks distance to itself (0)
    assert buggy_crash[1] == True, "Bug should cause agent 1 to falsely crash"
    assert not np.any(correct_crash), "Correct implementation should have no crashes"

    print("\n[PASS] Bug scenario verified and fixed")


if __name__ == '__main__':
    test_meshgrid_permutation_values()
    test_no_self_collision()
    test_actual_collision_detected()
    test_edge_at_tolerance()
    test_compare_with_loop_implementation()
    test_original_bug_scenario()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
