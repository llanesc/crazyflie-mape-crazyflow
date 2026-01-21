"""Curriculum learning manager for progressive difficulty training.

This module provides a curriculum learning system that automatically advances
through difficulty levels based on agent performance (blue win rate).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class CurriculumLevel:
    """A single difficulty level in the curriculum.

    Attributes:
        name: Human-readable name for this level.
        params: Dictionary of environment config parameters to override.
            Can include any field from RedVsBlueEnvConfig (e.g., br_crash_tolerance,
            bb_crash_tolerance, boundary_size, reward_capture, etc.).
        spawn: Spawn configuration dict for this level (optional).
    """
    name: str
    params: dict = field(default_factory=dict)
    spawn: dict = field(default_factory=dict)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning.

    Attributes:
        enabled: Whether curriculum learning is active.
        advance_threshold: Blue win rate required to advance (0.0-1.0).
        window_size: Number of episodes to track for win rate calculation.
        levels: List of curriculum levels from easiest to hardest.
        allow_regression: Whether to go back to easier levels if performance drops.
        regression_threshold: Win rate below which to regress (if enabled).
    """
    enabled: bool = True
    advance_threshold: float = 0.6
    window_size: int = 100
    levels: list[CurriculumLevel] = field(default_factory=list)
    allow_regression: bool = False
    regression_threshold: float = 0.3


class CurriculumManager:
    """Manages curriculum learning progression based on blue win rate.

    Tracks episode outcomes over a rolling window and advances/regresses
    through difficulty levels based on performance thresholds.

    Attributes:
        config: Curriculum configuration.
        current_level: Index of current difficulty level.
        episode_outcomes: Rolling window of episode outcomes (True = blue win).
        total_episodes: Total episodes seen across all levels.
        level_episodes: Episodes completed at each level.
    """

    def __init__(self, config: CurriculumConfig):
        """Initialize the curriculum manager.

        Args:
            config: Curriculum configuration with levels and thresholds.
        """
        self.config = config
        self.current_level = 0
        self.episode_outcomes: deque[bool] = deque(maxlen=config.window_size)
        self.total_episodes = 0
        self.level_episodes: list[int] = [0] * len(config.levels)
        self._on_level_change_callbacks: list[Callable[[int, CurriculumLevel], None]] = []

    @property
    def current_level_config(self) -> CurriculumLevel:
        """Get the current level configuration."""
        return self.config.levels[self.current_level]

    @property
    def blue_win_rate(self) -> float:
        """Calculate current blue win rate over the window."""
        if len(self.episode_outcomes) == 0:
            return 0.0
        return sum(self.episode_outcomes) / len(self.episode_outcomes)

    @property
    def is_final_level(self) -> bool:
        """Check if currently on the final (hardest) level."""
        return self.current_level >= len(self.config.levels) - 1

    @property
    def window_filled(self) -> bool:
        """Check if the episode window is fully filled."""
        return len(self.episode_outcomes) >= self.config.window_size

    def on_level_change(self, callback: Callable[[int, CurriculumLevel], None]):
        """Register a callback for level changes.

        Args:
            callback: Function called with (new_level_index, new_level_config)
                when the level changes.
        """
        self._on_level_change_callbacks.append(callback)

    def check_advancement(self, win_rate: float) -> bool:
        """Check if win rate exceeds threshold and advance if so.

        Args:
            win_rate: Current blue win rate (0.0-1.0).

        Returns:
            True if advanced to a new level, False otherwise.
        """
        if self.is_final_level:
            return False

        if win_rate >= self.config.advance_threshold:
            self._advance_level()
            return True

        return False

    def record_episode(self, blue_won: bool) -> dict[str, Any]:
        """Record an episode outcome and check for level advancement.

        Args:
            blue_won: Whether blue team won this episode.

        Returns:
            Dictionary with curriculum state info:
                - level: Current level index
                - level_name: Current level name
                - win_rate: Current blue win rate
                - advanced: Whether we just advanced to a new level
                - regressed: Whether we just regressed to an easier level
        """
        self.episode_outcomes.append(blue_won)
        self.total_episodes += 1
        self.level_episodes[self.current_level] += 1

        advanced = False
        regressed = False
        win_rate_at_change = None  # Track win rate that triggered level change

        # Only check for advancement once window is filled
        if self.window_filled:
            win_rate = self.blue_win_rate

            # Check for advancement
            if not self.is_final_level and win_rate >= self.config.advance_threshold:
                win_rate_at_change = win_rate
                self._advance_level()
                advanced = True
            # Check for regression
            elif (self.config.allow_regression
                  and self.current_level > 0
                  and win_rate < self.config.regression_threshold):
                win_rate_at_change = win_rate
                self._regress_level()
                regressed = True

        return {
            "level": self.current_level,
            "level_name": self.current_level_config.name,
            "win_rate": win_rate_at_change if win_rate_at_change is not None else self.blue_win_rate,
            "window_episodes": len(self.episode_outcomes),
            "advanced": advanced,
            "regressed": regressed,
        }

    def record_episodes_batch(self, blue_wins: np.ndarray) -> dict[str, Any]:
        """Record multiple episode outcomes from parallel environments.

        Args:
            blue_wins: Boolean array of shape (n_worlds,) indicating blue wins.

        Returns:
            Dictionary with curriculum state info (same as record_episode).
        """
        result = None
        for won in blue_wins:
            result = self.record_episode(bool(won))
        return result if result else {
            "level": self.current_level,
            "level_name": self.current_level_config.name,
            "win_rate": self.blue_win_rate,
            "window_episodes": len(self.episode_outcomes),
            "advanced": False,
            "regressed": False,
        }

    def _advance_level(self):
        """Advance to the next difficulty level."""
        old_level = self.current_level
        self.current_level = min(self.current_level + 1, len(self.config.levels) - 1)
        # Clear window when advancing to get fresh measurements
        self.episode_outcomes.clear()

        # Notify callbacks
        for callback in self._on_level_change_callbacks:
            callback(self.current_level, self.current_level_config)

        print(f"[Curriculum] Advanced from level {old_level} to {self.current_level} "
              f"({self.current_level_config.name})")

    def _regress_level(self):
        """Regress to an easier difficulty level."""
        old_level = self.current_level
        self.current_level = max(self.current_level - 1, 0)
        # Clear window when regressing
        self.episode_outcomes.clear()

        # Notify callbacks
        for callback in self._on_level_change_callbacks:
            callback(self.current_level, self.current_level_config)

        print(f"[Curriculum] Regressed from level {old_level} to {self.current_level} "
              f"({self.current_level_config.name})")

    def get_env_params(self) -> dict[str, Any]:
        """Get environment parameters for the current level.

        Returns:
            Dictionary of environment parameters to apply, including
            all params from the level config plus spawn configuration.
        """
        level = self.current_level_config
        params = dict(level.params)  # Copy to avoid mutation
        if level.spawn:
            params["spawn"] = level.spawn
        return params

    def get_stats(self) -> dict[str, Any]:
        """Get curriculum statistics for logging.

        Returns:
            Dictionary of curriculum stats suitable for tensorboard logging.
        """
        return {
            "curriculum/level": self.current_level,
            "curriculum/win_rate": self.blue_win_rate,
            "curriculum/total_episodes": self.total_episodes,
            "curriculum/window_episodes": len(self.episode_outcomes),
            "curriculum/level_episodes": self.level_episodes[self.current_level],
        }


def load_curriculum_config(config: dict) -> Optional[CurriculumConfig]:
    """Load curriculum configuration from experiment config dict.

    Args:
        config: Full experiment configuration dictionary.

    Returns:
        CurriculumConfig if curriculum is defined, None otherwise.

    Example YAML structure:
        curriculum:
          enabled: true
          advance_threshold: 0.6
          window_size: 100
          levels:
            - name: "Easy"
              params:
                br_crash_tolerance: 0.5
                bb_crash_tolerance: 0.3
                boundary_size: 4.0
              spawn:
                method: "deterministic"
                blue_x: -2.0
                red_x: 2.0
            - name: "Hard"
              params:
                br_crash_tolerance: 0.2
                bb_crash_tolerance: 0.2
                boundary_size: 3.0
    """
    curriculum_cfg = config.get("curriculum")
    if curriculum_cfg is None or not curriculum_cfg.get("enabled", False):
        return None

    levels = []
    for i, level_dict in enumerate(curriculum_cfg.get("levels", [])):
        # Support both "name" key and auto-generated name from "level" key or index
        name = level_dict.get("name")
        if name is None:
            level_num = level_dict.get("level", i)
            name = f"Level {level_num}"

        # Extract params (all keys except 'name', 'level', 'spawn')
        params = {k: v for k, v in level_dict.items() if k not in ("name", "level", "spawn")}

        level = CurriculumLevel(
            name=name,
            params=params,
            spawn=level_dict.get("spawn", {}),
        )
        levels.append(level)

    if not levels:
        print("[Curriculum] Warning: No levels defined, disabling curriculum")
        return None

    return CurriculumConfig(
        enabled=True,
        advance_threshold=curriculum_cfg.get("advance_threshold", 0.6),
        window_size=curriculum_cfg.get("window_size", 100),
        levels=levels,
        allow_regression=curriculum_cfg.get("allow_regression", False),
        regression_threshold=curriculum_cfg.get("regression_threshold", 0.3),
    )
