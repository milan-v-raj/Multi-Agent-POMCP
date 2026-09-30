"""
Deep-POMCP Multi-Agent Pursuit-Evasion Benchmark Environment Package.
"""

from .scenarios import Obstacle, ScenarioConfig, ScenarioGenerator
from .evaders import BaseEvader, SmartRaycastEvader, RandomEvader, KeyboardEvader
from .stats_logger import StatsLogger, EpisodeStats
from .pathfinder import Pathfinder
from .core_env import PursuitEvasionEnv

def make_env(density_preset: str = "dense_maze",
             num_hunters: int = 2,
             evader_speed_mult: float = 1.0,
             render_mode: str = None,
             seed: int = None,
             max_steps: int = 1500) -> PursuitEvasionEnv:
    """Factory helper to quickly construct configured environments."""
    config = ScenarioConfig(
        density_preset=density_preset,
        num_hunters=num_hunters,
        evader_speed_mult=evader_speed_mult,
        max_steps=max_steps,
        seed=seed
    )
    return PursuitEvasionEnv(config=config, render_mode=render_mode)

__all__ = [
    "PursuitEvasionEnv",
    "ScenarioConfig",
    "ScenarioGenerator",
    "Obstacle",
    "BaseEvader",
    "SmartRaycastEvader",
    "RandomEvader",
    "KeyboardEvader",
    "StatsLogger",
    "EpisodeStats",
    "make_env"
]

