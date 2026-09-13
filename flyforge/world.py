"""The physical world: a fly with odor sensors in an arena with odor sources.

Thin wrapper over flygym-gymnasium (NeuroMechFly v2). Vision is off by default --
it requires rendering, which MuJoCo does not support headlessly on Windows, while
odor is computed from arena geometry and needs no renderer at all.
"""

from __future__ import annotations

import numpy as np
from flygym_gymnasium import Fly
from flygym_gymnasium.arena import OdorArena
from flygym_gymnasium.examples.locomotion import HybridTurningController
from flygym_gymnasium.preprogrammed import default_leg_sensor_placements

# Attractive source ahead and to the left, so a working brain has to actually steer.
DEFAULT_ODOR_SOURCE = np.array([[24.0, 8.0, 1.5]])
DEFAULT_PEAK_INTENSITY = np.array([[1.0, 0.0]])  # (attractive, aversive)


def make_world(
    odor_source: np.ndarray | None = None,
    peak_odor_intensity: np.ndarray | None = None,
    timestep: float = 1e-4,
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.2),
    arena_size: tuple[float, float] = (300, 300),
    seed: int = 0,
) -> HybridTurningController:
    """Build the sim. Action is (2,) descending drive, in Box(-0.5, 1.5)."""
    odor_source = DEFAULT_ODOR_SOURCE if odor_source is None else np.asarray(odor_source)
    peak = (
        DEFAULT_PEAK_INTENSITY
        if peak_odor_intensity is None
        else np.asarray(peak_odor_intensity)
    )

    fly = Fly(
        enable_olfaction=True,
        enable_vision=False,
        spawn_pos=spawn_pos,
        spawn_orientation=(0, 0, 0),
        # HybridTurningController needs tibia/tarsus contacts for stumbling detection.
        contact_sensor_placements=default_leg_sensor_placements,
    )
    arena = OdorArena(
        size=arena_size,
        odor_source=odor_source,
        peak_odor_intensity=peak,
        marker_colors=[(1.0, 0.4, 0.1, 1.0)] * len(odor_source),
    )
    return HybridTurningController(fly=fly, arena=arena, timestep=timestep, seed=seed)


def fly_xy(obs: dict) -> np.ndarray:
    """Fly's (x, y) position in arena coordinates."""
    return np.asarray(obs["fly"][0][:2], dtype=np.float64)


def distance_to_source(obs: dict, odor_source: np.ndarray) -> float:
    """Distance to the nearest odor source, in mm."""
    return float(np.linalg.norm(np.asarray(odor_source)[:, :2] - fly_xy(obs), axis=1).min())
