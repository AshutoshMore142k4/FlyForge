"""Brains map odor readings to a 2D descending command.

The 2D action is the descending-neuron bottleneck: HybridTurningController turns
[left_drive, right_drive] into leg movement via its CPG, so a brain never has to
drive joints directly.
"""

from typing import Protocol

import numpy as np

# Sensor layout of obs["odor_intensity"], shape (k_dims, 4).
# Verified against flygym's fly.olfaction_sensor_positions -- see tests/test_loop.py.
LEFT_SENSORS = (0, 2)   # left antenna, left maxillary palp
RIGHT_SENSORS = (1, 3)  # right antenna, right maxillary palp


class Brain(Protocol):
    """Odor in, descending drive out."""

    def reset(self) -> None: ...

    def step(self, odor: np.ndarray) -> np.ndarray:
        """odor: (k_dims, 4) -> action: (2,) in [0, 1]."""
        ...


def _valence(k_dims: int) -> np.ndarray:
    """First odor dimension attracts, any others repel."""
    w = -np.ones(k_dims, dtype=np.float64)
    w[0] = 1.0
    return w


class StubBrain:
    """Klinotaxis baseline: steer toward whichever side smells better.

    Not scaffolding -- this is the behavioural control that ConnectomeBrain gets
    scored against, so it stays in the repo permanently.

    ponytail: pure gradient-following with no arrest term, so it overshoots the
    source and sails past (25mm -> 7mm -> back out to 14mm over 30k steps). Fine as
    a navigation baseline; add odor-gated slowing if Phase 4 needs it to actually
    stay on the food.
    """

    def __init__(
        self,
        gain: float = 3.0,
        base_drive: float = 1.0,
        min_drive: float = 0.2,
        noise: float = 0.05,
        seed: int | None = None,
    ):
        self.gain = gain
        self.base_drive = base_drive
        self.min_drive = min_drive
        self.noise = noise
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def step(self, odor: np.ndarray) -> np.ndarray:
        odor = np.atleast_2d(np.asarray(odor, dtype=np.float64))
        w = _valence(odor.shape[0])

        # Weighted attractiveness per sensor, then pooled per side.
        per_sensor = w @ odor                      # (4,)
        left = per_sensor[list(LEFT_SENSORS)].sum()
        right = per_sensor[list(RIGHT_SENSORS)].sum()

        # Normalised asymmetry: scale-free, so it works near and far from the source.
        denom = abs(left) + abs(right)
        asym = (left - right) / denom if denom > 1e-12 else 0.0
        asym += self._rng.normal(0.0, self.noise)

        # Higher drive on one side turns the fly away from it, so the stronger
        # side gets the *lower* drive.
        turn = self.gain * asym
        action = np.array([self.base_drive - turn, self.base_drive + turn])
        return np.clip(action, self.min_drive, 1.0)
