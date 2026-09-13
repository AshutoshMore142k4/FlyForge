"""Smoke tests: the sensor -> brain -> motor loop closes and the wiring is right."""

import numpy as np
import pytest

from flyforge.brain import LEFT_SENSORS, RIGHT_SENSORS, StubBrain
from flyforge.run import run
from flyforge.world import DEFAULT_ODOR_SOURCE, fly_xy, make_world


def test_stub_brain_steers_toward_stronger_side():
    """Higher drive on a side turns the fly away from it, so the stronger-smelling
    side must get the lower drive."""
    brain = StubBrain(noise=0.0, seed=0)

    odor = np.zeros((2, 4))
    odor[0, list(LEFT_SENSORS)] = 1.0
    left_action = brain.step(odor)
    assert left_action[0] < left_action[1], "odor on the left should turn the fly left"

    odor = np.zeros((2, 4))
    odor[0, list(RIGHT_SENSORS)] = 1.0
    right_action = brain.step(odor)
    assert right_action[0] > right_action[1]


def test_brain_output_contract():
    """Whatever the odor, the action stays a finite (2,) inside the action space."""
    brain = StubBrain(seed=0)
    rng = np.random.default_rng(0)
    for odor in [np.zeros((2, 4)), np.ones((2, 4)), rng.random((2, 4)) * 1e6]:
        action = brain.step(odor)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))
        assert np.all(action >= -0.5) and np.all(action <= 1.5)


def test_sensor_sides_match_simulator():
    """brain.py hardcodes which sensor index is left. If flygym ever reorders them,
    every brain silently steers the wrong way -- so pin it against the simulator."""
    sim = make_world(odor_source=np.array([[0.0, 20.0, 1.5]]))  # source to the fly's left
    obs, _ = sim.reset(seed=0)
    intensity = obs["odor_intensity"][0]

    left = intensity[list(LEFT_SENSORS)].sum()
    right = intensity[list(RIGHT_SENSORS)].sum()
    assert left > right, f"expected LEFT_SENSORS to read stronger, got {intensity}"


def test_closed_loop_runs_and_approaches_source():
    """The whole loop turns, and klinotaxis makes real progress up the gradient."""
    log = run(StubBrain(seed=0), steps=600, seed=0)

    assert log["action"].shape == (600, 2)
    assert np.all(np.isfinite(log["action"]))
    assert log["odor"].max() > 0, "fly never smelled anything"

    moved = np.linalg.norm(log["pos"][-1] - log["pos"][0])
    assert moved > 0.1, f"fly barely moved ({moved:.3f} mm)"
    assert log["distance"][-1] < log["distance"][0], "fly did not approach the source"


def test_observation_shape_is_two_dims_four_sensors():
    sim = make_world()
    obs, _ = sim.reset(seed=0)
    assert obs["odor_intensity"].shape == (2, 4)
    assert fly_xy(obs).shape == (2,)
