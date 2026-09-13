"""Close the loop: world -> odor -> brain -> descending drive -> world."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from flyforge.brain import Brain, StubBrain
from flyforge.world import DEFAULT_ODOR_SOURCE, distance_to_source, fly_xy, make_world

BRAINS = {"stub": StubBrain}


def run(
    brain: Brain,
    steps: int = 5000,
    seed: int = 0,
    odor_source: np.ndarray | None = None,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    """Run the loop and return per-step logs."""
    source = DEFAULT_ODOR_SOURCE if odor_source is None else np.asarray(odor_source)
    sim = make_world(odor_source=source, seed=seed)
    obs, _ = sim.reset(seed=seed)
    brain.reset()

    log = {k: [] for k in ("pos", "action", "odor", "distance")}
    t0 = time.time()

    for i in range(steps):
        action = brain.step(obs["odor_intensity"])
        obs, _, terminated, truncated, _ = sim.step(action)

        log["pos"].append(fly_xy(obs))
        log["action"].append(np.asarray(action, dtype=np.float64))
        log["odor"].append(obs["odor_intensity"][0].copy())
        log["distance"].append(distance_to_source(obs, source))

        if progress and i % 1000 == 0:
            print(f"  step {i:6d}  d={log['distance'][-1]:6.2f} mm", flush=True)
        if terminated or truncated:
            print(f"  sim ended early at step {i}")
            break

    out = {k: np.asarray(v) for k, v in log.items()}
    out["elapsed"] = np.asarray(time.time() - t0)
    return out


def summarise(log: dict[str, np.ndarray]) -> str:
    d = log["distance"]
    elapsed = float(log["elapsed"])
    n = len(d)
    travelled = float(np.linalg.norm(np.diff(log["pos"], axis=0), axis=1).sum())
    return "\n".join(
        [
            f"  steps            {n}",
            f"  wall time        {elapsed:.1f}s  ({n / elapsed:.0f} steps/s)",
            f"  distance start   {d[0]:.2f} mm",
            f"  distance end     {d[-1]:.2f} mm",
            f"  closest approach {d.min():.2f} mm",
            f"  net progress     {d[0] - d[-1]:+.2f} mm",
            f"  path length      {travelled:.2f} mm",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the FlyForge closed loop")
    p.add_argument("--brain", choices=sorted(BRAINS), default="stub")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=Path, help="write per-step logs to this .npz")
    args = p.parse_args(argv)

    brain = BRAINS[args.brain](seed=args.seed)
    print(f"brain={args.brain} steps={args.steps} seed={args.seed}")
    log = run(brain, steps=args.steps, seed=args.seed, progress=True)
    print(summarise(log))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, **log)
        print(f"  saved -> {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
