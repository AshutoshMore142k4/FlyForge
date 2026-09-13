# FlyForge

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/AshutoshMore142k4/FlyForge)
![GitHub stars](https://img.shields.io/github/stars/AshutoshMore142k4/FlyForge?style=social)

An embodied simulation of the *Drosophila* connectome: a real fly nervous system
wired into a physics-simulated body, in a world it has to survive in.

The question is not "can this hit a benchmark" but **what does the connectome
actually do when you embed it in a world?**

```
odor  →  brain  →  descending drive  →  CPG  →  legs  →  world  ↺
```

## Contents

- [Status](#status)
- [Architecture](#architecture)
- [Install](#install)
- [Run](#run)
- [Layout](#layout)
- [Connectome data](#connectome-data)
- [Roadmap](#roadmap)
- [Design notes](#design-notes)
- [Credits](#credits)

## Status

🟢 **Phase 1** — sensorimotor loop closes end to end, klinotaxis baseline.
🟡 **Phase 3** — `ConnectomeBrain` builds and runs on the real 164k-neuron MaleCNS
graph (GPU sparse CSR), but is unverified: no dedicated tests yet, and not wired
into the CLI. Treat it as a working prototype, not a finished feature.

## Architecture

`HybridTurningController` accepts a **2D action** — descending drive to the left and
right side — and expands it into leg movement through a CPG. That bottleneck is not a
convenience: it mirrors how real descending neurons carry low-dimensional steering
commands from brain to ventral nerve cord. A brain here never drives 42 joints. It
produces two numbers.

```
        OdorArena  (flygym-gymnasium / MuJoCo)
              │  obs["odor_intensity"] → (2, 4)
              │  2 odor dims × 4 sensors (2 antennae, 2 maxillary palps)
              ▼
    ┌──────────────────────────────┐
    │  Brain                       │   StubBrain (shipped)
    │  odor (2,4) → action (2,)    │   ConnectomeBrain (prototype)
    └──────────────┬───────────────┘
                   ▼
     HybridTurningController → CPG → legs → world
```

Sensor layout is `LEFT = (0, 2)`, `RIGHT = (1, 3)`, verified against the simulator in
`tests/test_loop.py` rather than assumed — if flygym ever reorders them, every brain
would silently steer backwards.

## Install

Python 3.12 specifically: `flygym-gymnasium` requires `>=3.10,<3.13`.

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
pip install flygym-gymnasium
```

`--system-site-packages` inherits an existing CUDA torch instead of re-downloading it.
`flygym-gymnasium` pins `numba==0.60.0`, which caps NumPy below 2.1 — inside the venv
that downgrade is harmless, globally it would break a CUDA torch install.

## Run

```powershell
python -m flyforge.run --brain stub --steps 30000 --save runs/stub.npz
pytest tests/ -q
```

The baseline navigates from 25.3 mm to within 7.2 mm of the source, then overshoots
and drifts back out — pure klinotaxis with no arrest term. That is a real property of
the controller, not a bug, and it is what `ConnectomeBrain` has to beat.
Throughput is ~670 steps/s (timestep 1e-4, so roughly 15x slower than real time).

```powershell
python -c "from flyforge.connectome import build_graph; g = build_graph(); print(g.n_neurons, 'neurons')"
```

builds (or loads the cached) whole-CNS graph — 164,481 neurons, ~23.3M signed edges —
in a couple of seconds once the data is downloaded.

## Layout

| File | What |
|---|---|
| `flyforge/world.py` | arena, fly, odor sources; flygym wrapper |
| `flyforge/brain.py` | `Brain` protocol, `StubBrain`, `ConnectomeBrain` (prototype) |
| `flyforge/connectome.py` | MaleCNS v1.0 download + sparse graph build |
| `flyforge/run.py` | the closed loop + CLI |
| `tests/test_loop.py` | smoke tests; pins the sensor-side assumption |

`StubBrain` is not scaffolding. It is the behavioural control that every connectome
network gets scored against.

## Connectome data

```powershell
python -m flyforge.connectome --inspect
```

Pulls three files (~560 MB) from the public `gs://flyem-male-cns` bucket — no neuPrint
token needed. MaleCNS v1.0 is CC-BY 4.0 (Janelia FlyEM / Google Research / MRC LMB):
176,422 neurons.

| File | Size | Why |
|---|---|---|
| `connectome-weights-…-significant-only` | 502 MB | the graph |
| `body-annotations-…` | 14.5 MB | `type`, `class`, `superclass`, `somaSide` |
| `body-neurotransmitters-…` | 43.3 MB | excitatory vs. inhibitory sign |

~23 GB of per-synapse coordinate data (`syn-points`, `syn-partners`, `tbar-…`,
`body-stats`) is deliberately not downloaded — a connectivity-driven rate model never
reads it.

`ConnectomeBrain` signs each edge from its presynaptic neuron's predicted
neurotransmitter (ACh/Glu → excitatory, GABA → inhibitory), row-normalises the
resulting sparse matrix, and injects odor drive at ORN rows / reads steering out at DN
rows — the whole graph runs on GPU, not just a hop-limited subgraph.

## Roadmap

- [x] **0** Environment
- [x] **1** Closed loop, stub brain, tests
- [x] **2** MaleCNS download + schema
- [ ] **3** `ConnectomeBrain` — builds and runs; needs tests + CLI wiring + behavioural tuning
- [ ] **4** Survival: energy, food, death
- [ ] **5** Plasticity, and the comparison the project exists to make:
      connectome fixed / connectome + plasticity / **degree-matched shuffle** /
      conventional ANN at matched parameter count
- [ ] **6** Vision (needs rendering → likely WSL2; `flyvis` is the starting point)

Phase 5's shuffled control is the one that matters — it separates "this connectome's
specific wiring does something" from "any sparse recurrent net of this size does
something".

## Design notes

**Why olfaction first.** The optic lobes are the majority of the brain's neurons;
the olfactory pathway is ~10k in the loop. Odor is also computed from arena geometry,
so it needs no renderer — and MuJoCo has no headless rendering backend on Windows.
Vision is what forces WSL2, so it comes last.

**Why rate units, not spiking.** One sparse matmul per step is enough to ask the
behavioural question. Spiking multiplies cost and tuning burden before there is any
behaviour to explain.

**The GPU is not the constraint.** 164k neurons and ~23M edges is well under 200 MB as
sparse CSR. The hard part is the sensory and motor *mapping* — which neurons the
sensors drive, and how activity becomes movement.

## Credits

- [NeuroMechFly v2](https://neuromechfly.org/) — Ramdya lab, EPFL ([Nature Methods 2024](https://www.nature.com/articles/s41592-024-02497-y))
- [MaleCNS connectome](https://male-cns.janelia.org/) — Janelia FlyEM, Google Research, MRC LMB

MIT licensed.
