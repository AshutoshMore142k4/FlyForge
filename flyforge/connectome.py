"""MaleCNS v1.0 connectome data.

The flat-connectome files live in a publicly readable GCS bucket, so no neuPrint
token is needed. Data is CC-BY 4.0 (Janelia FlyEM / Google Research / MRC LMB).

We fetch three files (~560 MB) and deliberately skip ~23 GB of per-synapse
coordinate data (syn-points, syn-partners, tbar-neurotransmitters, body-stats),
which a connectivity-driven rate model never reads.
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0"
    "/connectome-data/flat-connectome/"
)

FILES: dict[str, str] = {
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(name: str, data_dir: Path = DATA_DIR, force: bool = False) -> Path:
    """Download one file, skipping it if already complete."""
    filename = FILES[name]
    dest = data_dir / filename
    data_dir.mkdir(parents=True, exist_ok=True)

    url = BASE_URL + filename
    with urllib.request.urlopen(url) as resp:
        remote_size = int(resp.headers["Content-Length"])

        if dest.exists() and not force:
            if dest.stat().st_size == remote_size:
                print(f"  have {filename} ({_human(remote_size)})")
                return dest
            print(f"  size mismatch, refetching {filename}")

        # Write to a temp file first so an interrupted download never looks complete.
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"  get  {filename} ({_human(remote_size)})")
        done = 0
        with open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                pct = 100 * done / remote_size
                print(f"\r       {pct:5.1f}%  {_human(done)}", end="", flush=True)
        print()

    if tmp.stat().st_size != remote_size:
        tmp.unlink()
        raise IOError(f"{filename}: truncated download")
    shutil.move(tmp, dest)
    return dest


def download_all(data_dir: Path = DATA_DIR, force: bool = False) -> dict[str, Path]:
    print(f"MaleCNS v1.0 flat connectome -> {data_dir}  (~560 MB)")
    return {k: download(k, data_dir, force) for k in FILES}


def load(name: str, data_dir: Path = DATA_DIR):
    """Read one table, downloading it on first use."""
    import pandas as pd

    return pd.read_feather(download(name, data_dir))


# --- whole-CNS graph for ConnectomeBrain --------------------------------------

# Per-presynaptic-neuron NT -> sign of every synapse it makes (recommended mapping
# from the data investigation). Neuromodulatory/unresolved NTs are excluded (0),
# not forced to a sign.
NT_SIGN: dict[str, float] = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "dopamine": 0.0,
    "serotonin": 0.0,
    "octopamine": 0.0,
    "histamine": 0.0,
    "unclear": 0.0,
}

GRAPH_CACHE_NAME = "connectome_graph.pt"

# W: torch.sparse_csr_tensor, shape (n_neurons, n_neurons). W[i, j] = signed,
# row-normalised weight from presynaptic neuron j onto postsynaptic neuron i
# (row = post, col = pre) -- so `W @ r` sums each neuron's presynaptic input,
# matching r_new = r + (dt/tau) * (-r + relu(W @ r + I_ext)).
# orn_left/orn_right/dn_left/dn_right are int64 node-index tensors (positions in
# the same 0..n_neurons-1 space as W), not raw connectome bodyIds.
ConnectomeGraph = namedtuple(
    "ConnectomeGraph", ["W", "n_neurons", "orn_left", "orn_right", "dn_left", "dn_right"]
)


def _build_sparse_csr(body_pre: np.ndarray, body_post: np.ndarray, signed_weight: np.ndarray):
    """Pure, testable core: edge lists -> (bodyId index, row-normalised CSR).

    Vectorised (pandas.Index.get_indexer + torch.scatter_add) -- no Python loop
    over rows, so this stays fine at tens of millions of edges.
    """
    all_ids = pd.unique(np.concatenate([body_pre, body_post]))
    id_index = pd.Index(all_ids)
    n = len(id_index)

    row = id_index.get_indexer(body_post)  # postsynaptic = row
    col = id_index.get_indexer(body_pre)  # presynaptic = col
    indices = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    values = torch.tensor(signed_weight, dtype=torch.float32)
    coo = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()

    # Row-normalise by L1 so relu(W @ r) can't blow up regardless of raw synapse
    # counts (1-2591) or a neuron's fan-in.
    # ponytail: crude stabiliser, not a biological claim -- revisit with a
    # principled per-neuron gain if the model ever needs to match real firing
    # rates instead of just producing stable, steerable dynamics.
    idx, val = coo.indices(), coo.values()
    row_abs = torch.zeros(n, dtype=torch.float32).scatter_add_(0, idx[0], val.abs())
    val = val / row_abs[idx[0]].clamp_min(1e-8)
    W = torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce().to_sparse_csr()
    return id_index, W


def build_graph(
    min_weight: int = 1,
    min_nt_confidence: float = 0.5,
    force_rebuild: bool = False,
    data_dir: Path = DATA_DIR,
    cache_path: Path | None = None,
) -> ConnectomeGraph:
    """Build (or load the cached) whole-CNS signed connectivity graph.

    Whole-CNS, not a hop-limited subgraph: at ~176k neurons / ~million edges a
    sparse tensor is trivially small, so injecting drive at ORN rows and reading
    out DN rows already behaves like a subgraph without truncating pathways.
    """
    cache_path = cache_path or (data_dir / GRAPH_CACHE_NAME)
    params = {"min_weight": min_weight, "min_nt_confidence": min_nt_confidence}

    if cache_path.exists() and not force_rebuild:
        cached = torch.load(cache_path, weights_only=False)
        if cached.get("params") == params:
            return ConnectomeGraph(
                W=cached["W"],
                n_neurons=cached["n_neurons"],
                orn_left=cached["orn_left"],
                orn_right=cached["orn_right"],
                dn_left=cached["dn_left"],
                dn_right=cached["dn_right"],
            )

    weights = load("weights", data_dir)
    if min_weight > 1:
        weights = weights[weights["weight"] >= min_weight]
    nt = load("neurotransmitters", data_dir)
    ann = load("annotations", data_dir)

    # Sign per edge is decided by the PRESYNAPTIC neuron's predicted NT, gated by
    # that same neuron's own predicted_nt_confidence -- predicted_nt is the only
    # NT column with a matching per-body confidence column (consensus_nt has none),
    # and min_nt_confidence must gate a real confidence value, not a fabricated one.
    sign_by_body = nt.set_index("body")["predicted_nt"].map(NT_SIGN)
    conf_by_body = nt.set_index("body")["predicted_nt_confidence"]

    pre_sign = weights["body_pre"].map(sign_by_body).fillna(0.0).to_numpy()
    pre_conf = weights["body_pre"].map(conf_by_body).fillna(0.0).to_numpy()
    pre_sign = np.where(pre_conf >= min_nt_confidence, pre_sign, 0.0)

    signed_weight = weights["weight"].to_numpy(dtype=np.float64) * pre_sign
    keep = signed_weight != 0.0
    body_pre = weights["body_pre"].to_numpy()[keep]
    body_post = weights["body_post"].to_numpy()[keep]
    signed_weight = signed_weight[keep]

    id_index, W = _build_sparse_csr(body_pre, body_post, signed_weight)
    n = len(id_index)

    def side_idx(mask: pd.Series, side_col: str, side_val: str) -> torch.Tensor:
        ids = ann.loc[mask & (ann[side_col] == side_val), "bodyId"].to_numpy()
        pos = id_index.get_indexer(ids)
        return torch.tensor(pos[pos >= 0], dtype=torch.long)

    orn_mask = ann["class"] == "olfactory"
    dn_mask = ann["superclass"] == "descending_neuron"
    # ORNs have somaSide==NaN (peripheral); left/right lives in rootSide instead.
    orn_left = side_idx(orn_mask, "rootSide", "L")
    orn_right = side_idx(orn_mask, "rootSide", "R")
    dn_left = side_idx(dn_mask, "somaSide", "L")
    dn_right = side_idx(dn_mask, "somaSide", "R")

    graph = ConnectomeGraph(W, n, orn_left, orn_right, dn_left, dn_right)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "params": params,
            "W": graph.W,
            "n_neurons": graph.n_neurons,
            "orn_left": graph.orn_left,
            "orn_right": graph.orn_right,
            "dn_left": graph.dn_left,
            "dn_right": graph.dn_right,
        },
        cache_path,
    )
    return graph


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Fetch MaleCNS connectome tables")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--force", action="store_true")
    p.add_argument("--inspect", action="store_true", help="print schema after download")
    args = p.parse_args(argv)

    paths = download_all(args.data_dir, args.force)

    if args.inspect:
        import pandas as pd

        for name, path in paths.items():
            df = pd.read_feather(path)
            print(f"\n=== {name}: {len(df):,} rows ===")
            print(df.dtypes.to_string())
            print(df.head(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
