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
from pathlib import Path

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
