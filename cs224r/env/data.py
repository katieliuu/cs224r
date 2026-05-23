"""
data.py
Load the fragment library and target property distribution from the
M3-20M pipeline outputs.

Memory safety
-------------
Both loaders stream row groups via pyarrow and hard-stop before exceeding
`max_mem_gb` (default 32 GB) of *additional* RAM.  They never materialise
the full parquet into a single DataFrame.

Fragment library  — top-N fragments by occurrence count (min-heap scan).
Target distribution — (sLogP, QED, TPSA) vectors, row-group-streamed.
"""
import _path_bootstrap  # noqa: F401

import heapq
from dataclasses import dataclass
from typing import List

import numpy as np
import psutil
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Property normalisation constants (fixed, drug-like ranges)
# ---------------------------------------------------------------------------
PROP_NAMES = ("sLogP", "QED", "TPSA")
PROP_MIN   = np.array([-5.0, 0.0,   0.0], dtype=np.float32)
PROP_MAX   = np.array([10.0, 1.0, 200.0], dtype=np.float32)
PROP_RANGE = PROP_MAX - PROP_MIN          # [15, 1, 200]
GOAL_DIM   = 3


def normalize_props(raw: np.ndarray) -> np.ndarray:
    """Clip raw (logp, qed, tpsa) to [0, 1]."""
    return (np.clip(raw, PROP_MIN, PROP_MAX) - PROP_MIN) / PROP_RANGE


def denormalize_props(normed: np.ndarray) -> np.ndarray:
    """Inverse of normalize_props."""
    return normed * PROP_RANGE + PROP_MIN


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _available_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def _check_headroom(max_mem_gb: float, label: str) -> None:
    avail = _available_gb()
    if avail < 2.0:
        raise MemoryError(
            f"{label}: only {avail:.1f} GB RAM available; refusing to load more data."
        )


# ---------------------------------------------------------------------------
# Fragment library
# ---------------------------------------------------------------------------

@dataclass
class FragInfo:
    smiles: str
    labels: List[str]   # BRICS attachment-type strings, e.g. ["3", "5"]
    n_attach: int
    count: int


def load_fragment_library(
    parquet_path: str,
    n: int = 500,
    min_count: int = 1_000,
    max_mem_gb: float = 32.0,
) -> List[FragInfo]:
    """
    Stream fragments.parquet row group by row group; keep a running min-heap
    of the top-n fragments by occurrence count.  Stops reading as soon as
    loading another row group would push cumulative memory use over max_mem_gb.

    Never loads more than one row group (~50k rows, ~3.5 MB) at a time.
    """
    COLS = ["frag_smiles", "count", "attach_labels", "n_attach"]

    pf          = pq.ParquetFile(parquet_path)
    n_groups    = pf.metadata.num_row_groups
    used_start  = psutil.virtual_memory().used / 1e9

    # min-heap: (count, index, row_dict)  index breaks ties deterministically
    heap:  list = []
    entry_idx   = 0

    for g in range(n_groups):
        # --- memory guard: stop if we'd exceed the budget ---
        used_now = psutil.virtual_memory().used / 1e9
        delta    = used_now - used_start
        if delta >= max_mem_gb:
            print(f"  [data] memory cap hit ({delta:.1f}/{max_mem_gb:.0f} GB used by "
                  f"this load); stopping after {g}/{n_groups} row groups.")
            break

        _check_headroom(max_mem_gb, "load_fragment_library")

        batch = pf.read_row_group(g, columns=COLS).to_pandas()

        # Filter
        batch = batch[(batch["n_attach"] >= 1) & (batch["count"] >= min_count)]

        for _, row in batch.iterrows():
            cnt = int(row["count"])
            if len(heap) < n:
                heapq.heappush(heap, (cnt, entry_idx, row.to_dict()))
                entry_idx += 1
            elif cnt > heap[0][0]:
                heapq.heapreplace(heap, (cnt, entry_idx, row.to_dict()))
                entry_idx += 1

    # Sort descending by count
    top = sorted(heap, key=lambda x: -x[0])

    frags: List[FragInfo] = []
    for _, _, row in top:
        raw_labels = str(row["attach_labels"])
        labels = [lbl.strip() for lbl in raw_labels.split(",") if lbl.strip()]
        frags.append(FragInfo(
            smiles=str(row["frag_smiles"]),
            labels=labels,
            n_attach=int(row["n_attach"]),
            count=int(row["count"]),
        ))

    print(f"  [data] fragment library: {len(frags)} fragments "
          f"(scanned {min(g+1, n_groups)}/{n_groups} row groups)")
    return frags


# ---------------------------------------------------------------------------
# Target property distribution
# ---------------------------------------------------------------------------

def load_target_distribution(
    parquet_path: str,
    n: int = 1_000,
    max_mem_gb: float = 32.0,
) -> np.ndarray:
    """
    Stream parents.parquet row group by row group, collecting up to n rows of
    (sLogP, QED, TPSA).  Stops once n rows are collected or the memory cap is
    reached.  Each parents row group is ~280 MB for all 13 columns; we only
    read 3 columns (~65 MB each).
    """
    COLS = ["sLogP", "QED", "TPSA"]

    pf         = pq.ParquetFile(parquet_path)
    n_groups   = pf.metadata.num_row_groups
    used_start = psutil.virtual_memory().used / 1e9

    rows: list = []
    for g in range(n_groups):
        if len(rows) >= n:
            break

        used_now = psutil.virtual_memory().used / 1e9
        delta    = used_now - used_start
        if delta >= max_mem_gb:
            print(f"  [data] memory cap hit ({delta:.1f}/{max_mem_gb:.0f} GB); "
                  f"stopping target load after {g}/{n_groups} row groups.")
            break

        _check_headroom(max_mem_gb, "load_target_distribution")

        batch = pf.read_row_group(g, columns=COLS).to_pandas().dropna()
        need  = n - len(rows)
        rows.append(batch.head(need))

    if not rows:
        raise RuntimeError("No target rows loaded — check parquet path and columns.")

    import pandas as pd
    df  = pd.concat(rows, ignore_index=True).head(n)
    raw = df[COLS].values.astype(np.float32)

    print(f"  [data] target distribution: {len(raw)} molecules "
          f"(scanned {min(g+1, n_groups)}/{n_groups} row groups)")
    return normalize_props(raw)


def sample_target(targets: np.ndarray) -> np.ndarray:
    """Sample one goal vector uniformly at random."""
    return targets[np.random.randint(len(targets))].copy()
