"""Export the exact fields used by the legacy scaling plot from a local raw M=20 campaign.
Usage: python code/export_legacy_m20.py RAW_DIRECTORY NEW_OUTPUT_DIRECTORY
The output directory must not exist. No simulation or rounding is performed.
"""
from pathlib import Path
import sys
import numpy as np
source, target = map(Path, sys.argv[1:])
target.mkdir(parents=True, exist_ok=False)
keys = ("q", "M_per_pair", "n0", "n1", "delta_pair_global", "r2_global",
        "delta_pair_mean", "delta_pair_std", "sigma_pair_mean", "ns")
for q in (29, 61, 101, 151, 211):
    with np.load(source / f"q{q}_o25.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in keys}
    assert int(arrays["M_per_pair"]) == 20
    np.savez_compressed(target / f"q{q}_o25.npz", **arrays)
