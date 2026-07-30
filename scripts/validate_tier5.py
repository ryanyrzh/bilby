#!/usr/bin/env python
"""Validate Tier 5 minimal SNR grid smoke-test outputs."""
import glob
import math
import os
import sys

import numpy as np


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "output_test"
    plotdir = sys.argv[2] if len(sys.argv) > 2 else "plots_test"

    npz_files = glob.glob(os.path.join(outdir, "*smoke*.npz"))
    if not npz_files:
        npz_files = glob.glob(os.path.join(outdir, "result_Mc2_R2_agn_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ in {outdir}")

    pdf_files = glob.glob(os.path.join(plotdir, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF in {plotdir}")

    data = np.load(npz_files[0])
    for key in ("snr_1", "snr_2"):
        if key not in data:
            raise KeyError(f"Missing {key} in {npz_files[0]}")
        arr = data[key]
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} has non-finite values")
        if not np.all(arr > 0):
            raise ValueError(f"{key} has non-positive values")

    print(f"PASS: {npz_files[0]}")
    print(f"PASS: {pdf_files[0]}")
    print("Tier 5 validation: ALL PASS")


if __name__ == "__main__":
    main()
