#!/usr/bin/env python3
"""Validate T=Theta(n^2): plot iterations-to-converge vs. grid size.

Reads the exp03 CSV produced by scripts/convergence.slurm (one *converged*
run per grid size, itmax raised until dphimax<eps is reached) and plots
iterations vs. n on log-log axes against a Theta(n^2) reference curve.

A straight line of slope 2 on log-log axes confirms T=Theta(n^2); the fitted
slope is printed and shown in the plot title.

Usage:
    python analysis/plot_convergence.py
    python analysis/plot_convergence.py --csv results/convergence.csv \
                                         --outdir results --itmax 3000000

Expected CSV columns (same RESULT format as results/benchmark.csv):
    grid_size,nprocs,idim,kdim,iterations,wall_time_s,comm_time_s,criterion_time_s
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("results/convergence.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--itmax", type=int, default=3_000_000,
                         help="itmax used in convergence.slurm, to flag runs "
                              "that hit the cap without converging")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"convergence CSV not found: {args.csv}\n"
                         f"run scripts/convergence.slurm first (or pass --csv).")

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv).sort_values("grid_size").reset_index(drop=True)

    stuck = df[df["iterations"] >= args.itmax]
    if not stuck.empty:
        print("WARNING: these runs hit itmax without converging "
              "(dphimax never dropped below eps) -- re-run with a higher "
              "--itmax in scripts/convergence.slurm:")
        print(stuck[["grid_size", "nprocs", "iterations"]].to_string(index=False))

    n = df["grid_size"].to_numpy(dtype=float)
    iters = df["iterations"].to_numpy(dtype=float)
    if len(n) < 2:
        raise SystemExit("need >= 2 grid sizes to fit a slope / draw the reference curve")

    # fit log(iters) = slope * log(n) + intercept -> slope ~ 2 confirms Theta(n^2)
    slope, _ = np.polyfit(np.log(n), np.log(iters), 1)
    print(f"fitted slope = {slope:.3f}  (Theta(n^2) predicts 2.0)")

    c = iters[0] / n[0] ** 2  # anchor c*n^2 on the smallest grid
    ref = c * n ** 2

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n, iters, "o-", label="measured")
    ax.plot(n, ref, "k--", alpha=0.5, label=r"$\Theta(n^2)$ reference")
    ax.set(xlabel="grid size (n)", ylabel="iterations to converge",
           title=f"Convergence cost vs. problem size (fitted slope={slope:.2f})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()

    out = args.outdir / "convergence.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
