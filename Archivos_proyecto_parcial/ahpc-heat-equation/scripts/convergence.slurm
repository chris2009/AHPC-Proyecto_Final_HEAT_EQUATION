#!/usr/bin/env bash
#SBATCH --job-name=heat-conv
#SBATCH --output=results/slurm-conv-%j.out
#SBATCH --partition=standard
#SBATCH --nodelist=n003
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
# exp03: validate T=Theta(n^2) -- ONE run per grid size, with itmax raised
# high enough that the solver stops on dphimax<eps (real convergence)
# instead of hitting the itmax cap, as it did in exp01/exp02 (itmax=20000,
# all 20 (n,p) combinations reported iterations=20001, i.e. none converged).
#
# p=16 fixed for every n (same single-node layout/placement as exp01): the
# iteration count to converge is a property of the numerics and the global
# MPI_Allreduce stopping criterion, NOT of p, so any p gives the same
# "iterations". p=16 just gives the fastest wall-clock for the largest grid
# (n=1024, ~2.3M predicted iterations from the c*n^2 fit in docs/pram-y-complejidad.md).
#
# Submit with:   sbatch scripts/convergence.slurm
#
# Output: results/convergence.csv, one row per grid size, same RESULT format
# as results/benchmark.csv (grid_size,nprocs,idim,kdim,iterations,wall_time_s,
# comm_time_s,criterion_time_s). Plot it with analysis/plot_convergence.py.
# ----------------------------------------------------------------------------
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

NS=(128 256 512 1024)   # grid sizes (imax=kmax=N)
P=16                     # fixed process count (see note above)
ITMAX=3000000            # generous cap (~1.3x the n=1024 prediction); solver
                         # stops earlier via dphimax<eps if it converges sooner
BIN="bin/heat-mpi-big"
CSV="results/convergence.csv"
RAW="results/raw_convergence"

module load gnu12/12.4.0
module load openmpi4/4.1.6

make mpi
mkdir -p "$RAW"

# CSV header (write once)
if [[ ! -f "$CSV" ]]; then
  echo "grid_size,nprocs,idim,kdim,iterations,wall_time_s,comm_time_s,criterion_time_s" > "$CSV"
fi

for N in "${NS[@]}"; do
  OUT="$RAW/run_n${N}_p${P}.out"
  echo ">> N=$N P=$P (itmax=$ITMAX)"
  if mpirun --map-by socket --bind-to core -np "$P" "$BIN" "$N" "$ITMAX" 0 | tee "$OUT"; then
    scripts/collect_results.sh "$OUT" >> "$CSV"
  else
    echo "!! run N=$N FAILED, skipping" >&2
  fi
done

echo ">> wrote $CSV"
