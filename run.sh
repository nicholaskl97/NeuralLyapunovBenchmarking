#!/bin/bash

#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2

# Initialize the module command first
source /etc/profile

# Load julia and CUDA
module load local-julia-1.12.5
module load cuda/12.9

# Set environment variables
export LD_LIBRARY_PATH=""
export JULIA_CPU_TARGET="generic;skylake-avx512,clone_all;cascadelake,clone_all"

# Make TMPDIR
export TMPDIR=/state/partition1/user/$USER
mkdir -p "$TMPDIR"

# Set project directory and make results directory
export PROJDIR="$HOME/NeuralLyapunovBenchmarking"
mkdir -p "$PROJDIR/results"

# Set CUDA runtime
julia --project="$PROJDIR" -e 'using LuxCUDA; CUDA.set_runtime_version!(v"12.9"; local_toolkit=true)'

# Run script(s) using make
# Usage examples:
#   make all
#   make decrease_condition
#   make decrease_condition/controlled
#   make sampling_method/quadrotor_planar_lqr
make -C "$PROJDIR" "$@"

# Collect individual trial CSV results into a single CSV file for the experiment
julia --project="$PROJDIR" "$PROJDIR/scripts/collect_csvs.jl" "@"