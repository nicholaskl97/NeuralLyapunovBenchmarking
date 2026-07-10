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

# Set project directory
export PROJDIR="$HOME/NeuralLyapunovBenchmarking"

# Set CUDA runtime
julia --project="$PROJDIR" -e 'using LuxCUDA; CUDA.set_runtime_version!(v"12.9"; local_toolkit=true)' 1>"$TMPDIR/cuda_runtime.out" 2>"$TMPDIR/cuda_runtime.err"
mkdir -p "$PROJDIR/results"
mv "$TMPDIR/cuda_runtime.out" "$PROJDIR/results/cuda_runtime.out"
mv "$TMPDIR/cuda_runtime.err" "$PROJDIR/results/cuda_runtime.err"

make -C "$PROJDIR" "$@"
