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
mkdir -p $TMPDIR

# Set project directory
export PROJDIR=$HOME/NeuralLyapunovBenchmarking

# Set CUDA runtime
julia --project="$PROJDIR" -e 'using LuxCUDA; CUDA.set_runtime_version!(v"12.9"; local_toolkit=true)' 1>"$TMPDIR/cuda_runtime.out" 2>"$TMPDIR/cuda_runtime.err"
mkdir -p "$PROJDIR/results"
mv "$TMPDIR/cuda_runtime.out" "$PROJDIR/results/cuda_runtime.out"
mv "$TMPDIR/cuda_runtime.err" "$PROJDIR/results/cuda_runtime.err"

# Run each experiment
for EXPERIMENT_DIR in "$PROJDIR/scripts"/*; do
    if [ -d "$EXPERIMENT_DIR" ]; then
        EXPERIMENT=$(basename "$EXPERIMENT_DIR")
        # Run each trial
        for TRIAL_PATH in "$EXPERIMENT_DIR"/*.jl; do
            TRIAL=$(basename "$TRIAL_PATH" .jl)
            # Run the script and redirect output to temporary files
            julia --project="$PROJDIR" "$TRIAL_PATH" 1>"$TMPDIR/$TRIAL.out" 2>"$TMPDIR/$TRIAL.err"

            # Move the output files to the results directory, which was created by the script
            mkdir -p "$PROJDIR/results/$EXPERIMENT/$TRIAL"
            mv "$TMPDIR/$TRIAL.out" "$PROJDIR/results/$EXPERIMENT/$TRIAL/$TRIAL.out"
            mv "$TMPDIR/$TRIAL.err" "$PROJDIR/results/$EXPERIMENT/$TRIAL/$TRIAL.err"
        done
    fi
done
