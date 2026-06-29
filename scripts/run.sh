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

# Run each experiment
for EXPERIMENT in $PROJDIR/scripts/*; do
    if [ -d $PROJDIR/scripts/$EXPERIMENT ]; then
        # Run each trial
        for TRIAL in $PROJDIR/scripts/$EXPERIMENT/*.jl; do
            # Run the script and redirect output to temporary files
            julia --project=$PROJDIR $PROJDIR/scripts/$EXPERIMENT/$TRIAL.jl 1>$TMPDIR/$TRIAL.out 2>$TMPDIR/$TRIAL.err

            # Move the output files to the results directory, which was created by the script
            mv $TMPDIR/$TRIAL.out $PROJDIR/results/$EXPERIMENT/$TRIAL/$TRIAL.out
            mv $TMPDIR/$TRIAL.err $PROJDIR/results/$EXPERIMENT/$TRIAL/$TRIAL.err
        done
    fi
done
