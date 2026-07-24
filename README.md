# NeuralLyapunovBenchmarking

This repository benchmarks different ways of training neural Lyapunov functions for control and stability problems. The local Julia package in [src](src) is implemented in [src/NeuralLyapunovBenchmarking.jl](src/NeuralLyapunovBenchmarking.jl) and is used to run experiment scripts that exercise the NeuralLyapunov ecosystem.

The work is built on [NeuralLyapunov.jl](https://docs.sciml.ai/NeuralLyapunov/) and uses systems from [NeuralLyapunovProblemLibrary.jl](https://docs.sciml.ai/NeuralLyapunov/dev/lib/). The benchmarked systems include:

- the single pendulum, either unactuated or fully actuated,
- the double pendulum in unactuated, fully actuated, and acrobot configurations,
- a planar approximation of a quadrotor, and
- a full 3D quadrotor model.

For controlled systems, the experiments compare either evaluating a classical controller (typically an LQR controller) using a neural Lyapunov certificate, or jointly learning a neural Lyapunov function and a neural policy.

## Experiment overview

### 1. Decrease Condition (`decrease_condition`)
Tests different ways of setting up the Lyapunov decrease condition, analogous to the conditions for stability in the sense of Lyapunov, asymptotic stability, and exponential stability. This experiment also examines the effect of the choice of rectifier to turn the Lyapunov condition's partial differential inequality (PDI) into a PDE for NeuralPDE.jl.

### 2. Lyapunov-Net Variants (`lyapunov-net_variants`)
Tests variants of the Lyapunov-Net architecture, comparing the traditional additive form with a newer multiplicative variant.

### 3. Sampling Method (`sampling_method`)
Tests different training sampling strategies: sampling on a grid, quasirandom sampling, and true stochastic sampling.

## Trial groups

The experiment scripts are grouped into three trial families:

- undriven: `pendulum_undriven`, `double_pendulum_undriven`
- controlled: `pendulum_controlled`, `double_pendulum_lqr`, `acrobot_lqr`, `quadrotor_planar_lqr`, `quadrotor_3d_lqr`
- neural_policy_search: `pendulum_driven`, `double_pendulum`, `acrobot`, `quadrotor_planar`, `quadrotor_3d`

These groups are used by the Makefile targets below to run experiments in a consistent way.

## Repository layout

- [src](src): the local Julia package implementation for the benchmark workflow.
- [Project.toml](Project.toml) and [Manifest.toml](Manifest.toml): Julia project and dependency manifests for the local package.
- [scripts](scripts): shared helper scripts used by the workflow.
  - [scripts/collect_csvs.jl](scripts/collect_csvs.jl): aggregates per-trial summaries into experiment-level summaries.
  - [scripts/generate_sampling_method_plot.jl](scripts/generate_sampling_method_plot.jl): creates a scatter plot for results of the Sampling Method study.
- [decrease_condition](decrease_condition): experiment-specific scripts and results for the Decrease Condition study.
- [sampling_method](sampling_method): experiment-specific scripts and results for the Sampling Method study.
- [lyapunov-net_variants](lyapunov-net_variants): experiment-specific scripts and results for the Lyapunov-Net architecture study.

## Script usage

### Shared helper scripts

Aggregate trial-level CSV files into experiment-level summaries:

```bash
julia --project=. scripts/collect_csvs.jl all
```

You can also target a single experiment or trial group:

```bash
julia --project=. scripts/collect_csvs.jl decrease_condition
julia --project=. scripts/collect_csvs.jl decrease_condition/controlled
```

Generate a plot for a sampling-method summary CSV:

```bash
julia --project=. scripts/generate_sampling_method_plot.jl path/to/summary.csv
```

### Experiment-specific scripts

Each experiment folder contains Julia scripts for the corresponding trials. For example:

```bash
julia --project=. decrease_condition/scripts/acrobot_lqr.jl
julia --project=. sampling_method/scripts/quadrotor_planar_lqr.jl
julia --project=. lyapunov-net_variants/scripts/pendulum_undriven.jl
```

The experiment scripts write their outputs into the corresponding experiment's results directory, such as:

- [decrease_condition/results](decrease_condition/results)
- [sampling_method/results](sampling_method/results)
- [lyapunov-net_variants/results](lyapunov-net_variants/results)

## Using make

The repository's Makefile exposes targets for each experiment and trial group.

Example usage:

```bash
make all
make decrease_condition
make decrease_condition/controlled
make sampling_method/quadrotor_planar_lqr
```

The Makefile automatically:

- finds the relevant experiment scripts,
- creates the appropriate results directory for each trial, and
- writes stdout/stderr into per-trial output files under that experiment's results tree.

## Running with the cluster script

The helper script [run.sh](run.sh) is intended for execution on MIT's SuperCloud cluster. It sets up the environment, creates experiment result directories, runs the requested Makefile target, and then calls the CSV aggregation script.
