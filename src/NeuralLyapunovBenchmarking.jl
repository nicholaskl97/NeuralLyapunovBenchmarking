module NeuralLyapunovBenchmarking

using NeuralLyapunov, NeuralPDE, NeuralLyapunovProblemLibrary, ModelingToolkit, Plots
using ModelingToolkit: t_nounits as t, D_nounits as Dt, getname, unbound_inputs
using ControlSystemsBase: lqr, Continuous
using Lux, LuxCUDA, ComponentArrays
using Boltz.Layers: PeriodicEmbedding, MLP, ShiftTo
using Random
using Random: default_rng
using LinearAlgebra: I, diagm
using StableRNGs
using Serialization, CSV
using ZipArchives: ZipWriter, zip_newfile, ZipReader, zip_readentry
using DataFrames

# Include the system setups
include("system_setups/acrobot.jl")
include("system_setups/double_pendulum.jl")
include("system_setups/pendulum.jl")
include("system_setups/quadrotor_planar.jl")
include("system_setups/quadrotor_3d.jl")

export acrobot_setup, pendulum_setup, double_pendulum_setup, quadrotor_planar_setup,
    quadrotor_3d_setup

const cpud = cpu_device()
const gpud = gpu_device()

include("network_setup.jl")

export additive_lyapunov_net_setup, multiplicative_lyapunov_net_setup

include("precompile.jl")

export benchmark_with_precompile

include("postprocessing.jl")

export plot_losses, write_summary

include("loop.jl")

export run_benchmark

end
