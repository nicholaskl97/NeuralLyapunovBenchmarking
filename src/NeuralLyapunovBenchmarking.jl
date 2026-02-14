module NeuralLyapunovBenchmarking

using NeuralLyapunov, NeuralPDE, NeuralLyapunovProblemLibrary, ModelingToolkit, Plots
using ModelingToolkit: t_nounits as t, D_nounits as Dt, getname
using Lux, LuxCUDA, ComponentArrays
using Boltz.Layers: PeriodicEmbedding, MLP, ShiftTo
using Random
using Random: default_rng
using StableRNGs
using Serialization, CSV
using ZipArchives: ZipWriter, zip_newfile
using DataFrames

include("system_setups/acrobot.jl")

export acrobot_setup

const cpud = cpu_device()
const gpud = gpu_device()

include("network_setup.jl")

export additive_lyapunov_net_setup, multiplicative_lyapunov_net_setup

include("precompile.jl")

export benchmark_with_precompile

include("postprocessing.jl")

export plot_losses, split_state_columns

include("loop.jl")

export run_benchmark

end
