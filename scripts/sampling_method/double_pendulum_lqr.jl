using NeuralLyapunovBenchmarking, NeuralLyapunov
using NeuralPDE: QuasiRandomTraining, StochasticTraining, GridTraining, QuadratureTraining
using OptimizationOptimisers: Adam

# Get double-pendulum-specific variables
dynamics, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
    periodic_embedding_layer, periodic_pos_def, endpoint_check = double_pendulum_setup(lqr = true);

# Set up neural network
dim_hidden = 25
hidden_layers = 3
dim_out = 10
control_dim = 0
chain, ps, st, structure, minimization_condition = additive_lyapunov_net_setup(
    dim_hidden,
    hidden_layers,
    dim_out,
    fixed_point_embedded,
    control_dim;
    embedding = periodic_embedding_layer
);

# Define optimization parameters
opt = [Adam(0.1), Adam(0.01)]
optimization_args = [:maxiters => 2000]

# Define evaluation parameters
n = 1000
simulation_time = 3.0f3
log_frequency = 1

# Define decrease condition
decrease_condition = StabilityISL()

# Define discretization strategy
root4_N = 8
N = root4_N^4
grid_spacings = Float32[
    (bound.domain.right - bound.domain.left) / (root4_N - 1) for bound in bounds
]

strategies = [
    ("QuasiRandomTraining", QuasiRandomTraining(N)),
    ("StochasticTraining", StochasticTraining(N)),
    ("GridTraining", GridTraining(grid_spacings)),
#    ("QuadratureTraining", QuadratureTraining(; maxiters = 18)),
]

#################################### Run the benchmarks ####################################
experiment_name = "sampling_method"
for (trial_name, strategy) in strategies
    run_benchmark(
        dynamics,
        bounds,
        p,
        structure,
        minimization_condition,
        decrease_condition,
        chain,
        strategy,
        opt,
        n,
        fixed_point,
        optimization_args,
        simulation_time,
        endpoint_check,
        ps,
        st,
        log_frequency,
        experiment_name,
        trial_name
    )
end

write_summary(dynamics, experiment_name, "Sampling Method")
