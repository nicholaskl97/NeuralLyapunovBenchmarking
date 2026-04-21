using NeuralLyapunovBenchmarking
using NeuralLyapunov: StabilityISL
using NeuralPDE: QuasiRandomTraining, StochasticTraining, GridTraining, QuadratureTraining
using OptimizationOptimisers: Adam

# Get pendulum-specific variables
dynamics, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
    periodic_embedding_layer, periodic_pos_def, endpoint_check = pendulum_setup(controlled = true);

# Set up neural network
dim_hidden = 10
hidden_layers = 2
dim_out = 3
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
opt = Adam()
optimization_args = [:maxiters => 1000]

# Define evaluation parameters
n = 1000
simulation_time = 3.0f2
log_frequency = 1

# Define decrease condition
decrease_condition = StabilityISL()

# Define discretization strategy
root_N = 32
N = root_N^2
grid_spacings = Float32[
    (bound.domain.right - bound.domain.left) / (root_N - 1) for bound in bounds
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
