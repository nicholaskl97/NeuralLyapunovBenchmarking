using NeuralLyapunovBenchmarking, NeuralLyapunov
using NeuralPDE: QuasiRandomTraining
using OptimizationOptimisers: Adam

# Get pendulum-specific variables
dynamics, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
    periodic_embedding_layer, periodic_pos_def, endpoint_check = pendulum_setup(driven = false);

# Set up neural network
dim_hidden = 10
hidden_layers = 2
dim_out = 3
variants = [
    ("AdditiveLyapunovNet", additive_lyapunov_net_setup),
    ("MultiplicativeLyapunovNet", multiplicative_lyapunov_net_setup),
]

# Define optimization parameters
opt = Adam()
optimization_args = [:maxiters => 1000]

# Define evaluation parameters
n = 1000
simulation_time = 3.0f2
log_frequency = 1

# Define decrease conditions
decrease_condition = StabilityISL()

# Define discretization strategy
N = 1024
strategy = QuasiRandomTraining(N)

#################################### Run the benchmarks ####################################
experiment_name = "lyapunov-net_variants"
for (trial_name, setup) in variants
    chain, ps, st, structure, minimization_condition = setup(
        dim_hidden,
        hidden_layers,
        dim_out,
        fixed_point_embedded;
        embedding = periodic_embedding_layer
    );

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
