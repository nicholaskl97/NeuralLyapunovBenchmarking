using NeuralLyapunovBenchmarking, NeuralLyapunov
using NeuralPDE: QuasiRandomTraining
using OptimizationOptimisers: Adam

# Get acrobot-specific variables
dynamics, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
    periodic_embedding_layer, periodic_pos_def, endpoint_check = acrobot_setup(lqr = true);

# Set up neural network
dim_hidden = 25
hidden_layers = 3
dim_out = 10
control_dim = 0
variants = [
    ("AdditiveLyapunovNet", additive_lyapunov_net_setup),
    ("MultiplicativeLyapunovNet", multiplicative_lyapunov_net_setup),
]

# Define optimization parameters
opt = [Adam(0.1), Adam(0.01)]
optimization_args = [:maxiters => 2000]

# Define evaluation parameters
n = 1000
simulation_time = 3.0f4
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
        fixed_point_embedded,
        control_dim;
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

write_summary(dynamics, experiment_name, "Architecture")
