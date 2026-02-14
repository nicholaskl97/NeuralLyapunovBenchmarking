using NeuralLyapunovBenchmarking, NeuralLyapunov
using NeuralPDE: QuasiRandomTraining
using OptimizationOptimisers: Adam

# Get acrobot-specific variables
dynamics, p, bounds, ω01, ω02, fixed_point, fixed_point_embedded, periodic_embedding,
    periodic_embedding_layer, periodic_pos_def, endpoint_check = acrobot_setup();

# Set up neural network
dim_hidden = 25
hidden_layers = 3
dim_out = 10
control_dim = 1
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
strategy = QuasiRandomTraining(1024)

# Define evaluation parameters
n = 10
simulation_time = 3.0f3
log_frequency = 1

# Define decrease conditions
decrease_conditions = [
    ("StabilityISL", StabilityISL()),
    ("ExponentialStability", ExponentialStability(sqrt(ω01 * ω02))),
    ("AsymptoticStability", AsymptoticStability(strength = periodic_pos_def)),
];

#################################### Run the benchmarks ####################################
experiment_name = "decrease_condition"
for (trial_name, decrease_condition) in decrease_conditions
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
