using NeuralLyapunovBenchmarking, NeuralLyapunov
using NeuralPDE: QuasiRandomTraining
using OptimizationOptimisers: Adam

# Get pendulum-specific variables
dynamics, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
    periodic_embedding_layer, periodic_pos_def, endpoint_check = pendulum_setup();

# Set up neural network
dim_hidden = 10
hidden_layers = 2
dim_out = 3
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
opt = Adam()
optimization_args = [:maxiters => 1000]
strategy = QuasiRandomTraining(1024)

# Define evaluation parameters
n = 10
simulation_time = 3.0f2
log_frequency = 1

# Define decrease conditions
ζ = p[1]
decrease_conditions = [
    ("StabilityISL", StabilityISL()),
    ("ExponentialStability", ExponentialStability(ζ)),
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
