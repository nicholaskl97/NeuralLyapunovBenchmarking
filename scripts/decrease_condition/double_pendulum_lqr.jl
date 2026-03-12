using NeuralLyapunovBenchmarking, NeuralLyapunov
using NeuralPDE: QuasiRandomTraining
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
strategy = QuasiRandomTraining(1024)

# Define evaluation parameters
n = 1000
simulation_time = 3.0f3
log_frequency = 1

# Define decrease conditions
I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
ω0 = sqrt(g * min(m1 * lc1 / I1, m2 * lc2 / I2))
decrease_conditions = [
    ("StabilityISL", StabilityISL()),
    ("ExponentialStability", ExponentialStability(ω0)),
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
