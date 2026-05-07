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
simulation_time = 3.0f4
log_frequency = 1

# Define decrease conditions
b1, b2 = p[10:11]
k = min(b1, b2)
relu(x) = max(zero(x), x)
softplus(x) = max(zero(x), x) + log1p(exp(-abs(x)))
squareplus(x) = max(zero(x), x) + one(x) / (abs(x) + sqrt(abs2(x) + 2))

rectifiers = [
    ("relu", relu),
    ("softplus", softplus),
    ("squareplus", squareplus)
]
decrease_conditions = reduce(
    vcat,
    [
        ("StabilityISL - $(name)", StabilityISL(; rectifier)),
        ("ExponentialStability - $(name)", ExponentialStability(k; rectifier)),
        ("AsymptoticStability - $(name)", AsymptoticStability(strength = periodic_pos_def; rectifier))
    ]
    for (name, rectifier) in rectifiers
)

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

write_summary(dynamics, experiment_name, "Decrease Condition - Rectifier")
