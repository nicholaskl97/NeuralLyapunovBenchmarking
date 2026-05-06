using NeuralLyapunovBenchmarking
using NeuralLyapunov: StabilityISL, ExponentialStability, AsymptoticStability
using NeuralPDE: QuasiRandomTraining
using OptimizationOptimisers: Adam

# Get planar-quadrotor-specific variables
dynamics, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
    periodic_embedding_layer, periodic_pos_def, endpoint_check = quadrotor_planar_setup();

# Set up neural network
dim_hidden = 25
hidden_layers = 3
dim_out = 10
control_dim = 2
m, I_quad, g, r = p
chain, ps, st, structure, minimization_condition = additive_lyapunov_net_setup(
    dim_hidden,
    hidden_layers,
    dim_out,
    fixed_point_embedded,
    control_dim;
    embedding = periodic_embedding_layer,
    u_eq = fill(m * g / 2, control_dim)
);

# Define optimization parameters
opt = [Adam(0.1), Adam(0.01), Adam(0.001)]
optimization_args = [[:maxiters => 500], [:maxiters => 1000], [:maxiters => 1000]]
strategy = QuasiRandomTraining(1024)

# Define evaluation parameters
n = 10
simulation_time = 3.0f3
log_frequency = 1

# Define decrease conditions
k = sqrt(sqrt(g / r))
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
