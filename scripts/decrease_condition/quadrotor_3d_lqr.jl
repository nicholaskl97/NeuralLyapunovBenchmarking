using NeuralLyapunovBenchmarking
using NeuralLyapunov: StabilityISL, ExponentialStability, AsymptoticStability
using NeuralPDE: QuasiRandomTraining
using OptimizationOptimisers: Adam
using ModelingToolkit: EnsembleSerial

# Get 3D-quadrotor-specific variables
m = 0.775f0
g = 9.81f0
Ixx::Float32 = 3 / 2000
Iyy::Float32 = 5 / 2000
Izz::Float32 = 7 / 2000
Ixy = Ixz = Iyz = 0.0f0
k_F = 1.0f0
k_M::Float32 = 49 / 2000
L = 0.15f0
p = Float32[m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz, k_F, k_M, L]
dynamics, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
    periodic_embedding_layer, periodic_pos_def, endpoint_check = quadrotor_3d_setup(;
        p, lqr = true, speed_control = true
);

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
opt = [Adam(0.1), Adam(0.01), Adam(0.001)]
optimization_args = [[:maxiters => 500], [:maxiters => 1000], [:maxiters => 1000]]
strategy = QuasiRandomTraining(1024)

# Define evaluation parameters
n = 100
simulation_time = 3.0f3
log_frequency = 1

# Define decrease conditions
k = sqrt(sqrt(g / sqrt(minimum(abs, [Ixx, Iyy, Izz]) / m)))
relu(x) = max(zero(x), x)
softplus(x) = max(zero(x), x) + log1p(exp(-abs(x)))
squareplus(x) = max(zero(x), x) + one(x) / (abs(x) + sqrt(abs2(x) + 2))

rectifiers = [
    ("relu", relu),
    ("softplus", softplus),
#    ("squareplus", squareplus)
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
        trial_name,
        EnsembleSerial()
    )
end

write_summary(dynamics, experiment_name, "Decrease Condition - Rectifier")
