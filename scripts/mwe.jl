using NeuralLyapunovBenchmarking, NeuralLyapunov, NeuralPDE
import DifferentiationInterface as DI

# Get 3D-quadrotor-specific variables
m = 0.775f0
g = 9.81f0
Ixx::Float32 = 3 / 2000
Iyy::Float32 = 5 / 2000
Izz::Float32 = 7 / 2000
Ixy = Ixz = Iyz = 0.0f0
L = 0.15f0
p = Float32[m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
dynamics, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
    periodic_embedding_layer, periodic_pos_def, endpoint_check = quadrotor_3d_setup(; p);

# Set up neural network
dim_hidden = 25
hidden_layers = 3
dim_out = 10
control_dim = 4
chain, ps, st, structure, minimization_condition = additive_lyapunov_net_setup(
    dim_hidden,
    hidden_layers,
    dim_out,
    fixed_point_embedded,
    control_dim;
    embedding = periodic_embedding_layer,
    u_eq = Float32[m * g, 0, 0, 0],
    gpu = true
);

# Define optimization parameters
strategy = QuasiRandomTraining(1024)
decrease_condition = StabilityISL()

spec = NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)

@named pde_system = NeuralLyapunovPDESystem(
    dynamics,
    bounds,
    spec;
    fixed_point
)

discretization = PhysicsInformedNN(chain, strategy; init_params = ps, init_states = st)
opt_prob = discretize(pde_system, discretization)

f = Base.Fix2(opt_prob.f.f, SciMLBase.NullParameters())
u0 = opt_prob.u0

f(u0)

ad = ARGS[1]

if ad == "Zygote"
    import Zygote
    backend = DI.AutoZygote()
elseif ad == "Mooncake"
    import Mooncake
    backend = DI.AutoMooncake()
elseif ad == "Enzyme"
    import Enzyme
    backend = DI.AutoEnzyme()
elseif ad == "ForwardDiff"
    import ForwardDiff
    backend = DI.AutoForwardDiff()
else
    error("Unsupported AD backend")
end

∇ = DI.gradient(f, backend, u0)
@show ∇
