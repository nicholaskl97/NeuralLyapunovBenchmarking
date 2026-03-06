function additive_lyapunov_net_setup(
    dim_hidden,
    hidden_layers,
    dim_out,
    fixed_point,
    control_dim = 0;
    activation = tanh,
    embedding = nothing,
    u_eq = [],
    rng = StableRNG(0),
    gpu = true
)
    return lyapunov_net_setup(
        dim_hidden,
        hidden_layers,
        dim_out,
        fixed_point,
        control_dim,
        activation,
        embedding,
        u_eq,
        rng,
        gpu,
        AdditiveLyapunovNet
    )
end

function multiplicative_lyapunov_net_setup(
    dim_hidden,
    hidden_layers,
    dim_out,
    fixed_point,
    control_dim = 0;
    activation = tanh,
    embedding = nothing,
    u_eq = [],
    rng = StableRNG(0),
    gpu = true
)
    return lyapunov_net_setup(
        dim_hidden,
        hidden_layers,
        dim_out,
        fixed_point,
        control_dim,
        activation,
        embedding,
        u_eq,
        rng,
        gpu,
        MultiplicativeLyapunovNet
    )
end

function lyapunov_net_setup(
    dim_hidden,
    hidden_layers,
    dim_out,
    fixed_point,
    control_dim,
    activation,
    embedding,
    u_eq,
    rng,
    gpu,
    lyapunov_net
)
    dim_in = length(fixed_point)
    dims_hidden = fill(dim_hidden, hidden_layers)

    V = lyapunov_net(
        MLP(
            dim_in,
            Tuple(vcat(dims_hidden, [dim_out])),
            activation
        );
        dim_ϕ = dim_out,
        fixed_point
    )

    if !isnothing(embedding)
        V = Chain(embedding, V)
        structure = NoAdditionalStructure()
    end

    dev = gpu ? gpud : cpud

    if control_dim < 1
        chain = V

        ps, st = Lux.setup(rng, chain)
        ps = ps |> ComponentArray |> dev |> f32
    else
        if !isempty(u_eq) && length(u_eq) != control_dim
            error("u_eq supplied with length $(length(u_eq)) ≠ control_dim = $control_dim")
        end
        idx = isempty(u_eq) ? (1:control_dim) : eachindex(u_eq)

        u = [
            ShiftTo(
                MLP(dim_in, Tuple(vcat(dims_hidden, [1])), activation),
                fixed_point,
                isempty(u_eq) ? [zero(eltype(fixed_point))] : u_eq[i]
            ) for i in idx
        ]

        if !isnothing(embedding)
            u = map(Base.Fix1(Chain, embedding), u)
        end

        chain = vcat(V, u)
        structure = add_policy_search(NoAdditionalStructure(), control_dim)

        ps, st = Lux.setup(rng, chain)
        ps = ps .|> ComponentArray |> dev |> f32
    end

    st = st |> dev |> f32

    minimization_condition = DontCheckNonnegativity(check_fixed_point = false)
    return chain, ps, st, structure, minimization_condition
end
