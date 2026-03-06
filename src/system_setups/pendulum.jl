function pendulum_setup(;
        driven = true, p = Float32[0.5, 1.0], fixed_point = Float32[π * driven, 0.0]
    )
    # Define the System
    name = driven ? :pendulum_driven : :pendulum_undriven
    pendulum = Pendulum(; driven, defaults = p, name)

    # Define the bounds
    θ = unknowns(pendulum)[1]
    ζ, ω0 = p

    bounds = [
        θ ∈ (0, 2.0f0 * π),
        Dt(θ) ∈ (-10, 10) .* ω0,
    ]

    # Define an embedding layer that is periodic with period 2π with respect to θ
    # Note: RNG used doesn't matter since the embedding is deterministic
    periodic_embedding_layer = PeriodicEmbedding([1], Float32[2π])
    ps, st = Lux.setup(default_rng(), periodic_embedding_layer)
    periodic_embedding(x) = first(periodic_embedding_layer(x, ps, st))
    fixed_point_embedded = periodic_embedding(fixed_point)

    periodic_pos_def = let k = 1 ./ (2.0f0 * π, 20ω0)
        (x, x0) -> sum(abs2, periodic_embedding(k .* x) - periodic_embedding(k .* x0))
    end
    endpoint_check = let x0 = copy(fixed_point_embedded)
        (x) -> ≈(periodic_embedding(x), x0, atol = 5.0e-3)
    end

    if driven
        pendulum = mtkcompile(pendulum; inputs = unbound_inputs(pendulum))
    else
        pendulum = mtkcompile(pendulum)
    end

    return pendulum, p, bounds, ζ, ω0, fixed_point, fixed_point_embedded,
        periodic_embedding, periodic_embedding_layer, periodic_pos_def, endpoint_check
end
