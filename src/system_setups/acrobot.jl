function acrobot_setup(; p = Float32[], fixed_point = Float32[π, π, 0.0, 0.0])
    # Define the parameters
    if isempty(p)
        # Assume uniform rods of random mass and length
        m1, m2 = ones(Float32, 2)
        l1, l2 = ones(Float32, 2)
        lc1, lc2 = l1 / 2, l2 / 2
        I1 = m1 * l1^2 / 3
        I2 = m2 * l2^2 / 3
        g = 1.0f0
        p = Float32[I1, I2, l1, l2, lc1, lc2, m1, m2, g]
    else
        I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
    end

    # Define the System
    @named acrobot = Acrobot(; defaults = p)

    # Define the bounds
    x = get_double_pendulum_state_symbols(acrobot)
    θ1, θ2 = unknowns(acrobot)[1:2]

    ω01 = sqrt(m1 * g * lc1 / I1)
    ω02 = sqrt(m2 * g * lc2 / I2)

    bounds = [
        θ1 ∈ (0, 2f0 * π),
        θ2 ∈ (0, 2f0 * π),
        Dt(θ1) ∈ (-10, 10) .* ω01,
        Dt(θ2) ∈ (-10, 10) .* ω02
    ]

    # Define an embedding layer that is periodic with period 2π with respect to θ
    # Note: RNG used doesn't matter since the embedding is deterministic
    periodic_embedding_layer = PeriodicEmbedding([1, 2], Float32[2π, 2π])
    ps, st = Lux.setup(default_rng(), periodic_embedding_layer)
    periodic_embedding(x) = first(periodic_embedding_layer(x, ps, st))
    fixed_point_embedded = periodic_embedding(fixed_point)

    periodic_pos_def = let k = 1 ./ (2f0 * π, 2f0 * π, 20ω01, 20ω02)
        function (state, fixed_point)
            return sum(
                abs2,
                periodic_embedding(k .* state) - periodic_embedding(k .* fixed_point)
            )
        end
    end
    endpoint_check = let x0 = copy(fixed_point_embedded)
        (x) -> ≈(periodic_embedding(x), x0, atol = 5e-3)
    end

    return acrobot, p, bounds, ω01, ω02, fixed_point, fixed_point_embedded,
            periodic_embedding, periodic_embedding_layer, periodic_pos_def, endpoint_check
end

acrobot_state_vars() = ["θ2", "θ1", "ω2", "ω1"]
