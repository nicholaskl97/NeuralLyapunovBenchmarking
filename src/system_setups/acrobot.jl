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
    θ1, θ2 = unknowns(acrobot)[1:2]
    acrobot = mtkcompile(acrobot; inputs = unbound_inputs(acrobot))

    # Define the bounds
    ω01 = sqrt(m1 * g * lc1 / I1)
    ω02 = sqrt(m2 * g * lc2 / I2)
    bounds = [
        θ1 ∈ (0, 2f0 * π),
        θ2 ∈ (0, 2f0 * π),
        Dt(θ1) ∈ (-10, 10) .* ω01,
        Dt(θ2) ∈ (-10, 10) .* ω02
    ]

    # Define periodic embedding
    k = 1 ./ (2f0 * π, 2f0 * π, 20ω01, 20ω02)
    periodic_embedding_layer, periodic_embedding, fixed_point_embedded, periodic_pos_def,
        endpoint_check = angular_embedding_setup([1, 2], k, fixed_point)

    return acrobot, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
        periodic_embedding_layer, periodic_pos_def, endpoint_check
end
