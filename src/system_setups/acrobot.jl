function acrobot_setup(; p = Float32[], fixed_point = Float32[π, 0, 0, 0], lqr = false)
    # Define the parameters
    if isempty(p)
        # Assume uniform rods of random mass and length
        m1, m2 = ones(Float32, 2)
        l1, l2 = 1.0f0, 2.0f0
        lc1, lc2 = l1 / 2, l2 / 2
        I1 = m1 * l1^2 / 3
        I2 = m2 * l2^2 / 3
        g = 1.0f0
        b1, b2 = 0.1f0, 0.1f0
        p = Float32[I1, I2, l1, l2, lc1, lc2, m1, m2, g, b1, b2]
    else
        I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
    end

    # Define the System
    @named acrobot = Acrobot(; defaults = p)
    if lqr
        π_lqr = let L = acrobot_lqr_matrix(p; fixed_point)
            (x, _, _) -> -L * (x .- fixed_point)
        end
        @mtkcompile acrobot_lqr = control_double_pendulum(acrobot, π_lqr)
        acrobot = acrobot_lqr
        θ1, θ2, ω2, ω1 = unknowns(acrobot)
        order = [1, 2, 4, 3]
    else
        acrobot = mtkcompile(acrobot; inputs = unbound_inputs(acrobot))
        θ2, θ1 = unknowns(acrobot)[1:2]
        ω1, ω2 = Dt(θ1), Dt(θ2)
        order = [2, 1, 4, 3]
    end

    # Define the bounds
    ω01 = sqrt(m1 * g * lc1 / I1)
    ω02 = sqrt(m2 * g * lc2 / I2)
    scale = 10.0f0
    bounds = [
        θ1 ∈ (0, 2.0f0 * π),
        θ2 ∈ (0, 2.0f0 * π),
        ω1 ∈ (-scale / 2, scale / 2) .* ω01,
        ω2 ∈ (-scale / 2, scale / 2) .* ω02,
    ]
    bounds = bounds[order]

    # Define periodic embedding
    scales = [2.0f0 * π, 2.0f0 * π, scale * ω01, scale * ω02]
    k = 1 ./ Tuple(scales[order])
    fixed_point = fixed_point[order]
    periodic_embedding_layer, periodic_embedding, fixed_point_embedded, periodic_pos_def,
        endpoint_check = angular_embedding_setup([1, 2], k, fixed_point)

    return acrobot, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
        periodic_embedding_layer, periodic_pos_def, endpoint_check
end

function acrobot_lqr_matrix(
        p; fixed_point = Float32[π, 0, 0, 0], Q = diagm(Float32[10, 10, 1, 1]), R = I(1)
    )
    return _double_pendulum_lqr_matrix(p, fixed_point, [0, 1], Q, R)
end
