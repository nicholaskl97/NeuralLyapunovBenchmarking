function double_pendulum_setup(;
    p = Float32[],
    fixed_point = Float32[π * driven, 0, 0, 0],
    driven = true,
    lqr = false
)
    # Define the parameters
    if isempty(p)
        # Assume uniform rods of random mass and length
        m1, m2 = ones(Float32, 2)
        l1, l2 = ones(Float32, 2)
        lc1, lc2 = l1 / 2, l2 / 2
        I1 = m1 * l1^2 / 3
        I2 = m2 * l2^2 / 3
        g = 1.0f0
        b1, b2 = 0.1f0, 0.1f0
        p = Float32[I1, I2, l1, l2, lc1, lc2, m1, m2, g, b1, b2]
    else
        I1, I2, l1, l2, lc1, lc2, m1, m2, g, b1, b2 = p
    end

    # Define the System
    @named double_pendulum = DoublePendulum(; defaults = p)
    if lqr
        if !driven
            @warn "Control is not meaningful for the undriven pendulum. Ignoring driven = false."
        end
        π_lqr = let L = double_pendulum_lqr_matrix(p; fixed_point)
            (x, _, _) -> -L * (x .- fixed_point)
        end
        @mtkcompile double_pendulum_lqr = control_double_pendulum(double_pendulum, π_lqr)
        double_pendulum = double_pendulum_lqr
    elseif driven
        double_pendulum = mtkcompile(
            double_pendulum;
            inputs = unbound_inputs(double_pendulum)
        )
    else
        @mtkcompile double_pendulum_undriven = control_double_pendulum(
            double_pendulum,
            Returns(zeros(Float32, 2))
        )
        double_pendulum = double_pendulum_undriven
    end
    θ2, θ1 = unknowns(double_pendulum)[1:2]

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
    k = 1 ./ (2f0 * π, 2f0 * π, 20ω02, 20ω01)
    fixed_point = fixed_point[[2, 1, 4, 3]]
    periodic_embedding_layer, periodic_embedding, fixed_point_embedded, periodic_pos_def,
        endpoint_check = angular_embedding_setup([1, 2], k, fixed_point)

    return double_pendulum, p, bounds, fixed_point, fixed_point_embedded,
            periodic_embedding, periodic_embedding_layer, periodic_pos_def, endpoint_check
end

function double_pendulum_lqr_matrix(p; fixed_point = Float32[π, 0, 0, 0], Q = I(4), R = I(2))
    return _double_pendulum_lqr_matrix(p, fixed_point, I(2), Q, R)
end

function _double_pendulum_lqr_matrix(p, fixed_point, B, Q, R)
    I1, I2, l1, l2, lc1, lc2, m1, m2, g, b1, b2 = p
    θ1, θ2 = fixed_point[1:2]

    # Assumes linearization around a fixed point
    M = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2) I2 + m2 * l1 * lc2 * cos(θ2);
        I2 + m2 * l1 * lc2 * cos(θ2) I2
    ]
    Jτ_g = [
        -m1 * g * lc1 * cos(θ1) - m2 * g * (l1 * cos(θ1) + lc2 * cos(θ1 + θ2)) -m2 * g * lc2 * cos(θ1 + θ2);
        -m2 * g * lc2 * cos(θ1 + θ2) -m2 * g * lc2 * cos(θ1 + θ2)
    ]
    Jb = diagm([b1, b2])

    A_lin = [
        zeros(2, 2) I(2);
        M \ Jτ_g -M \ Jb
    ]
    B_lin = [zeros(size(B)); M \ B]

    return lqr(Continuous, A_lin, B_lin, Q, R)
end
