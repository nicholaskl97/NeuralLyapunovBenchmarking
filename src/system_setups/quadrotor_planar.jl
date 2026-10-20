function quadrotor_planar_setup(; p = Float32[], fixed_point = zeros(Float32, 6), lqr = false)
    # Define the parameters
    if isempty(p)
        # Assume rotors are negligible mass when calculating the moment of inertia
        m = 1.0f0
        r = 1.0f-1
        g = 9.81f0
        I_quad = m * r^2 / 12
        p = Float32[m, I_quad, g, r]
    else
        m, I_quad, g, r = p
    end

    # Define the System
    @named quadrotor_planar = QuadrotorPlanar(; defaults = p)
    if lqr
        π_lqr = let L = quadrotor_planar_lqr_matrix(p), x0 = copy(fixed_point)
            (x, _, _) -> -L * (x .- x0) .+ m * g / 2
        end
        @mtkcompile quadrotor_planar_lqr = control_quadrotor_planar(quadrotor_planar, π_lqr)
        quadrotor_planar = quadrotor_planar_lqr
    else
        quadrotor_planar = mtkcompile(
            quadrotor_planar;
            inputs = unbound_inputs(quadrotor_planar)
        )
    end
    states = unknowns(quadrotor_planar)
    state_names = last.(split.(string.(getname.(states)), '₊'))
    order = [
        only(findall(==(name), state_names)) for
        name in ("x", "y", "θ", "xˍt", "yˍt", "θˍt")
    ]
    state_order = invperm(order)
    x, y, θ, vx, vy, ω = states[order]

    # Define the bounds
    scale = 10.0f0
    dom0 = (-scale / 2, scale / 2)
    v0 = sqrt(r * g)
    ω0 = sqrt(g / r)
    bounds = [
        x ∈ dom0 .* 2r,
        y ∈ dom0 .* 2r,
        θ ∈ (0, 2f0 * π),
        vx ∈ dom0 .* v0,
        vy ∈ dom0 .* v0,
        ω ∈ dom0 .* ω0
    ]
    bounds = bounds[state_order]

    # Define periodic embedding
    scales = [2r, 2r, 2f0 * π, v0, v0, ω0] .* scale
    k = 1 ./ Tuple(scales[state_order])
    fixed_point = fixed_point[state_order]
    periodic_dimensions = findall(==(3), state_order)
    periodic_embedding_layer, periodic_embedding, fixed_point_embedded,
        periodic_pos_def, endpoint_check = angular_embedding_setup(
        periodic_dimensions, k, fixed_point
    )

    return quadrotor_planar, p, bounds, fixed_point, fixed_point_embedded,
            periodic_embedding, periodic_embedding_layer, periodic_pos_def, endpoint_check
end

function quadrotor_planar_lqr_matrix(p; Q = I(6), R = I(2))
    m, I_quad, g, r = p

    # Assumes linearization around a fixed point
    # x_eq = (x*, y*, 0, 0, 0, 0), u_eq = (mg / 2, mg / 2)
    A_lin = zeros(6, 6)
    A_lin[1:3, 4:6] .= I(3)
    A_lin[4, 3] = -g

    B_lin = zeros(6, 2)
    B_lin[5, :] .= 1 / m
    B_lin[6, :] .= r / I_quad, -r / I_quad

    return lqr(Continuous, A_lin, B_lin, Q, R)
end
