function quadrotor_planar_setup(; p = Float32[], fixed_point = zeros(Float32, 6))
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
    x, y, θ = unknowns(quadrotor_planar)[1:3]
    quadrotor_planar = mtkcompile(quadrotor_planar; inputs = unbound_inputs(quadrotor_planar))

    # Define the bounds
    scale = 10.0f0
    dom0 = (-scale / 2, scale / 2)
    v0 = sqrt(r * g)
    ω0 = sqrt(g / r)
    bounds = [
        x ∈ dom0 .* 2r,
        y ∈ dom0 .* 2r,
        θ ∈ (0, 2f0 * π),
        Dt(x) ∈ dom0 .* v0,
        Dt(y) ∈ dom0 .* v0,
        Dt(θ) ∈ dom0 .* ω0
    ]

    # Define periodic embedding
    k = 1 ./ (2r, 2r, 2f0 * π, v0, v0, ω0) ./ scale
    periodic_embedding_layer, periodic_embedding, fixed_point_embedded,
        periodic_pos_def, endpoint_check = angular_embedding_setup([3], k, fixed_point)

    return quadrotor_planar, p, bounds, fixed_point, fixed_point_embedded,
            periodic_embedding, periodic_embedding_layer, periodic_pos_def, endpoint_check
end
