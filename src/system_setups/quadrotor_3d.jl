function quadrotor_3d_setup(;
    p = Float32[],
    fixed_point = zeros(Float32, 12),
    lqr = false,
    speed_control = false
)
    # Define the parameters
    if isempty(p)
        # Assume rotors are negligible mass when calculating the moment of inertia
        m, L = 1.0f0, 1.0f0
        g = 9.81f0
        Ixx = Iyy = m * L^2 / 6
        Izz = m * L^2 / 3
        Ixy = Ixz = Iyz = 0.0f0
        p = [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

        if speed_control
            k_F, k_M = 1.0f0, 1.0f0
            p = vcat(p, [k_F, k_M, L])
        end

        ω0 = sqrt(g / L)
    else
        m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz = p[1:8]

        if speed_control && length(p) >= 11
            k_F, k_M, L = p[9:11]
            ω0 = sqrt(g / L)
        elseif speed_control
            error("When using speed_control, p must contain k_F, k_M, and L.")
        else
            L = sqrt(maximum(abs, [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]) / m)
            ω0 = sqrt(g / sqrt(minimum(abs, filter(!=(0), [Ixx, Ixy, Ixz, Iyy, Iyz, Izz])) / m))
        end
    end

    # Define the System
    @named quadrotor_3d = Quadrotor3D(; defaults = p, speed_control)
    if lqr
        T = m * g
        u_eq = if speed_control
            fill(T / 4 / k_F, 4)
        else
            Float32[T, 0, 0, 0]
        end

        π_lqr = let L = quadrotor_3d_lqr_matrix(p; fixed_point, speed_control), _u_eq = copy(u_eq)
            (x, _, _) -> -L * (x .- fixed_point) + _u_eq
        end

        @mtkcompile quadrotor_3d_lqr = control_quadrotor_3d(quadrotor_3d, π_lqr)
        quadrotor_3d = quadrotor_3d_lqr
        x, y, z, vx, vy, vz, ϕ, θ, ψ, ωϕ, ωθ, ωψ = reverse(unknowns(quadrotor_3d))
        order = reverse([1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12])
    else
        quadrotor_3d = mtkcompile(quadrotor_3d; inputs = unbound_inputs(quadrotor_3d))
        x, y, z, vx, vy, vz, ϕ, θ, ψ, ωϕ, ωθ, ωψ = reverse(unknowns(quadrotor_3d))
        order = reverse([1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12])
    end

    # Define the bounds
    scale = 10.0f0
    dom0 = (-scale / 2, scale / 2)
    v0 = sqrt(L * g)
    bounds = [
        x ∈ dom0 .* 2L,
        y ∈ dom0 .* 2L,
        z ∈ dom0 .* 2L,
        ϕ ∈ (0, 2f0 * π),
        θ ∈ (0, 2f0 * π),
        ψ ∈ (0, 2f0 * π),
        vx ∈ dom0 .* v0,
        vy ∈ dom0 .* v0,
        vz ∈ dom0 .* v0,
        ωϕ ∈ dom0 .* ω0,
        ωθ ∈ dom0 .* ω0,
        ωψ ∈ dom0 .* ω0
    ]
    bounds = bounds[order]

    # Define periodic embedding
    scales = Float32[ω0, ω0, ω0, 0, 0, 0, v0, v0, v0, 2L, 2L, 2L] .* scale
    scales[4:6] .= 2f0 * π
    k = 1 ./ Tuple(scales[order])
    fixed_point = fixed_point[order]
    periodic_embedding_layer, periodic_embedding, fixed_point_embedded,
        periodic_pos_def, endpoint_check = angular_embedding_setup(4:6, k, fixed_point)

    return quadrotor_3d, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
        periodic_embedding_layer, periodic_pos_def, endpoint_check
end

function quadrotor_3d_lqr_matrix(
        p;
        fixed_point = zeros(Float32, 12),
        u_eq = Float32[p[1] * p[2], 0, 0, 0],
        Q = diagm(vcat(fill(10f0, 6), fill(1f0, 6))),
        R = I(4),
        speed_control = false
    )
    @named quad = Quadrotor3D(; speed_control)

    u = unbound_inputs(quad)
    x = setdiff(unknowns(quad), u)
    params = parameters(quad)

    op = Dict(vcat(x .=> fixed_point, u .=> u_eq, params .=> p))

    mats, sys = linearize(quad, u, x; op)

    # Create permutation matrices Px : x_new = Px * x and Pu : u_new = Pu * u
    x_new = unknowns(sys)
    u_new = unbound_inputs(sys)

    Px = Symbolics.value.(x_new .- x') .=== 0
    Pu = Symbolics.value.(u_new .- u') .=== 0

    A = Px' * mats[:A] * Px
    B = Px' * mats[:B] * Pu

    return lqr(Continuous, A, B, Q, R)
end
