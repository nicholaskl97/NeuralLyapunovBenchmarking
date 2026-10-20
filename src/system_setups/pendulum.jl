function pendulum_setup(;
        driven = true,
        p = Float32[0.5, 1.0],
        fixed_point = Float32[π * driven, 0.0],
        controlled = false
    )
    # Define the System
    name = driven ? :pendulum_driven : :pendulum_undriven
    pendulum = Pendulum(; driven, defaults = p, name)
    if controlled
        if !driven
            @warn "Control is not meaningful for the undriven pendulum. Ignoring driven = false."
        end
        π_cancellation = (x, p, t) -> p[2]^2 * max(-1, min(1, 2 * sin(x[1])))
        @mtkcompile pendulum_controlled = control_pendulum(pendulum, π_cancellation)
        pendulum = pendulum_controlled
    elseif driven
        pendulum = mtkcompile(pendulum; inputs = unbound_inputs(pendulum))
    else
        pendulum = mtkcompile(pendulum)
    end

    states = unknowns(pendulum)
    state_names = last.(split.(string.(getname.(states)), '₊'))
    order = [only(findall(==(name), state_names)) for name in ("θ", "θˍt")]
    state_order = invperm(order)
    θ, ω = states[order]

    # Define the bounds
    ω0 = p[2]
    bounds = [
        θ ∈ (0, 2.0f0 * π),
        ω ∈ (-10, 10) .* ω0,
    ]
    bounds = bounds[state_order]

    # Define periodic embedding
    scales = [2.0f0 * π, 20ω0]
    k = 1 ./ Tuple(scales[state_order])
    fixed_point = fixed_point[state_order]
    periodic_dimensions = findall(==(1), state_order)
    periodic_embedding_layer, periodic_embedding, fixed_point_embedded,
        periodic_pos_def, endpoint_check = angular_embedding_setup(
        periodic_dimensions, k, fixed_point
    )

    return pendulum, p, bounds, fixed_point, fixed_point_embedded, periodic_embedding,
        periodic_embedding_layer, periodic_pos_def, endpoint_check
end
