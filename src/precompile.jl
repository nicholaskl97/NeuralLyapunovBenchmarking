function benchmark_with_precompile(
        dynamics::System,
        bounds,
        spec,
        chain,
        strategy,
        opt;
        n,
        fixed_point = nothing,
        optimization_args = [],
        simulation_time,
        endpoint_check = nothing,
        ps,
        st,
        ensemble_alg = EnsembleDistributed(),
        log_frequency = 10
    )
    # Precompilation run
    println("Precompilation run...")
    result = benchmark(
        dynamics,
        bounds,
        spec,
        chain,
        QuasiRandomTraining(8),
        opt;
        fixed_point,
        simulation_time = 1f0,
        n = 3,
        optimization_args = [:maxiters => 5],
        endpoint_check = Returns(true),
        init_params = ps,
        init_states = st,
        ensemble_alg,
        log_frequency = 1
    )

    # Seed the random number generators for reproducibility
    rng = StableRNG(0)
    Random.seed!(200)

    # Run benchmark
    println("Running benchmark...")
    result = benchmark(
        dynamics,
        bounds,
        spec,
        chain,
        strategy,
        opt;
        fixed_point,
        simulation_time,
        n,
        optimization_args,
        endpoint_check,
        rng,
        init_params = ps,
        init_states = st,
        ensemble_alg,
        log_frequency
    )

end
