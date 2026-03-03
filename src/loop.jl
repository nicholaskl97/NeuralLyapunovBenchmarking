function run_benchmark(
    dynamics,
    bounds,
    p,
    structure,
    minimization_condition,
    decrease_condition,
    chain,
    strategy,
    opt,
    n,
    fixed_point,
    optimization_args,
    simulation_time,
    endpoint_check,
    ps,
    st,
    log_frequency,
    experiment_name,
    trial_name
)
    println("Beginning $trial_name benchmark...")

    # Construct neural Lyapunov specification
    spec = NeuralLyapunovSpecification(
        structure,
        minimization_condition,
        decrease_condition
    )

    # Run benchmark
    result = benchmark_with_precompile(
        dynamics,
        bounds,
        spec,
        chain,
        strategy,
        opt;
        n,
        fixed_point,
        optimization_args,
        simulation_time,
        endpoint_check,
        ps,
        st,
        log_frequency
    )
    println(
        "$trial_name took $(result.training_time) seconds to train and ",
        "$(result.evaluation_time) seconds to evaluate."
    )

    # Generate time table
    tt = time_table(result.training_time, result.evaluation_time)

    # Print confusion matrix
    cm = result.confusion_matrix
    println(cm)

    # Plot training losses
    loss_plt = plot_losses(result.training_losses, trial_name)

    # Package parameters
    params = (phi = result.phi, θ = result.θ, dynamics, structure, fixed_point, p)

    # Save intermediate results
    sys_name = string(getname(dynamics))
    write_zip(result, cm, tt, params, loss_plt, sys_name, experiment_name, trial_name)

    return nothing
end
