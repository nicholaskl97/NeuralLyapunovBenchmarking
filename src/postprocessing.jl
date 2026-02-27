function plot_losses(result, name)
    # Plot training losses
    return plot(
        result.training_losses.Iteration,
        result.training_losses.Loss .+ eps(Float32);
        xlabel = "Iteration", ylabel = "Loss", yaxis=:log10,
        legend = false,
        title = "$name Training Loss"
    )
end

function split_state_columns(data, state_vars)
    init_states = DataFrame(
        vcat(
            "Initial State" => data[!, "Initial State"],
            [
                "Initial " * var => map(d -> d[i], data[!, "Initial State"])
                for (i, var) in enumerate(state_vars)
            ]
        )
    )
    final_states = DataFrame(
        vcat(
            "Final State" => data[!, "Final State"],
            [
                "Final " * var => map(d -> d[i], data[!, "Final State"])
                for (i, var) in enumerate(state_vars)
            ]
        )
    )
    data = select(innerjoin(final_states, data; on = "Final State"), Not("Final State"))
    data = select(innerjoin(init_states, data; on = "Initial State"), Not("Initial State"))
    return data
end

function write_zip(
    result,
    cm,
    params,
    loss_plt,
    dynamics_name,
    experiment_name,
    trial_name
)
    dir = "results/$experiment_name/$(dynamics_name)"
    mkpath(dir)
    ZipWriter("$dir/$(trial_name).zip") do zip
        zip_newfile(zip, "training_loss.csv")
        CSV.write(zip, result.training_losses)

        zip_newfile(zip, "simulation_data.csv")
        CSV.write(zip, result.data; bom=true)

        zip_newfile(zip, "confusion_matrix.csv")
        CSV.write(zip, cm)

        zip_newfile(zip, "params.dat")
        serialize(zip, params)

        zip_newfile(zip, "training_loss.png")
        png(loss_plt, zip)
    end
    return nothing
end
