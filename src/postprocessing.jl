function plot_losses(losses, name)
    # Plot training losses
    return plot(
        losses.Iteration,
        losses.Loss .+ eps(Float32);
        xlabel = "Iteration", ylabel = "Loss", yaxis=:log10,
        legend = false,
        title = "$name Training Loss"
    )
end

function time_table(training_time, eval_time)
    return DataFrame(
        "Stage" => ["Training", "Evaluation"],
        "Time [s]" => [training_time, eval_time]
    )
end

function write_zip(
    result,
    cm,
    tt,
    params,
    loss_plt,
    dynamics_name,
    experiment_name,
    trial_name
)
    dir = "results/$experiment_name/$(dynamics_name)"
    mkpath(dir)
    ZipWriter("$dir/$(trial_name).zip") do zip
        zip_newfile(zip, "confusion_matrix.csv")
        CSV.write(zip, cm)

        zip_newfile(zip, "timing_table.csv")
        CSV.write(zip, tt)

        zip_newfile(zip, "training_loss.csv")
        CSV.write(zip, result.training_losses)

        zip_newfile(zip, "training_loss.png")
        png(loss_plt, zip)

        zip_newfile(zip, "simulation_data.csv")
        CSV.write(zip, result.data; bom=true)

        zip_newfile(zip, "params.dat")
        serialize(zip, params)
    end
    return nothing
end
