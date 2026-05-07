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

function write_summary(dynamics, experiment_name, trial_category_name = experiment_name)
    # Make empty DataFrame
    trial_category_names = split(trial_category_name, " - ")
    df = DataFrame(
        vcat(
            [name => String[] for name in trial_category_names],
            [
                "True Positives" => Int[],
                "False Positives" => Int[],
                "True Negatives" => Int[],
                "False Negatives" => Int[],
                "Training Time" => Float64[]
            ]
        )
    )

    # Get directory of results
    sys_name = string(getname(dynamics))
    zip_dir = joinpath("results", experiment_name, sys_name)

    # For each trial/zip file in the directory
    for zip_name in readdir(zip_dir)
        # Skip any pre-existing summary files, which will be overwritten
        if zip_name == "summary.csv"
            continue
        end

        # Open the zip file
        zip_adr = joinpath(zip_dir, zip_name)
        _, ext = splitext(zip_adr)
        if ext != ".zip"
            @warn "Skipping $zip_name, as it is not a .zip file"
            continue
        end
        archive = ZipReader(read(zip_adr))

        # Initialize the row with the trial name
        trial_name, _ = splitext(zip_name)
        row = Dict(trial_category_names .=> split(trial_name, " - "))

        # Add the confusion matrix data to the row
        cm = DataFrame(CSV.File(zip_readentry(archive, "confusion_matrix.csv")))
        row = merge(row, Dict(first(eachcol(cm)) .=> last(eachcol(cm))))

        # Add the training time to the row
        tt = DataFrame(CSV.File(zip_readentry(archive, "timing_table.csv")))
        row = merge(row, Dict("Training Time" => tt[tt.Stage .== "Training", "Time [s]"][]))

        push!(df, row)
    end

    # Write the summary to a CSV file
    csv_adr = joinpath(zip_dir, "summary.csv")
    CSV.write(csv_adr, df)
    return nothing
end
