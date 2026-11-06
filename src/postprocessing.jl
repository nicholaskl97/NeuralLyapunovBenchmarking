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
    proj_dir = get(ENV, "PROJDIR", ".")
    dir = joinpath(proj_dir, experiment_name, "results", dynamics_name)
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

function fit_accuracy_curves(df, group_column)
    fit_parameters = DataFrame(
        group_column => String[],
        "Asymptotic Value" => Float64[],
        "Coefficient" => Float64[],
        "Power" => Float64[]
    )
    model(training_time, p) = p[1] .- p[2] .* training_time .^ (-p[3])

    for group_value in unique(df[!, group_column])
        group_df = df[df[!, group_column] .== group_value, :]
        training_times = Float64.(group_df[!, "Training Time"])
        accuracy = Float64.(group_df[!, "Accuracy"])
        valid = isfinite.(training_times) .& (training_times .> 0) .& isfinite.(accuracy)
        training_times = training_times[valid]
        accuracy = accuracy[valid]
        if length(training_times) < 3
            throw(ArgumentError("At least three valid observations are required to fit group '$group_value'."))
        end

        initial_power = 1.0
        initial_asymptote = max(maximum(accuracy), eps(Float64))
        initial_coefficient = max(initial_asymptote - minimum(accuracy), eps(Float64)) *
            minimum(training_times)^initial_power
        fit = curve_fit(
            model,
            training_times,
            accuracy,
            [initial_asymptote, initial_coefficient, initial_power];
            lower = zeros(3)
        )
        push!(fit_parameters, (
            group_value,
            fit.param[1],
            fit.param[2],
            fit.param[3]
        ))
    end

    return fit_parameters
end

function pretty_system_name(dynamics_name)
    words = map(split(string(dynamics_name), '_')) do word
        lowercase_word = lowercase(word)
        lowercase_word in ("3d", "lqr") ? uppercase(lowercase_word) : uppercasefirst(lowercase_word)
    end
    return join(words, ' ')
end

function save_accuracy_plot(df, plot_path, group_column, fit_parameters, dynamics_name)
    group_values = unique(df[!, group_column])
    colors = palette(:default, max(length(group_values), 1))
    accuracy_plt = plot(
        xscale = :log10,
        xlabel = "Training Time [s]",
        ylabel = "Accuracy",
        title = pretty_system_name(dynamics_name),
        legend = :best
    )

    for (group_index, group_value) in enumerate(group_values)
        group_df = df[df[!, group_column] .== group_value, :]
        training_times = group_df[!, "Training Time"]
        accuracy = group_df[!, "Accuracy"]
        scatter!(
            accuracy_plt,
            training_times,
            accuracy;
            label = string(group_value),
            color = colors[group_index]
        )

        fit_row = fit_parameters[fit_parameters[!, group_column] .== group_value, :]
        parameters = fit_row[1, :]
        curve_times = exp.(range(
            log(minimum(training_times)),
            log(maximum(training_times));
            length = 200
        ))
        curve_accuracy = parameters["Asymptotic Value"] .-
            parameters["Coefficient"] .* curve_times .^ (-parameters["Power"])
        plot!(
            accuracy_plt,
            curve_times,
            curve_accuracy;
            label = false,
            color = colors[group_index]
        )
    end

    savefig(accuracy_plt, plot_path)
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
                "Accuracy" => Float64[],
                "Training Time" => Float64[]
            ]
        )
    )

    # Get directory of results
    sys_name = string(getname(dynamics))
    proj_dir = get(ENV, "PROJDIR", ".")
    zip_dir = joinpath(proj_dir, experiment_name, "results", sys_name)

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
        row["Accuracy"] = (row["True Positives"] + row["True Negatives"]) /
            (row["True Positives"] + row["True Negatives"] +
             row["False Positives"] + row["False Negatives"])

        # Add the training time to the row
        tt = DataFrame(CSV.File(zip_readentry(archive, "timing_table.csv")))
        row = merge(row, Dict("Training Time" => tt[tt.Stage .== "Training", "Time [s]"][]))

        push!(df, row)
    end

    # Write the summary to a CSV file
    CSV.write(joinpath(zip_dir, "summary.csv"), df)

    # Write the accuracy fit parameters and plot to files
    group_column = first(trial_category_names)
    fit_parameters = fit_accuracy_curves(df, group_column)
    CSV.write(joinpath(zip_dir, "accuracy_fit_parameters.csv"), fit_parameters)
    save_accuracy_plot(
        df,
        joinpath(zip_dir, "accuracy_vs_training_time.png"),
        group_column,
        fit_parameters,
        sys_name
    )

    # Write versioninfo, Status, and Manifest
    reproducibility_adr = joinpath(zip_dir, "reproducibility.txt")
    open(reproducibility_adr, "w") do f
        versioninfo(f)
        println(f, "\n")
        status(; io = f)
        println(f, "\n")
        status(; mode = PKGMODE_MANIFEST, io = f)
    end

    return nothing
end
