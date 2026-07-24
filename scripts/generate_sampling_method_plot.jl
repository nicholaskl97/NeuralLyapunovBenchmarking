#!/usr/bin/env julia
"""Generate a scatter plot from a sampling method summary CSV.

Reads a `summary.csv` file with `Training Time`, `Accuracy`, `System`, and
`Sampling Method` columns and writes `summary_plot.png` beside the CSV.
"""

using CSV
using DataFrames
using Plots
using Plots.PlotMeasures

const SAMPLING_METHOD_MARKERS = Dict(
    "GridTraining" => :utriangle,
    "QuasiRandomTraining" => :square,
    "StochasticTraining" => :circle,
)

"""Find a DataFrame column by name in a case-insensitive way."""
function find_col(df::DataFrame, colname)
    target = lowercase(string(colname))
    for name in names(df)
        if lowercase(string(name)) == target
            return name
        end
    end
    return nothing
end

"""Create and save the training-time vs accuracy scatter plot.

The plot uses separate colors for each `System` and separate marker shapes
for each sampling method. Two legend panels are rendered on the right.
"""
function save_accuracy_scatter(df::DataFrame, outpath::AbstractString)
    training_col = find_col(df, "Training Time")
    accuracy_col = find_col(df, :Accuracy)
    system_col = find_col(df, :System)
    method_col = find_col(df, "Sampling Method")

    if any(isnothing, (training_col, accuracy_col, system_col, method_col))
        @warn "Cannot create plot because required columns are missing"
        return false
    end

    systems = unique(df[!, system_col])
    colors = palette(:darktest, length(systems))
    system_color = Dict(systems[i] => colors[i] for i in eachindex(systems))

    point_colors = [system_color[s] for s in df[!, system_col]]
    point_markers = [get(SAMPLING_METHOD_MARKERS, m, :circle) for m in df[!, method_col]]

    main_plot = scatter(
        df[!, training_col],
        df[!, accuracy_col],
        color = point_colors,
        marker = point_markers,
        markersize = 12,
        xlabel = "Training Time",
        ylabel = "Accuracy",
        title = "Training Time vs Accuracy",
        legend = false,
        grid = false,
        guidefont = 24,
        tickfont = 20,
        titlefont = 28,
    )

    system_plot = plot(
        legendfont = 20,
        framestyle = :none,
        xaxis = false,
        yaxis = false,
    )
    for system in systems
        scatter!(system_plot, [NaN], [NaN], label = system, color = system_color[system], marker = :circle, markersize = 8)
    end
    plot!(system_plot, legend = :right, framestyle = :none, xaxis = false, yaxis = false)

    methods = unique(df[!, method_col])
    method_plot = plot(
        legendfont = 20,
        framestyle = :none,
        xaxis = false,
        yaxis = false,
    )
    for method in methods
        shape = get(SAMPLING_METHOD_MARKERS, method, :circle)
        scatter!(method_plot, [NaN], [NaN], label = method, color = :black, marker = shape, markersize = 12)
    end
    plot!(method_plot, legend = :right)

    right_panel = plot(
        system_plot,
        method_plot,
        layout = @layout([b; c]),
        margin = 6mm,
    )

    # Compose the final figure with the scatter on the left and legends on the right.
    full_plot = plot(
        main_plot,
        right_panel,
        layout = @layout([a{0.7w} b]),
        size = (1800, 1000),
        left_margin = 10mm,
        bottom_margin = 10mm,
        top_margin = 10mm,
        right_margin = 10mm,
    )

    savefig(full_plot, outpath)
    return true
end

function main(args)
    if length(args) < 1
        println("Usage: julia scripts/generate_sampling_method_plot.jl <summary-csv>")
        exit(1)
    end

    csv_path = args[1]
    if !isfile(csv_path)
        @warn "Summary CSV not found: $csv_path"
        exit(1)
    end

    df = CSV.read(csv_path, DataFrame)
    outpath = joinpath(dirname(csv_path), "summary_plot.png")
    if save_accuracy_scatter(df, outpath)
        @info "Wrote plot: $outpath"
    end
end

main(ARGS)
