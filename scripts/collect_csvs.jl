#!/usr/bin/env julia
"""Aggregate trial-level summary CSV files into experiment-level summary CSV files.

This script scans `<experiment>/results/<trial>/summary.csv` for the requested
experiment target and writes a merged `<experiment>/results/summary.csv`.
A `System` column is inserted from the trial name, and `Accuracy` is computed
from true/false positive/negative counts when available. It also aggregates
accuracy fit parameters and writes a Markdown report with the plots and tables.
"""

using CSV
using DataFrames

"""Convert a snake_case trial identifier to a display-friendly system name.

Special cases: `3d` becomes `3D` and `lqr` becomes `LQR`.
"""
function titlecase_system(name::AbstractString)
    parts = map(split(name, '_')) do p
        lp = lowercase(p)
        if lp == "3d"
            return "3D"
        elseif lp == "lqr"
            return "LQR"
        else
            return uppercasefirst(lp)
        end
    end
    return join(parts, ' ')
end

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

function markdown_cell(value)
    ismissing(value) && return ""
    return replace(string(value), "|" => "\\|", "\n" => " ", "\r" => " ")
end

function markdown_table(df)
    column_names = names(df)
    isempty(column_names) && return "_No data available._"

    lines = [
        "| " * join(markdown_cell.(column_names), " | ") * " |",
        "| " * join(fill("---", length(column_names)), " | ") * " |"
    ]
    for row in eachrow(df)
        push!(lines, "| " * join((markdown_cell(row[name]) for name in column_names), " | ") * " |")
    end
    return join(lines, "\n")
end

function markdown_plot_table(plot_rows)
    lines = ["|  |  |  |", "| --- | --- | --- |"]
    for row_start in 1:3:length(plot_rows)
        row_end = min(row_start + 2, length(plot_rows))
        cells = [
            "![Accuracy vs Training Time for $(plot_rows[index][1])]($(plot_rows[index][2]))<br>$(plot_rows[index][1])"
            for index in row_start:row_end
        ]
        append!(cells, fill("", 3 - length(cells)))
        push!(lines, "| " * join(cells, " | ") * " |")
    end
    return join(lines, "\n")
end

function summary_markdown(experiment, plot_rows, fit_parameters, summary)
    return join([
        "# $experiment Summary",
        "## Accuracy vs Training Time Plots",
        markdown_plot_table(plot_rows),
        "## Accuracy Fit Parameters",
            raw"$$\mathrm{Accuracy}(t) = A - C \cdot t^{-p}, \quad A, C, p \geq 0, \quad t = \mathrm{Training\ Time}.$$",
        markdown_table(fit_parameters),
        "## Trial Summary",
        markdown_table(summary)
    ], "\n\n")
end

function main(args)
    project_root = get(ENV, "PROJDIR", normpath(joinpath(@__DIR__, "..")))
    makefile_path = joinpath(project_root, "Makefile")

    if !isdir(project_root)
        @warn "project root not found: $project_root"
        exit(1)
    end

    # Parse Makefile to extract EXPERIMENTS, GROUPS, and TRIALS_* lists if available.
    EXPERIMENTS = String[]
    GROUPS = String[]
    TRIALS = Dict{String, Vector{String}}()
    if isfile(makefile_path)
        for line in readlines(makefile_path)
            if occursin("EXPERIMENTS", line) && occursin("=", line)
                m = match(r"^EXPERIMENTS\s*:?=\s*(.*)$", line)
                if m !== nothing
                    EXPERIMENTS = split(strip(m.captures[1]))
                end
            elseif occursin("GROUPS", line) && occursin("=", line)
                m = match(r"^GROUPS\s*:?=\s*(.*)$", line)
                if m !== nothing
                    GROUPS = split(strip(m.captures[1]))
                end
            else
                m = match(r"^TRIALS_([A-Za-z0-9_]+)\s*:?=\s*(.*)$", line)
                if m !== nothing
                    g = m.captures[1]
                    vals = split(strip(m.captures[2]))
                    TRIALS[g] = vals
                end
            end
        end
    end

    # Fallback if Makefile parsing failed or items missing; this keeps the script
    # usable even when the Makefile is not present or its variables change.
    if isempty(EXPERIMENTS)
        EXPERIMENTS = ["decrease_condition", "lyapunov-net_variants", "sampling_method"]
    end
    if isempty(GROUPS)
        GROUPS = ["undriven", "controlled", "neural_policy_search"]
    end

    # Determine requested target from args
    target = length(args) >= 1 ? args[1] : "all"

    if target == "" || target == "all"
        experiments = EXPERIMENTS
        trials_filter = Dict{String, Vector{String}}()
    else
        parts = split(target, '/')
        if length(parts) == 1
            # single experiment
            experiments = [parts[1]]
            trials_filter = Dict{String, Vector{String}}()
        elseif length(parts) == 2
            exp = parts[1]
            rest = parts[2]
            if rest in GROUPS
                # experiment/group -> include only trials from that group
                experiments = [exp]
                group_trials = get(TRIALS, rest, String[])
                trials_filter = Dict(exp => group_trials)
            else
                # experiment/trial -> specific trial target; nothing to do
                exit(0)
            end
        else
            @warn "Unsupported target format: $target"
            exit(1)
        end
    end

    for exp in experiments
        expdir = joinpath(project_root, exp, "results")
        if !isdir(expdir)
            @info "Skipping missing experiment results directory: $expdir"
            continue
        end
        dfs = DataFrame[]
        fit_dfs = DataFrame[]
        plot_rows = Tuple{String, String}[]

        # decide which trials to scan
        allowed_trials = haskey(trials_filter, exp) ? trials_filter[exp] : nothing

        for trial in filter(t -> isdir(joinpath(expdir, t)), readdir(expdir))
            if allowed_trials !== nothing && !(trial in allowed_trials)
                continue
            end
            trialdir = joinpath(expdir, trial)
            summary_path = joinpath(trialdir, "summary.csv")
            sysname = titlecase_system(trial)
            if isfile(summary_path)
                df = nothing
                try
                    df = CSV.read(summary_path, DataFrame)
                catch err
                    @warn "Failed to read $(summary_path): $err"
                    continue
                end
                if df === nothing
                    continue
                end
                insertcols!(df, 1, :System => fill(sysname, nrow(df)))
                push!(dfs, df)
            end

            fit_path = joinpath(trialdir, "accuracy_fit_parameters.csv")
            if isfile(fit_path)
                try
                    fit_df = CSV.read(fit_path, DataFrame)
                    insertcols!(fit_df, 1, :System => fill(sysname, nrow(fit_df)))
                    push!(fit_dfs, fit_df)
                catch err
                    @warn "Failed to read $(fit_path): $err"
                end
            end

            plot_path = joinpath(trialdir, "accuracy_vs_training_time.png")
            if isfile(plot_path)
                push!(plot_rows, (sysname, relpath(plot_path, expdir)))
            end
        end

        if !isempty(dfs)
            combined = vcat(dfs...; cols = :union)

            # Compute Accuracy when all required confusion-matrix columns exist.
            if all(
                name -> name in names(combined),
                ["True Positives", "True Negatives", "False Positives", "False Negatives"]
            )
                combined[!, :Accuracy] = (
                    combined[!, "True Positives"] .+ combined[!, "True Negatives"]
                ) ./ (
                    combined[!, "True Positives"] .+ combined[!, "True Negatives"] .+
                        combined[!, "False Positives"] .+ combined[!, "False Negatives"]
                )
            else
                @warn "Could not compute Accuracy because one or more required columns are missing for experiment: $exp"
                combined[!, :Accuracy] = missing
            end

            outpath = joinpath(expdir, "summary.csv")
            try
                CSV.write(outpath, combined)
                @info "Wrote summary: $outpath"
            catch err
                @warn "Failed to write $(outpath): $err"
            end

            fit_parameters = isempty(fit_dfs) ? DataFrame(System = String[]) :
                vcat(fit_dfs...; cols = :union)
            fit_outpath = joinpath(expdir, "fit_parameters.csv")
            try
                CSV.write(fit_outpath, fit_parameters)
                @info "Wrote fit parameters: $fit_outpath"
            catch err
                @warn "Failed to write $(fit_outpath): $err"
            end

            markdown_path = joinpath(expdir, "summary.md")
            try
                write(markdown_path, summary_markdown(exp, plot_rows, fit_parameters, combined))
                @info "Wrote Markdown summary: $markdown_path"
            catch err
                @warn "Failed to write $(markdown_path): $err"
            end
        else
            @info "No trial summary CSVs found for experiment: $exp"
        end
    end
    return nothing
end

main(ARGS)
