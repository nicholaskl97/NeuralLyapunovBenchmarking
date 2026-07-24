#!/usr/bin/env julia
"""Aggregate trial-level summary CSV files into experiment-level summary CSV files.

This script scans `results/<experiment>/<trial>/summary.csv` for the requested
experiment target and writes a merged `results/<experiment>/summary.csv`.
A `System` column is inserted from the trial name, and `Accuracy` is computed
from true/false positive/negative counts when available.
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

function main(args)
    project_root = get(ENV, "PROJDIR", normpath(joinpath(@__DIR__, "..")))
    results_dir = joinpath(project_root, "results")
    makefile_path = joinpath(project_root, "Makefile")

    if !isdir(results_dir)
        @warn "results directory not found: $results_dir"
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
        expdir = joinpath(results_dir, exp)
        if !isdir(expdir)
            @info "Skipping missing experiment results directory: $expdir"
            continue
        end
        dfs = DataFrame[]

        # decide which trials to scan
        allowed_trials = haskey(trials_filter, exp) ? trials_filter[exp] : nothing

        for trial in filter(t -> isdir(joinpath(expdir, t)), readdir(expdir))
            if allowed_trials !== nothing && !(trial in allowed_trials)
                continue
            end
            trialdir = joinpath(expdir, trial)
            summary_path = joinpath(trialdir, "summary.csv")
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
                sysname = titlecase_system(trial)
                insertcols!(df, 1, :System => fill(sysname, nrow(df)))
                push!(dfs, df)
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
        else
            @info "No trial summary CSVs found for experiment: $exp"
        end
    end
    return nothing
end

main(ARGS)
