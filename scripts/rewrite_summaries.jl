#!/usr/bin/env julia

using NeuralLyapunovBenchmarking: write_summary

const TRIAL_CATEGORIES = Dict(
    "decrease_condition" => "Decrease Condition - Rectifier",
    "lyapunov-net_variants" => "Architecture - N - Seed",
    "sampling_method" => "Sampling Method"
)

function make_definitions(makefile_path)
    experiments = String[]
    groups = String[]
    trials = Dict{String, Vector{String}}()

    for line in readlines(makefile_path)
        if (m = match(r"^EXPERIMENTS\s*:?=\s*(.*)$", line)) !== nothing
            experiments = split(strip(m.captures[1]))
        elseif (m = match(r"^GROUPS\s*:?=\s*(.*)$", line)) !== nothing
            groups = split(strip(m.captures[1]))
        elseif (m = match(r"^TRIALS_([A-Za-z0-9_]+)\s*:?=\s*(.*)$", line)) !== nothing
            trials[m.captures[1]] = split(strip(m.captures[2]))
        end
    end

    isempty(experiments) && error("No EXPERIMENTS definition found in $makefile_path")
    isempty(groups) && error("No GROUPS definition found in $makefile_path")
    return experiments, groups, trials
end

function systems_for_experiment(project_root, experiment, groups, trials)
    systems = String[]
    for group in groups
        for system in get(trials, group, String[])
            script_path = joinpath(project_root, experiment, "scripts", "$system.jl")
            if isfile(script_path) && !(system in systems)
                push!(systems, system)
            end
        end
    end
    return systems
end

function requested_targets(target, project_root, experiments, groups, trials)
    if target == "all"
        return [
            (experiment, system)
            for experiment in experiments
            for system in systems_for_experiment(project_root, experiment, groups, trials)
        ]
    end

    parts = split(target, '/')
    if length(parts) == 1
        experiment = only(parts)
        if !(experiment in experiments)
            throw(ArgumentError("Unknown experiment '$experiment'."))
        end
        return [
            (experiment, system)
            for system in systems_for_experiment(project_root, experiment, groups, trials)
        ]
    elseif length(parts) == 2
        experiment, selection = parts
        if !(experiment in experiments)
            throw(ArgumentError("Unknown experiment '$experiment'."))
        end

        available_systems = systems_for_experiment(project_root, experiment, groups, trials)
        if selection in groups
            return [
                (experiment, system)
                for system in get(trials, selection, String[])
                if system in available_systems
            ]
        elseif any(selection in get(trials, group, String[]) for group in groups)
            if !(selection in available_systems)
                throw(ArgumentError("System '$selection' is not available for '$experiment'."))
            end
            return [(experiment, selection)]
        else
            throw(ArgumentError("Unknown trial group or system '$selection'."))
        end
    end

    throw(ArgumentError("Unsupported target '$target'. Use all, experiment, or experiment/group_or_system."))
end

function main(args)
    if length(args) > 1
        throw(ArgumentError("Expected at most one target: all, experiment, or experiment/group_or_system."))
    end

    project_root = get(ENV, "PROJDIR", normpath(joinpath(@__DIR__, "..")))
    makefile_path = joinpath(project_root, "Makefile")
    if !isfile(makefile_path)
        error("Makefile not found: $makefile_path")
    end

    experiments, groups, trials = make_definitions(makefile_path)
    target = isempty(args) ? "all" : args[1]
    selected_targets = requested_targets(target, project_root, experiments, groups, trials)

    for (experiment, system) in selected_targets
        category = get(TRIAL_CATEGORIES, experiment, nothing)
        if category === nothing
            error("No summary category configured for experiment '$experiment'.")
        end
        @info "Rewriting summary" experiment system
        write_summary(system, experiment, category)
    end

    isempty(selected_targets) && @info "No matching experiment systems found for target '$target'."
    return nothing
end

main(ARGS)
