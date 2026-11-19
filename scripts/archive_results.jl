#!/usr/bin/env julia

using Dates
using ZipArchives: ZipWriter, zip_mkdir, zip_newfile

const PROJECT_ROOT = get(ENV, "PROJDIR", normpath(joinpath(@__DIR__, "..")))

function experiment_paths(name::AbstractString; project_root::AbstractString = PROJECT_ROOT)
    if isempty(name) || name in (".", "..") || occursin('/', name) || occursin('\\', name)
        throw(ArgumentError("experiment must be a directory name directly under $project_root"))
    end

    experiment_dir = joinpath(project_root, name)
    if !isdir(experiment_dir)
        throw(ArgumentError("experiment directory not found: $experiment_dir"))
    end
    if islink(experiment_dir)
        throw(ArgumentError("experiment directory must not be a symbolic link"))
    end

    results_dir = joinpath(experiment_dir, "results")
    if !isdir(results_dir)
        throw(ArgumentError("results directory not found: $results_dir"))
    end
    if islink(results_dir)
        throw(ArgumentError("results directory must not be a symbolic link"))
    end

    return experiment_dir, results_dir
end

function clear_results_contents!(directory::AbstractString)
    for path in readdir(directory; join = true)
        if isdir(path) && !islink(path)
            clear_results_contents!(path)
        else
            rm(path; force = true, recursive = true)
        end
    end
    return nothing
end

function archive_results(experiment_name::AbstractString; project_root::AbstractString = PROJECT_ROOT)
    experiment_dir, results_dir = experiment_paths(experiment_name; project_root)
    archives_dir = joinpath(experiment_dir, "archives")
    if islink(archives_dir)
        throw(ArgumentError("archives directory must not be a symbolic link"))
    end
    mkpath(archives_dir)

    timestamp = Dates.format(now(), dateformat"yyyy-mm-dd_HH-MM")
    archive_path = joinpath(archives_dir, "$timestamp.zip")
    if ispath(archive_path)
        throw(ArgumentError("archive already exists: $archive_path"))
    end

    temporary_archive = tempname(archives_dir)
    try
        ZipWriter(temporary_archive) do archive
            for (root, directories, files) in walkdir(results_dir; follow_symlinks = false)
                for entry in vcat(directories, files)
                    islink(joinpath(root, entry)) &&
                        throw(ArgumentError("cannot archive symbolic link: $(joinpath(root, entry))"))
                end

                if isempty(directories) && isempty(files) && root != results_dir
                    relative_dir = replace(relpath(root, results_dir), '\\' => '/') * "/"
                    zip_mkdir(archive, relative_dir)
                end

                for filename in files
                    source_path = joinpath(root, filename)
                    relative_path = replace(relpath(source_path, results_dir), '\\' => '/')
                    zip_newfile(archive, relative_path; compress = true)
                    open(source_path, "r") do source
                        while !eof(source)
                            write(archive, read(source, 1024 * 1024))
                        end
                    end
                end
            end
        end

        mv(temporary_archive, archive_path)
    finally
        isfile(temporary_archive) && rm(temporary_archive; force = true)
    end

    clear_results_contents!(results_dir)
    @info "Archived results to $archive_path and cleared $results_dir"
    return archive_path
end

function main(args)
    if length(args) != 1
        println("Usage: julia --project=. scripts/archive_results.jl <experiment-directory>")
        exit(1)
    end
    archive_results(args[1])
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
