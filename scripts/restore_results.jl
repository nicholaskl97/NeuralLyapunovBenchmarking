#!/usr/bin/env julia

include(joinpath(@__DIR__, "archive_results.jl"))

using ZipArchives: ZipReader, zip_isdir, zip_name, zip_nentries, zip_openentry,
    zip_test, zip_uncompressed_size

function resolve_archive(experiment_dir::AbstractString, archive_date::AbstractString)
    archives_dir = joinpath(experiment_dir, "archives")
    isdir(archives_dir) || throw(ArgumentError("archives directory not found: $archives_dir"))
    islink(archives_dir) && throw(ArgumentError("archives directory must not be a symbolic link"))

    if occursin(r"^\d{4}-\d{2}-\d{2}$", archive_date)
        Date(archive_date, dateformat"yyyy-mm-dd")
        matches = filter(readdir(archives_dir)) do filename
            occursin(Regex("^" * archive_date * "(?:_\\d{2}-\\d{2})?\\.zip\$"), filename)
        end
        if isempty(matches)
            throw(ArgumentError("no archive found for date $archive_date in $archives_dir"))
        elseif length(matches) > 1
            throw(ArgumentError(
                "multiple archives found for $archive_date; specify a date and time: $(join(sort(matches), ", "))"
            ))
        end
        return joinpath(archives_dir, only(matches))
    elseif occursin(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$", archive_date)
        DateTime(archive_date, dateformat"yyyy-mm-dd_HH-MM")
        archive_path = joinpath(archives_dir, "$archive_date.zip")
        if !isfile(archive_path)
            throw(ArgumentError("archive not found: $archive_path"))
        end
        return archive_path
    else
        throw(ArgumentError("date must be YYYY-MM-DD or YYYY-MM-DD_HH-mm"))
    end
end

function safe_archive_parts(name::AbstractString, is_directory::Bool)
    if isempty(name)
        throw(ArgumentError("archive contains an empty entry name"))
    end
    if startswith(name, '/') || occursin('\\', name)
        throw(ArgumentError("archive contains an unsafe entry path: $name"))
    end

    parts = split(name, '/'; keepempty = true)
    if is_directory
        if !endswith(name, '/')
            throw(ArgumentError("invalid directory entry in archive: $name"))
        end
        pop!(parts)
    elseif endswith(name, '/')
        throw(ArgumentError("invalid file entry in archive: $name"))
    end

    if isempty(parts) || any(part -> part in ("", ".", ".."), parts)
        throw(ArgumentError("archive contains an unsafe entry path: $name"))
    end
    if occursin(r"^[A-Za-z]:", first(parts))
        throw(ArgumentError("archive contains an unsafe entry path: $name"))
    end

    return parts
end

function extract_archive!(archive_path::AbstractString, staging_dir::AbstractString)
    archive = ZipReader(read(archive_path))
    zip_test(archive)

    entries = [
        (zip_name(archive, index), zip_isdir(archive, index), index)
        for index in 1:zip_nentries(archive)
    ]
    for (name, is_directory, index) in entries
        parts = safe_archive_parts(name, is_directory)
        destination = joinpath(staging_dir, parts...)

        if is_directory
            if zip_uncompressed_size(archive, index) != 0
                throw(ArgumentError("directory entry contains data: $name"))
            end
            mkpath(destination)
        else
            mkpath(dirname(destination))
            zip_openentry(archive, index) do source
                open(destination, "w") do output
                    while !eof(source)
                        write(output, read(source, 1024 * 1024))
                    end
                end
            end
        end
    end
    return nothing
end

function results_are_empty(directory::AbstractString)
    for path in readdir(directory; join = true)
        if isdir(path) && !islink(path)
            if !results_are_empty(path)
                return false
            end
        else
            return false
        end
    end
    return true
end

function choose_restore_action()
    println("The results directory is not empty. Choose an action:")
    println("  [o] Overwrite without archiving the current results")
    println("  [a] Archive the current results, then overwrite")
    println("  [q] Exit without changing the results")
    while true
        print("Choice [o/a/q]: ")
        flush(stdout)
        choice = lowercase(strip(readline(stdin)))
        if choice in ("o", "a", "q")
            return choice
        end
        println("Please enter o, a, or q.")
    end
end

function replace_results!(results_dir::AbstractString, staging_dir::AbstractString)
    backup_dir = tempname(dirname(results_dir))
    mv(results_dir, backup_dir)
    try
        mv(staging_dir, results_dir)
    catch
        mv(backup_dir, results_dir)
        rethrow()
    end
    rm(backup_dir; recursive = true)
    return nothing
end

function restore_results(
    experiment_name::AbstractString,
    archive_date::AbstractString;
    project_root::AbstractString = PROJECT_ROOT,
)
    experiment_dir, results_dir = experiment_paths(experiment_name; project_root)
    archive_path = resolve_archive(experiment_dir, archive_date)
    choice = results_are_empty(results_dir) ? "o" : choose_restore_action()
    if choice == "q"
        return nothing
    end

    mktempdir(dirname(results_dir)) do temporary_parent
        staging_dir = joinpath(temporary_parent, "restored-results")
        mkdir(staging_dir)
        extract_archive!(archive_path, staging_dir)

        if choice == "a"
            archive_results(experiment_name; project_root)
        end
        replace_results!(results_dir, staging_dir)
    end

    @info "Restored results from $archive_path to $results_dir"
    return nothing
end

function main(args)
    if length(args) != 2
        println("Usage: julia --project=. scripts/restore_results.jl <experiment-directory> <YYYY-MM-DD[_HH-mm]>")
        exit(1)
    end
    restore_results(args[1], args[2])
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
