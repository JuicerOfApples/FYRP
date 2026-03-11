using Printf

function get_system_params()::Dict{String,Any}
    return Dict{String,Any}(
        "J" => 1.0,
        "Δ" => 1.0,
        "μ" => 1.0,
        "NUM_SWEEPS" => 10,
        "MAX_BOND_DIM" => 1000,
        "ACC" => 1e-10
    )
end

"""Get a standardised filename for a set of system parameters."""
function extract_filename_from_system_params(params::Dict{String,Any})::String
    string = ""
    for (key, value) in params
        if !isempty(string)
            string *= "_"
        end
        # format floats in scientific notation with 2 decimal places
        formatted_value = value isa AbstractFloat ? @sprintf("%.2e", value) : value
        string *= "$(key)=$(formatted_value)"
    end
    return string
end



using Pkg;
Pkg.add(["HDF5", "ITensors"]);
using HDF5, ITensors
using Dates, Logging

function _read_params_from_group(g)::Dict{String,Any}
    d = Dict{String,Any}()
    for key in keys(g)
        try
            d[string(key)] = read(g, key)
        catch
            d[string(key)] = string(g[key])
        end
    end
    return d
end

function _write_params_to_group(g, params::Dict{String,Any})
    for (k, v) in params
        try
            g[k] = v
        catch e
            @warn "Could not write params[$k] as HDF5; saving as string" exception = (e, catch_backtrace())
            g[k] = string(v)
        end
    end
end

"""
Save an MPS with its system parameters and its run parameters.

Storage Layout (fresh format)
- /system_params                               (group)     : holds system-wide parameters
- /runs/<run_id>/params                        (group)     : canonical run-specific parameters
- /runs/<run_id>/instances/<instance_id>/psi   (dataset)   : the MPS for this instance of the run
- /runs/<run_id>/instances/<instance_id>/timestamp (dataset) : string timestamp for that instance

Behavior
- If a run group exists with the exact same `run_params`, a new instance is appended under that group.
- Otherwise a new run group is created (numeric next id or provided `run_id`).
"""
function save_mps_with_params(
    filepath::String,
    ψ::MPS,
    system_params::Dict{String,Any},
    run_params::Dict{String,Any};
    run_id::Union{String,Nothing}=nothing,
    param_safety::Bool=true
)
    mode = isfile(filepath) ? "r+" : "w"
    h5open(filepath, mode) do file
        # system params: create only if not present
        if !haskey(file, "system_params")
            system_group = create_group(file, "system_params")
            for (k, v) in system_params
                try
                    system_group[k] = v
                catch e
                    @warn "Could not write system_params[$k] as HDF5; saving as string" exception = (e, catch_backtrace())
                    system_group[k] = string(v)
                end
            end
        else
            # do not overwrite existing system params; optionally error if new keys are present
            existing = file["system_params"]
            for (k, v) in system_params
                if !haskey(existing, k)
                    @warn "system_params in file is missing key $k. This function will not overwrite."
                    if param_safety
                        error("param_safety is enabled; aborting because file system_params is missing key $k.")
                    else
                        @warn "Proceeding with write; system_params in file may be inconsistent."
                    end
                end
            end
        end

        # prepare runs group
        runs_group = haskey(file, "runs") ? file["runs"] : create_group(file, "runs")

        # Attempt to find existing run group with identical run_params
        function _read_params_from_group(g)::Dict{String,Any}
            d = Dict{String,Any}()
            for key in keys(g)
                try
                    d[string(key)] = read(g, key)
                catch
                    d[string(key)] = string(g[key])
                end
            end
            return d
        end

        matching_run_name = nothing
        for existing_name in keys(runs_group)
            existing_rg = runs_group[string(existing_name)]
            if haskey(existing_rg, "params")
                existing_params = _read_params_from_group(existing_rg["params"])
                if existing_params == run_params
                    matching_run_name = string(existing_name)
                    break
                end
            end
        end

        if matching_run_name !== nothing
            # Append instance to existing run group
            run_group = runs_group[matching_run_name]
            # ensure instances group exists
            instances_group = haskey(run_group, "instances") ? run_group["instances"] : create_group(run_group, "instances")

            existing_inst_names = collect(keys(instances_group))
            nums = Int[]
            for nm in existing_inst_names
                n = tryparse(Int, string(nm))
                if n !== nothing
                    push!(nums, n)
                end
            end
            next_num = isempty(nums) ? 1 : maximum(nums) + 1
            inst_name = string(next_num)
            inst_group = create_group(instances_group, inst_name)

            try
                write(inst_group, "psi", ψ)
            catch e
                @error "Failed to write MPS instance to HDF5" exception = (e, catch_backtrace())
                rethrow(e)
            end

            try
                inst_group["timestamp"] = string(Dates.now())
            catch
                # ignore timestamp failures
            end

            @info "Appended instance '$inst_name' to existing run '$matching_run_name' in $filepath."
            return matching_run_name
        else
            # Create a new run group
            if run_id === nothing
                existing_names = collect(keys(runs_group))
                nums = Int[]
                for nm in existing_names
                    n = tryparse(Int, string(nm))
                    if n !== nothing
                        push!(nums, n)
                    end
                end
                next_num = isempty(nums) ? 1 : maximum(nums) + 1
                run_id = string(next_num)
            else
                run_id = string(run_id)
                if haskey(runs_group, run_id)
                    base = run_id
                    counter = 1
                    while haskey(runs_group, "$(base)_$(counter)")
                        counter += 1
                    end
                    run_id = "$(base)_$(counter)"
                end
            end

            run_group = create_group(runs_group, run_id)
            # write params
            params_group = create_group(run_group, "params")
            _write_params_to_group(params_group, run_params)

            # create instances/1 and write psi
            instances_group = create_group(run_group, "instances")
            inst_group = create_group(instances_group, "1")
            try
                write(inst_group, "psi", ψ)
            catch e
                @error "Failed to write MPS to HDF5" exception = (e, catch_backtrace())
                rethrow(e)
            end
            try
                inst_group["timestamp"] = string(Dates.now())
            catch
                # ignore
            end

            @info "Created new run '$run_id' with first instance in $filepath."
            return run_id
        end
    end
end


"""
Load an MPS together with system and run params.

Assumes fresh format (params + instances). If `run_id` is omitted the most recently created
run is selected (largest numeric name if numeric run ids present, otherwise lexicographic last).

Returns (psi::MPS, system_params::Dict{String,Any}, run_params::Dict{String,Any})
"""
function load_mps_with_params(
    filepath::String;
    run_id::Union{String,Nothing}=nothing
)::Tuple{MPS,Dict{String,Any},Dict{String,Any}}
    system_params = Dict{String,Any}()
    run_params = Dict{String,Any}()

    result = h5open(filepath, "r") do file
        if haskey(file, "system_params")
            system_group = file["system_params"]
            for key in keys(system_group)
                try
                    system_params[string(key)] = read(system_group, key)
                catch
                    system_params[string(key)] = string(system_group[key])
                end
            end
        else
            @warn "No `system_params` group found in $filepath."
        end

        if !haskey(file, "runs")
            error("No 'runs' group found in $filepath")
        end

        runs_group = file["runs"]
        run_names = collect(keys(runs_group))
        if isempty(run_names)
            error("No runs found in 'runs' group of $filepath")
        end

        # choose run_id if not given
        if run_id === nothing
            nums_map = Dict{Int,String}()
            for nm in run_names
                n = tryparse(Int, string(nm))
                if n !== nothing
                    nums_map[n] = string(nm)
                end
            end
            run_id = if !isempty(nums_map)
                nums_map[maximum(keys(nums_map))]
            else
                sort(string.(run_names))[end]
            end
        else
            if !in(string(run_id), run_names)
                error("Requested run_id `$(run_id)` not found in file. Available runs: $(run_names).")
            end
            run_id = string(run_id)
        end

        run_group = runs_group[run_id]

        # run params must exist in fresh layout
        if !haskey(run_group, "params")
            error("Run '$run_id' missing 'params' group (expect fresh layout).")
        end
        run_params = _read_params_from_group(run_group["params"])

        # instances must exist in fresh layout
        if !haskey(run_group, "instances")
            error("Run '$run_id' missing 'instances' group (expect fresh layout).")
        end

        instances_group = run_group["instances"]
        inst_names = collect(keys(instances_group))
        if isempty(inst_names)
            error("Run '$run_id' has an empty 'instances' group.")
        end

        # choose most recent instance: prefer numeric instance ids
        nums = Int[]
        for nm in inst_names
            n = tryparse(Int, string(nm))
            if n !== nothing
                push!(nums, n)
            end
        end
        chosen_inst = if !isempty(nums)
            string(maximum(nums))
        else
            sort(string.(inst_names))[end]
        end

        inst_group = instances_group[chosen_inst]

        psi = try
            read(inst_group, "psi", MPS)
        catch e
            @warn "Direct read into MPS failed; attempting raw read for diagnostics."
            raw = try
                read(inst_group, "psi")
            catch inner
                @error "Raw read also failed" exception = (inner, catch_backtrace())
                rethrow(inner)
            end
            @info "Raw read type: $(typeof(raw))"
            if isa(raw, Dict)
                @info "Raw read keys: $(collect(keys(raw)))"
            end
            rethrow(e)
        end

        (psi, system_params, run_params)
    end

    return result
end


"""
Return the system parameters in a specific file.
"""
function get_system_params_of_file(filepath::String)::Union{Dict{String,Any},Nothing}
    system_params = Dict{String,Any}()

    h5open(filepath, "r") do file
        if haskey(file, "system_params")
            system_group = file["system_params"]
            for key in keys(system_group)
                try
                    system_params[string(key)] = read(system_group, key)
                catch
                    system_params[string(key)] = string(system_group[key])
                end
            end
        else
            @warn "No `system_params` group found in $filepath."
            return nothing
        end
    end

    return system_params
end


"""
Load all the run data from a file (fresh format).

Returns a tuple: (system_params::Dict{String,Any}, runs::Vector)
Each element of `runs` is a named tuple:
  (run_id::String, instance_id::String, psi::MPS, params::Dict{String,Any}, timestamp::Union{String,Nothing})
"""
function load_all_mps_from_file(filepath::String)
    function _read_params_from_group(g)::Dict{String,Any}
        d = Dict{String,Any}()
        for key in keys(g)
            try
                d[string(key)] = read(g, key)
            catch
                d[string(key)] = string(g[key])
            end
        end
        return d
    end

    system_params = get_system_params_of_file(filepath)
    if system_params === nothing
        system_params = Dict{String,Any}()
    end

    runs = Vector{NamedTuple{(:run_id, :instance_id, :psi, :params, :timestamp),Tuple{String,String,MPS,Dict{String,Any},Union{String,Nothing}}}}()

    h5open(filepath, "r") do file
        if !haskey(file, "runs")
            @warn "No 'runs' group found in $filepath"
            return (system_params, runs)
        end

        runs_group = file["runs"]
        run_names = collect(keys(runs_group))
        if isempty(run_names)
            @warn "No runs found in 'runs' group of $filepath"
            return (system_params, runs)
        end

        # order runs: prefer numeric ids ascending else lexicographic
        nums_map = Dict{Int,String}()
        for nm in run_names
            n = tryparse(Int, string(nm))
            if n !== nothing
                nums_map[n] = string(nm)
            end
        end
        ordered_run_names = if !isempty(nums_map)
            [nums_map[k] for k in sort(collect(keys(nums_map)))]
        else
            sort(string.(run_names))
        end

        for rn in ordered_run_names
            run_group = runs_group[rn]
            if !haskey(run_group, "params") || !haskey(run_group, "instances")
                @warn "Skipping run $rn because it does not conform to fresh layout (missing params or instances)."
                continue
            end
            run_params = _read_params_from_group(run_group["params"])

            instances_group = run_group["instances"]
            inst_names = collect(keys(instances_group))
            if isempty(inst_names)
                continue
            end

            # order instances: numeric ascending else lexicographic
            nums = Int[]
            for nm in inst_names
                n = tryparse(Int, string(nm))
                if n !== nothing
                    push!(nums, n)
                end
            end
            ordered_inst_names = if !isempty(nums)
                [string(i) for i in sort(nums)]
            else
                sort(string.(inst_names))
            end

            for inst in ordered_inst_names
                inst_group = instances_group[inst]
                psi = try
                    read(inst_group, "psi", MPS)
                catch e
                    @warn "Direct read into MPS failed for run $rn instance $inst; attempting raw read for diagnostics."
                    raw = try
                        read(inst_group, "psi")
                    catch inner
                        @error "Raw read also failed for run $rn instance $inst" exception = (inner, catch_backtrace())
                        rethrow(inner)
                    end
                    @info "Raw read type: $(typeof(raw))"
                    if isa(raw, Dict)
                        @info "Raw read keys: $(collect(keys(raw)))"
                    end
                    rethrow(e)
                end
                timestamp = haskey(inst_group, "timestamp") ? string(inst_group["timestamp"]) : nothing
                push!(runs, (run_id=string(rn), instance_id=string(inst), psi=psi, params=run_params, timestamp=timestamp))
            end
        end
    end

    return (system_params, runs)
end

function _params_match(existing::Dict{String,Any}, query::Dict{String,Any})::Bool
    # Match if for every (k,v) in query, existing[k] equals v (direct equality
    # or equality of string representations). This handles values saved as strings.
    for (k, vq) in query
        ks = string(k)
        if !haskey(existing, ks)
            return false
        end
        ev = existing[ks]
        if ev == vq
            continue
        end
        if string(ev) == string(vq)
            continue
        end
        return false
    end
    return true
end

"""
Find runs in `filepath` whose run parameters match `query_params`.

Arguments
- `filepath::String`: path to the HDF5 file.
- `query_params::Dict{String,Any}`: only runs that contain matching key/value pairs for all entries in this dict are returned.
  (Matching is tolerant to values that were stored as strings, i.e. `string(existing) == string(query)` will be considered equal.)
Keyword arguments
- `load_psi::Bool=false` : if `true`, the function will read and return the `psi` for the selected instance(s) (this can be slow).
- `instance_selection::Union{String,Symbol,Nothing}=:latest` :
    - `:latest` (default) -> choose the most recent instance of matching runs (prefers numeric instance ids)
    - `:all` -> return all instances for matching runs
    - a `String` or `Int` -> specific instance id to load/check

Returns
A `Vector` of named tuples. Each element contains:
  - `run_id::String`
  - `instance_id::String`
  - `psi::Union{MPS,Nothing}`  (present when `load_psi=true`, otherwise `nothing`)
  - `params::Dict{String,Any}` (the run params read from file)
  - `timestamp::Union{String,Nothing}`

Notes
- This function only reads run `params` groups for matching and will avoid loading MPS data unless asked (useful for large files).
"""
function find_runs_by_params(
    filepath::String,
    query_params::Dict{String, Any};
    load_psi::Bool=false,
    instance_selection::Union{String,Symbol,Nothing}=:latest
)
    matches = Vector{NamedTuple{(:run_id,:instance_id,:psi,:params,:timestamp),Tuple{String,String,Union{MPS,Nothing},Dict{String,Any},Union{String,Nothing}}}}()

    if !isfile(filepath)
        @warn "File does not exist: $filepath"
        return Vector{NamedTuple}()
    end

    h5open(filepath, "r") do file
        if !haskey(file, "runs")
            @info "No 'runs' group in $filepath"
            return matches
        end
        runs_group = file["runs"]
        for rn in keys(runs_group)
            run_group = runs_group[string(rn)]
            if !haskey(run_group, "params") || !haskey(run_group, "instances")
                # skip non-fresh-layout run groups
                continue
            end
            # read run params
            existing_params = _read_params_from_group(run_group["params"])
            if !_params_match(existing_params, query_params)
                continue
            end

            # We have a matching run. Decide instances to return
            instances_group = run_group["instances"]
            inst_names = collect(keys(instances_group))
            if isempty(inst_names)
                continue
            end

            ordered_inst_names = begin
                # prefer numeric ordering if possible
                nums = Int[]
                for nm in inst_names
                    n = tryparse(Int, string(nm))
                    if n !== nothing
                        push!(nums, n)
                    end
                end
                if !isempty(nums)
                    [string(i) for i in sort(nums)]
                else
                    sort(string.(inst_names))
                end
            end

            selected_instances = String[]
            if instance_selection === :all
                append!(selected_instances, ordered_inst_names)
            elseif instance_selection === :latest
                push!(selected_instances, ordered_inst_names[end])
            else
                # user provided specific instance id (string or integer)
                sel = string(instance_selection)
                if sel in ordered_inst_names
                    push!(selected_instances, sel)
                else
                    @warn "Requested instance '$sel' not found in run $(rn); available instances: $(ordered_inst_names). Skipping."
                    continue
                end
            end

            for inst in selected_instances
                inst_group = instances_group[inst]
                timestamp = haskey(inst_group, "timestamp") ? string(inst_group["timestamp"]) : nothing
                psi_val = nothing
                if load_psi
                    psi_val = try
                        read(inst_group, "psi", MPS)
                    catch e
                        @warn "Direct read into MPS failed for run $(rn) instance $(inst); attempting raw read for diagnostics."
                        raw = try
                            read(inst_group, "psi")
                        catch inner
                            @error "Raw read also failed for run $(rn) instance $(inst)" exception = (inner, catch_backtrace())
                            rethrow(inner)
                        end
                        @info "Raw read type: $(typeof(raw))"
                        if isa(raw, Dict)
                            @info "Raw read keys: $(collect(keys(raw)))"
                        end
                        rethrow(e)
                    end
                end
                # named tuple, containing run_id, instance_id, psi (if loaded), params and timestamp
                push!(matches, (run_id=string(rn), instance_id=string(inst), psi=psi_val, params=existing_params, timestamp=timestamp))
            end
        end
    end

    return matches
end



function get_squared_sorted_schmidt_spectrum_from_mps(psi::MPS)
    N = length(psi)

    # MIDDLE bond
    middle_bond = N ÷ 2

    orthogonalize!(psi, middle_bond)
    link_idx = linkind(psi, middle_bond)
    _, S, _ = svd(psi[middle_bond], (siteind(psi, middle_bond), link_idx))

    schmidt_dim = dim(S, 1)

    schmidt_vals = []
    for n in 1:schmidt_dim
        push!(schmidt_vals, S[n, n]) # steal diagonals
    end

    # sort desc and square
    return sort(schmidt_vals .^ 2, rev=true)
end



using Kneedle
using SavitzkyGolay

function find_deviation_idx(schmidt_spectrum::Vector{Float64}; window_size=11, poly_order=2)::Int64

    log_spec = log10.(max.(schmidt_spectrum, 1e-18))
    sg = savitzky_golay(log_spec, window_size, poly_order) # smooth to get better results
    smoothed_log_y = sg.y
    x = 1:length(smoothed_log_y)
    res = kneedle(x, smoothed_log_y)

    deviation_index = knees(res)
    return deviation_index[1]
end



