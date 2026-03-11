# ==============================================================================
# Helper Functions Library - MPS DMRG and HDF5 I/O
# ==============================================================================
# This module provides functions for working with Matrix Product States (MPS)
# and saving/loading them with system parameters to HDF5 files.
#
# Functions exported as a library (call from other files):
#   - generate_fully_connected_wam: Create random weighted adjacency matrix
#   - create_xxz_hamiltonian_mpo: Create XXZ Hamiltonian MPO
#   - solve_xxz_hamiltonian_dmrg: Solve using DMRG algorithm
#   - create_mps: Initialize random MPS with caching
#   - find_ground_state_mps: Find ground state for given parameters
#
# HDF5 I/O functions:
#   - save_mps_with_params: Save MPS with parameters to HDF5
#   - load_mps_with_params: Load MPS with parameters from HDF5
#   - load_all_mps_from_file: Load all runs from HDF5 file
#   - find_runs_by_params: Find specific runs matching query parameters
#   - get_system_params_of_file: Extract system parameters from file
#
# System parameter functions:
#   - get_system_params: Get default system parameters
#   - extract_filename_from_system_params: Generate filename from parameters

# ==============================================================================
# Package Imports
# ==============================================================================

using Pkg
# NOTE: Do NOT call `Pkg.add` inside files that are `include`d. Installing packages
# during `include` can trigger environment changes and precompilation/version
# resolution while the file is being parsed, causing confusing errors.
#
# Install required packages from the REPL or ensure the project's environment
# (Project.toml/Manifest.toml) lists them. From the Julia REPL run:
#
#    ] activate .
#    ] add HDF5 ITensors ITensorMPS Statistics Printf Distributed Dates Logging
#    ] instantiate
#
using HDF5
using ITensors
using Dates
using Logging
using LinearAlgebra
using ITensorMPS
using Statistics
using Printf
using Distributed


# ==============================================================================
# DMRG and Ground State Functions
# ==============================================================================

const _sites_cache = Dict{Tuple{Int,Bool},Any}()

"""
    generate_fully_connected_wam(N::Int, σ::Float64, μ::Float64) -> Matrix{Float64}

Generate a randomly weighted adjacency matrix for a fully connected graph of N nodes.
Each edge weight is drawn from a normal distribution with mean μ and standard deviation σ.
"""
function generate_fully_connected_wam(N::Int, σ::Float64, μ::Float64)::Matrix{Float64}
    A = zeros(Float64, N, N)
    for i in 1:N
        for j in (i+1):N
            weight = μ + σ * randn()
            A[i, j] = weight
            A[j, i] = weight
        end
    end
    return A
end


"""
    create_xxz_hamiltonian_mpo(N::Int, A::Matrix{Float64}, J::Float64, Δ::Float64, sites) -> MPO

Create the XXZ Hamiltonian as an MPO given an adjacency matrix A.
The Hamiltonian includes XX, YY, and ZZ interaction terms scaled by the adjacency weights.
"""
function create_xxz_hamiltonian_mpo(N::Int, A::Matrix{Float64}, J::Float64, Δ::Float64, sites)::MPO
    mpo = OpSum()
    for i = 1:N-1
        for j = i+1:N
            weight = A[i, j]
            if weight != 0.0
                # XX and YY terms: S+S- + S-S+ = 2(SxSx + SySy)
                # So to get J(SxSx + SySy), we need J/2 * (S+S- + S-S+)
                mpo += weight * J / 2, "S+", i, "S-", j
                mpo += weight * J / 2, "S-", i, "S+", j
                # ZZ term
                mpo += weight * J * Δ, "Sz", i, "Sz", j
            end
        end
    end
    return MPO(mpo, sites)
end


"""
    solve_xxz_hamiltonian_dmrg(H::MPO, ψ0::MPS, num_sweeps::Int, bond_dim::Int, cutoff::Float64) -> (E::Float64, ψ::MPS)

Apply the DMRG algorithm to find the ground state of a Hamiltonian.
"""
function solve_xxz_hamiltonian_dmrg(H::MPO, ψ0::MPS, num_sweeps::Int, bond_dim::Int, cutoff::Float64)::Tuple{Float64,MPS}
    E, ψ = dmrg(H, ψ0;
            nsweeps = num_sweeps,
            maxdim = bond_dim,
            cutoff = cutoff,
            outputlevel = 1)
    return E, ψ
end


"""
    create_mps(N::Int; conserve_qns::Bool=true) -> (MPS, Vector{Index})

Create a random MPS for a spin-1/2 system of size N.
Sites are cached for efficiency to avoid recreating them repeatedly.
"""
function create_mps(N::Int; conserve_qns::Bool=true)::Tuple{MPS,Vector{Index{Vector{Pair{QN,Int64}}}}}
    key = (N, conserve_qns)
    sites = get(_sites_cache, key, nothing)
    if sites === nothing
        sites = siteinds("S=1/2", N; conserve_qns=conserve_qns)
        _sites_cache[key] = sites
    end
    return MPS(sites, [isodd(i) ? "Up" : "Dn" for i = 1:N]), sites
end


"""
    find_ground_state_mps(run_params::Dict{String,Any}, system_params::Dict{String,Any}) -> MPS

Find the ground state MPS for given run and system parameters.
Constructs the adjacency matrix, creates the Hamiltonian, and solves with DMRG.
"""
function find_ground_state_mps(run_params::Dict{String,Any}, system_params::Dict{String,Any})::MPS
    N = run_params["N"]
    σ = run_params["σ"]

    J = system_params["J"]
    Δ = system_params["Δ"]
    μ = system_params["μ"]
    NUM_SWEEPS = system_params["NUM_SWEEPS"]
    MAX_BOND_DIM = system_params["MAX_BOND_DIM"]
    ACC = system_params["ACC"]

    ψ, sites = create_mps(N)
    A = generate_fully_connected_wam(N, σ, μ)
    H = create_xxz_hamiltonian_mpo(N, A, J, Δ, sites)
    _, ψ_gs = solve_xxz_hamiltonian_dmrg(H, ψ, NUM_SWEEPS, MAX_BOND_DIM, ACC)
    return ψ_gs
end



"""
get_squared_sorted_schmidt_spectrum_from_mps(psi::MPS) -> Vector{Float64}  
Calculate the squared and sorted Schmidt spectrum from an MPS at the center bond.
This function orthogonalizes the MPS at the center bond, performs an SVD to extract the singular values, squares them to get the Schmidt coefficients, and returns them sorted in descending order.
"""

function get_squared_sorted_schmidt_spectrum_from_mps(psi::MPS)
    N = length(psi)
    b = N ÷ 2 
    
    # Ensure the MPS is in orthogonal form at the center bond
    psi_cp = copy(psi)
    orthogonalize!(psi_cp, b)
    
    # Perform SVD at bond b to get singular values S
    # Split the tensor at site b into (Link b-1, Site b) and (Link b)
    U, S, V = svd(psi_cp[b], (linkind(psi_cp, b-1), siteind(psi_cp, b)))
    
    # Extract singular values from the diagonal of S, square them, and sort
    svs = [S[i, i] for i in 1:dim(S, 1)]
    return sort(svs .^ 2, rev=true)
end


# ==============================================================================
# HDF5 I/O Helper Functions
# ==============================================================================

"""
    _read_params_from_group(g) -> Dict{String,Any}

Read parameters from an HDF5 group, converting all values to appropriate types.
"""
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


"""
    _write_params_to_group(g, params::Dict{String,Any})

Write parameters to an HDF5 group, with fallback to string representation if needed.
"""
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
    _params_match(existing::Dict{String,Any}, query::Dict{String,Any}) -> Bool

Check if existing parameters match a query, with tolerance for string representations.
"""
function _params_match(existing::Dict{String,Any}, query::Dict{String,Any})::Bool
    for (k, vq) in query
        ks = string(k)
        if !haskey(existing, ks)
            return false
        end
        ev = existing[ks]
        if ev == vq || string(ev) == string(vq)
            continue
        end
        return false
    end
    return true
end


# ==============================================================================
# HDF5 Save Functions
# ==============================================================================

"""
    save_mps_with_params(filepath::String, ψ::MPS, system_params::Dict, run_params::Dict; 
                        run_id::Union{String,Nothing}=nothing, param_safety::Bool=true) -> String

Save an MPS with system and run parameters to an HDF5 file.

Storage format:
- /system_params: System-wide parameters
- /runs/<run_id>/params: Run-specific parameters
- /runs/<run_id>/instances/<instance_id>/psi: The MPS data
- /runs/<run_id>/instances/<instance_id>/timestamp: Timestamp of save

Returns the run_id (either new or existing if params matched).
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
        # Handle system params
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
            existing = file["system_params"]
            for (k, v) in system_params
                if !haskey(existing, k)
                    @warn "system_params in file is missing key $k."
                    if param_safety
                        error("param_safety enabled; aborting due to missing key $k.")
                    end
                end
            end
        end

        # Prepare runs group
        runs_group = haskey(file, "runs") ? file["runs"] : create_group(file, "runs")

        # Find matching run
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
            # Append to existing run
            run_group = runs_group[matching_run_name]
            instances_group = haskey(run_group, "instances") ? run_group["instances"] : create_group(run_group, "instances")
            
            existing_inst_names = collect(keys(instances_group))
            nums = [tryparse(Int, string(nm)) for nm in existing_inst_names]
            nums = filter(!isnothing, nums)
            next_num = isempty(nums) ? 1 : maximum(nums) + 1
            inst_name = string(next_num)
            inst_group = create_group(instances_group, inst_name)

            write(inst_group, "psi", ψ)
            try
                inst_group["timestamp"] = string(Dates.now())
            catch end

            @info "Appended instance '$inst_name' to existing run '$matching_run_name'."
            return matching_run_name
        else
            # Create new run
            if run_id === nothing
                existing_names = collect(keys(runs_group))
                nums = [tryparse(Int, string(nm)) for nm in existing_names]
                nums = filter(!isnothing, nums)
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
            params_group = create_group(run_group, "params")
            _write_params_to_group(params_group, run_params)

            instances_group = create_group(run_group, "instances")
            inst_group = create_group(instances_group, "1")
            write(inst_group, "psi", ψ)
            try
                inst_group["timestamp"] = string(Dates.now())
            catch end

            @info "Created new run '$run_id'."
            return run_id
        end
    end
end


# ==============================================================================
# HDF5 Load Functions
# ==============================================================================

"""
    load_mps_with_params(filepath::String; run_id::Union{String,Nothing}=nothing) -> 
        (MPS, Dict{String,Any}, Dict{String,Any})

Load an MPS with system and run parameters from an HDF5 file.
If run_id is not specified, loads the most recent run.
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

        # Select run_id
        if run_id === nothing
            nums_map = Dict{Int,String}()
            for nm in run_names
                n = tryparse(Int, string(nm))
                if n !== nothing
                    nums_map[n] = string(nm)
                end
            end
            run_id = !isempty(nums_map) ? nums_map[maximum(keys(nums_map))] : sort(string.(run_names))[end]
        else
            if !in(string(run_id), run_names)
                error("Requested run_id `$(run_id)` not found. Available: $(run_names).")
            end
            run_id = string(run_id)
        end

        run_group = runs_group[run_id]
        if !haskey(run_group, "params")
            error("Run '$run_id' missing 'params' group.")
        end
        run_params = _read_params_from_group(run_group["params"])

        if !haskey(run_group, "instances")
            error("Run '$run_id' missing 'instances' group.")
        end

        instances_group = run_group["instances"]
        inst_names = collect(keys(instances_group))
        if isempty(inst_names)
            error("Run '$run_id' has empty 'instances' group.")
        end

        # Select most recent instance
        nums = [tryparse(Int, string(nm)) for nm in inst_names]
        nums = filter(!isnothing, nums)
        chosen_inst = !isempty(nums) ? string(maximum(nums)) : sort(string.(inst_names))[end]
        
        inst_group = instances_group[chosen_inst]
        psi = read(inst_group, "psi", MPS)

        (psi, system_params, run_params)
    end

    return result
end


"""
    get_system_params_of_file(filepath::String) -> Union{Dict{String,Any},Nothing}

Extract system parameters from an HDF5 file without loading MPS data.
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
    load_all_mps_from_file(filepath::String) -> (Dict{String,Any}, Vector{NamedTuple})

Load all runs from an HDF5 file with their MPS and parameters.
Returns (system_params, runs) where each run is a NamedTuple with:
  run_id, instance_id, psi, params, timestamp
"""
function load_all_mps_from_file(filepath::String)
    system_params = get_system_params_of_file(filepath)
    system_params === nothing && (system_params = Dict{String,Any}())

    runs = Vector{NamedTuple{(:run_id, :instance_id, :psi, :params, :timestamp),Tuple{String,String,MPS,Dict{String,Any},Union{String,Nothing}}}}()

    h5open(filepath, "r") do file
        if !haskey(file, "runs")
            @warn "No 'runs' group found in $filepath"
            return (system_params, runs)
        end

        runs_group = file["runs"]
        run_names = collect(keys(runs_group))
        isempty(run_names) && (@warn "No runs found"; return (system_params, runs))

        # Order runs numerically if possible
        nums_map = Dict{Int,String}()
        for nm in run_names
            n = tryparse(Int, string(nm))
            n !== nothing && (nums_map[n] = string(nm))
        end
        ordered_run_names = !isempty(nums_map) ? [nums_map[k] for k in sort(collect(keys(nums_map)))] : sort(string.(run_names))

        for rn in ordered_run_names
            run_group = runs_group[rn]
            if !haskey(run_group, "params") || !haskey(run_group, "instances")
                @warn "Skipping run $rn (missing params or instances)."
                continue
            end
            run_params = _read_params_from_group(run_group["params"])

            instances_group = run_group["instances"]
            inst_names = collect(keys(instances_group))
            isempty(inst_names) && continue

            # Order instances numerically if possible
            nums = [tryparse(Int, string(nm)) for nm in inst_names]
            nums = filter(!isnothing, nums)
            ordered_inst_names = !isempty(nums) ? [string(i) for i in sort(nums)] : sort(string.(inst_names))

            for inst in ordered_inst_names
                inst_group = instances_group[inst]
                psi = read(inst_group, "psi", MPS)
                timestamp = haskey(inst_group, "timestamp") ? string(inst_group["timestamp"]) : nothing
                push!(runs, (run_id=string(rn), instance_id=string(inst), psi=psi, params=run_params, timestamp=timestamp))
            end
        end
    end

    return (system_params, runs)
end


"""
    find_runs_by_params(filepath::String, query_params::Dict{String,Any}; 
                       load_psi::Bool=false, instance_selection::Union{String,Symbol,Nothing}=:latest) 
        -> Vector{NamedTuple}

Find runs in an HDF5 file matching specific query parameters.

Keyword arguments:
- load_psi::Bool: If true, reads and returns MPS data (can be slow for large files)
- instance_selection::Union{Symbol,String}:
    - :latest (default): Most recent instance of each matching run
    - :all: All instances of matching runs
    - String/Int: Specific instance ID

Returns Vector of NamedTuples with: run_id, instance_id, psi, params, timestamp
"""
function find_runs_by_params(
    filepath::String,
    query_params::Dict{String, Any};
    load_psi::Bool=false,
    instance_selection::Union{String,Symbol,Nothing}=:latest
)
    matches = Vector{NamedTuple{(:run_id,:instance_id,:psi,:params,:timestamp),Tuple{String,String,Union{MPS,Nothing},Dict{String,Any},Union{String,Nothing}}}}()

    !isfile(filepath) && (@warn "File does not exist: $filepath"; return matches)

    h5open(filepath, "r") do file
        !haskey(file, "runs") && (@info "No 'runs' group in $filepath"; return matches)
        
        runs_group = file["runs"]
        for rn in keys(runs_group)
            run_group = runs_group[string(rn)]
            if !haskey(run_group, "params") || !haskey(run_group, "instances")
                continue
            end
            
            existing_params = _read_params_from_group(run_group["params"])
            !_params_match(existing_params, query_params) && continue

            instances_group = run_group["instances"]
            inst_names = collect(keys(instances_group))
            isempty(inst_names) && continue

            # Order instances
            nums = [tryparse(Int, string(nm)) for nm in inst_names]
            nums = filter(!isnothing, nums)
            ordered_inst_names = !isempty(nums) ? [string(i) for i in sort(nums)] : sort(string.(inst_names))

            # Select instances
            selected_instances = String[]
            if instance_selection === :all
                append!(selected_instances, ordered_inst_names)
            elseif instance_selection === :latest
                push!(selected_instances, ordered_inst_names[end])
            else
                sel = string(instance_selection)
                sel in ordered_inst_names ? push!(selected_instances, sel) : (@warn "Instance $sel not found"; continue)
            end

            for inst in selected_instances
                inst_group = instances_group[inst]
                timestamp = haskey(inst_group, "timestamp") ? string(inst_group["timestamp"]) : nothing
                psi_val = nothing
                if load_psi
                    psi_val = read(inst_group, "psi", MPS)
                end
                push!(matches, (run_id=string(rn), instance_id=string(inst), psi=psi_val, params=existing_params, timestamp=timestamp))
            end
        end
    end

    return matches
end


# ==============================================================================
# System Parameters
# ==============================================================================

"""
    get_system_params() -> Dict{String,Any}

Get default system parameters for simulations.
"""
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


"""
    extract_filename_from_system_params(params::Dict{String,Any}) -> String

Generate a standardized filename string from system parameters.
Formats floating point values in scientific notation.
"""
function extract_filename_from_system_params(params::Dict{String,Any})::String
    string = ""
    for (key, value) in params
        if !isempty(string)
            string *= "_"
        end
        formatted_value = value isa AbstractFloat ? @sprintf("%.2e", value) : value
        string *= "$(key)=$(formatted_value)"
    end
    return string
end
