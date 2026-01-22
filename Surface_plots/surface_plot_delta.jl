using Random
using ITensors, ITensorMPS, LinearAlgebra
using Statistics
using JLD2
using Base.Threads  

Random.seed!(1234);

function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = randomMPS(sites, initial_state)
    return ψ₀, sites
end

"""
Creates a weighted adjacency matrix for a completely connected graph.
"""
function create_weighted_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    if σ == 0.0
        A = ones(Float64, N, N)
        A -= Matrix{Float64}(I, N, N)
        return A
    end
    A = zeros(Float64, N, N)
    for i in 1:N, j in (i+1):N
        weight = μ + σ * randn()
        A[i, j] = A[j, i] = weight
    end
    return A
end

"""
Creates the MPO for the XXZ Hamiltonian on a graph with weighted interactions.
"""
function create_weighted_xxz_mpo(N::Int, adj_mat, sites; J::Float64, Δ::Float64)
    ampo = OpSum()
    for i in 1:N-1
        for j in i+1:N
            coupling_strength = adj_mat[i, j]
            if coupling_strength != 0.0
                ampo += coupling_strength * (J / 2), "S+", i, "S-", j
                ampo += coupling_strength * (J / 2), "S-", i, "S+", j
                ampo += coupling_strength * (J * Δ), "Sz", i, "Sz", j
            end
        end
    end
    return MPO(ampo, sites)
end

function run_simulation_delta(
    avg_bond_dims::Array{Float64, 3},
    N_range,
    delta_range,
    sigma_values,
    num_graphs_avg::Int,
    num_sweeps::Int,
    max_bond_dim_limit::Int,
    cutoff::Float64,
    μ::Float64,
    J::Float64
)

    filename = joinpath(@__DIR__, "surface_plot_delta_data(-1.0).jld2")

    for (i, N) in enumerate(N_range)
        
        # Checkpoint: if the first entry for this N is filled, skip
        if avg_bond_dims[i, 1, 1] != 0.0
            println("Skipping N = $N, results already loaded/computed.")
            continue
        end

        Threads.@threads for j in 1:length(delta_range)
            Δ_val = delta_range[j] 

            for (k, σ) in enumerate(sigma_values)
                
                bond_dims_for_avg = zeros(Float64, num_graphs_avg)
                
                for g in 1:num_graphs_avg
                    ψ₀, sites = create_MPS(N)
                    adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
                    H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=J, Δ=Δ_val)

                    sweeps = Sweeps(num_sweeps)
                    setmaxdim!(sweeps, max_bond_dim_limit)
                    setcutoff!(sweeps, cutoff)

                    _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)
                    
                    bond_dims_for_avg[g] = maxlinkdim(ψ_gs)
                end
                
                avg_bond_dims[i, j, k] = mean(bond_dims_for_avg)
            end
        end
        println("Completed N = $N")

        # Save checkpoint after each N
        try
            jldsave(filename; avg_bond_dims, N_range, delta_range, sigma_values)
            println("Checkpoint saved for N = $N")
        catch e
            println("WARNING: Could not save checkpoint for N = $N. Error: $e")
        end
    end
    
end


N_range = 10:1:100  
delta_range = [-1.0, -0.5, 0.0, 0.5, 1.0] 
sigma_values = [0.0, 0.002] 
num_graphs_avg = 10      
num_sweeps = 30
max_bond_dim_limit = 250
cutoff = 1E-10
μ = 1.0
J = -1.0

filename = joinpath(@__DIR__, "surface_plot_delta_data(-1.0).jld2")

if isfile(filename)
    println("Found existing data file. Loading progress...")
    try
        loaded_data = jldopen(filename, "r")
        N_range_loaded = read(loaded_data, "N_range")
        delta_range_loaded = read(loaded_data, "delta_range")
        sigma_values_loaded = read(loaded_data, "sigma_values")

        # Check if parameters match
        if N_range_loaded == N_range && 
           delta_range_loaded == delta_range && 
           sigma_values_loaded == sigma_values
            
            println("Parameters match. Resuming...")
            global avg_bond_dims = read(loaded_data, "avg_bond_dims") 
        else
            println("WARNING: Parameters in file do not match current script.")
            global avg_bond_dims = zeros(Float64, length(N_range), length(delta_range), length(sigma_values))
        end
        close(loaded_data)
    catch e
        println("WARNING: Could not load existing file. Error: $e")
        global avg_bond_dims = zeros(Float64, length(N_range), length(delta_range), length(sigma_values))
    end
else
    println("No existing data file found.")
    global avg_bond_dims = zeros(Float64, length(N_range), length(delta_range), length(sigma_values))
end


run_simulation_delta(
    avg_bond_dims,
    N_range,
    delta_range,
    sigma_values,
    num_graphs_avg,
    num_sweeps,
    max_bond_dim_limit,
    cutoff,
    μ,
    J
)

jldsave(filename; avg_bond_dims, N_range, delta_range, sigma_values)
println("Data saved successfully.\n")