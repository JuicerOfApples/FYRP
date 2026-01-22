using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2

Random.seed!(1234);


function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    # Initialize as a Néel state to help convergence
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = MPS(sites, initial_state) 
    return ψ₀, sites
end

"""
Creates a weighted adjacency matrix.
"""
function create_weighted_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    # Returns uniform all-to-all if sigma is 0 (Clean System)
    if σ == 0.0
        A = ones(Float64, N, N)
        A -= Matrix{Float64}(I, N, N)
        return A
    end
    
    # Returns weighted matrix (Noise System)
    # If μ=0 and σ=1, weights are purely centered on 0
    A = zeros(Float64, N, N)
    for i in 1:N, j in (i+1):N
        weight = μ + σ * randn()
        A[i, j] = A[j, i] = weight
    end
    return A
end

"""
Creates the MPO for the XXZ Hamiltonian.
"""
function create_weighted_xxz_mpo(N::Int, adj_mat, sites; J::Float64, Δ::Float64)
    ampo = OpSum()
    for i in 1:N-1
        for j in (i+1):N 
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


function run_simulation_save_full_states(
    N_range,
    num_sweeps::Int,
    max_bond_dim_limit::Int,
    cutoff::Float64
)
    filename = joinpath(@__DIR__, "noise_vs_clean_data.jld2")

    # Dictionaries to store the resulting MPS objects
    # Key = N (Int), Value = MPS
    results_clean = Dict{Int, MPS}()
    results_noise = Dict{Int, MPS}()
    
    # Check for existing data to resume
    if isfile(filename)
        println("Found existing data. Loading...")
        try
            f = jldopen(filename, "r")
            if haskey(f, "results_clean")
                results_clean = read(f, "results_clean")
            end
            if haskey(f, "results_noise")
                results_noise = read(f, "results_noise")
            end
            close(f)
        catch e
            println("Error loading file (starting fresh): $e")
        end
    end

    # Iterate sequentially
    for N in N_range
        
        # Check if this N is already done for both cases
        if haskey(results_clean, N) && haskey(results_noise, N)
             println("Skipping N = $N, states already saved.")
             continue
        end

        println("Running N = $N...")

        # Common sweeps config
        sweeps = Sweeps(num_sweeps)
        setmaxdim!(sweeps, max_bond_dim_limit)
        setcutoff!(sweeps, cutoff)
        
        if !haskey(results_clean, N)
            println("  - Simulating Clean System...")
            ψ₀_clean, sites_clean = create_MPS(N)
            adj_clean = create_weighted_adj_mat(N, 0.0; μ=1.0)
            H_clean = create_weighted_xxz_mpo(N, adj_clean, sites_clean; J=-1.0, Δ=-1.0)
            
            _, ψ_gs_clean = dmrg(H_clean, ψ₀_clean, sweeps; outputlevel=0)
            
            # Save to dictionary
            results_clean[N] = ψ_gs_clean
        end

        if !haskey(results_noise, N)
            println("  - Simulating Noise System...")
            ψ₀_noise, sites_noise = create_MPS(N)
            adj_noise = create_weighted_adj_mat(N, 1.0; μ=0.0)
            H_noise = create_weighted_xxz_mpo(N, adj_noise, sites_noise; J=-1.0, Δ=-1.0)
            
            _, ψ_gs_noise = dmrg(H_noise, ψ₀_noise, sweeps; outputlevel=0)
            
            # Save to dictionary
            results_noise[N] = ψ_gs_noise
        end

        # Save checkpoint to disk
        println("  - Saving checkpoint for N = $N...")
        jldsave(filename; results_clean, results_noise, N_range)
    end

end


N_range = 2:2:60 

num_sweeps = 30
max_bond_dim_limit = 250
cutoff = 1E-10

run_simulation_save_full_states(
    N_range,
    num_sweeps,
    max_bond_dim_limit,
    cutoff
)