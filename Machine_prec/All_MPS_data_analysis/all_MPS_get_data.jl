

using ITensors
using ITensorMPS
using JLD2
using LinearAlgebra
using Random
using Printf
using Statistics

# --- 1. System Setup Functions ---

function create_MPS(L::Int)
    # siteinds is an ITensors function. 
    # If this fails, ITensors is not loaded.
    sites = ITensors.siteinds("S=1/2", L; conserve_qns=true)
    
    # Start with a Néel state to avoid local minima
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = ITensorMPS.MPS(sites, initial_state) 
    return ψ₀, sites
end

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

function create_weighted_xxz_mpo(N::Int, adj_mat, sites; J::Float64, Δ::Float64)
    ampo = OpSum()
    for i in 1:N-1
        for j in (i+1):N 
            coupling_strength = adj_mat[i, j]
            if abs(coupling_strength) > 1E-14 # Filter negligible terms
                ampo += coupling_strength * (J / 2), "S+", i, "S-", j
                ampo += coupling_strength * (J / 2), "S-", i, "S+", j
                ampo += coupling_strength * (J * Δ), "Sz", i, "Sz", j
            end
        end
    end
    return MPO(ampo, sites)
end


function run_hpc_simulation()
    # HPC / Precision Constants
    MAX_BOND_DIM = typemax(Int) 
    ACC = eps(Float64)          # Machine epsilon (~2.22e-16)
    
    # Simulation Parameters
    N_range = 2:2:40            # Iterate from 2 to 40 in steps of 2
    sigma_values = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05] 
    
    J_val = -1.0
    Delta_val = -1.0
    mu_val = 1.0

    # DMRG Schedule (Arrays define the behavior per sweep)
    # We use 20 sweeps (MIN_SWEEPS)
    nsweeps = 20
    maxdim_schedule = [50, 100, 200, 400, 800, 2000, MAX_BOND_DIM]
    cutoff_schedule = ACC # Use machine precision cutoff
    noise_schedule = [1E-7, 1E-8, 1E-10, 0.0] 

    filename = joinpath(@__DIR__, "all_MPS_data.jld2")
    
    # Load existing data or create new
    if isfile(filename)
        println("Found existing data file. Loading to append...")
        # Note: JLD2 load syntax can vary. 
        # Safest way is to load the dictionary object directly.
        file_content = load(filename)
        if haskey(file_content, "results_db")
            results_db = file_content["results_db"]
        else
            results_db = Dict()
        end
    else
        results_db = Dict()
    end

    for N in N_range
        println("\n>>> Processing System Size N = $N")
        
        # Create sites once per N
        psi_template, sites = create_MPS(N)
        
        for σ in sigma_values
            key = (N, σ)
            if haskey(results_db, key)
                println("  Skipping N=$N, σ=$σ (Already computed)")
                continue
            end

            println("  Running σ = $σ ...")
            flush(stdout) 
            
            # Setup System
            Random.seed!(1234)
            adj_mat = create_weighted_adj_mat(N, σ; μ=mu_val)
            H = create_weighted_xxz_mpo(N, adj_mat, sites; J=J_val, Δ=Delta_val)
            psi0 = copy(psi_template)
            
            try
                # Run DMRG using Keyword Arguments (Modern ITensors Syntax)
                energy, psi = dmrg(H, psi0; 
                                   nsweeps=nsweeps, 
                                   maxdim=maxdim_schedule, 
                                   cutoff=cutoff_schedule, 
                                   noise=noise_schedule, 
                                   outputlevel=0)
                
                # Analysis: Spectrum at Center Bond
                center_b = N ÷ 2
                orthogonalize!(psi, center_b)
                
                # Robustly identify indices for the SVD cut
                # uniqueinds(A, B) returns indices in A that are NOT in B.
                # This automatically grabs the site index and the left link (if it exists).
                left_inds = uniqueinds(psi[center_b], psi[center_b+1])
                
                u, s, v = svd(psi[center_b], left_inds)
                
                # Extract eigenvalues (squared singular values)
                # ITensors DiagonalTensor storage access:
                spectrum = [s[i, i]^2 for i in 1:dim(inds(s)[1])]
                
                # Store Data
                data_entry = Dict(
                    "N" => N,
                    "sigma" => σ,
                    "mps" => psi,
                    "energy" => energy,
                    "spectrum" => spectrum,
                    "max_bd" => maxlinkdim(psi),
                    "avg_bd" => mean(linkdims(psi))
                )
                
                results_db[key] = data_entry
                
                @printf("    [Success] E=%.8f, MaxBD=%d\n", energy, maxlinkdim(psi))
                
            catch e
                println("    [Error] Simulation failed for N=$N, σ=$σ")
                println("    Error details: $e")
                if isa(e, OutOfMemoryError)
                    println("    CRITICAL: Out of Memory. Stopping this N.")
                    break 
                end
            end
        end
        
        # Save Checkpoint
        save(filename, "results_db", results_db)
        println("  (Checkpoint saved to $filename)")
    end
end

function main()
    run_hpc_simulation()
end

main()