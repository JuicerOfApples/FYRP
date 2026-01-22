using LinearAlgebra
using ITensors
using ITensorMPS
using JLD2, FileIO
using Statistics
using Random

N_vals = [6, 10, 14, 18, 22, 26, 30, 40, 50, 60, 70, 80]
σ_vals = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]           
μ = 1.0
num_sweeps = 20

max_bond_dim = 1000 # could not get this to simulate with anything more than this even on HPC
precision_cutoff = eps(Float64) 


function gen_full_conn(N::Int, σ::Float64; μ::Float64=1.0)
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

function create_xxz_mpo(N::Int, A::Matrix{Float64}, sites)
    J = -1.0
    Δ = -1.0
    
    os = OpSum()
    for i in 1:N
        for j in (i+1):N
            if A[i,j] != 0.0
                os += A[i,j] * (J/2), "S+", i, "S-", j
                os += A[i,j] * (J/2), "S-", i, "S+", j
                os += A[i,j] * (J*Δ), "Sz", i, "Sz", j
            end
        end
    end
    return MPO(os, sites)
end


output_path = joinpath(@__DIR__, "high_prec_data.jld2")

data = isfile(output_path) ? load(output_path) : Dict{String, Any}()

println("Starting simulation loop...")

# Simple nested loop
for N in N_vals
    for σ in σ_vals
        println("Processing N=$N, σ=$σ ...")
        flush(stdout)

        # Setup Model
        A = gen_full_conn(N, σ; μ=μ)
        sites = siteinds("S=1/2", N)
        H = create_xxz_mpo(N, A, sites)
        psi0 = randomMPS(sites, 2)
        
        # Setup DMRG Sweeps
        sweeps = Sweeps(num_sweeps)
        setmaxdim!(sweeps, max_bond_dim)
        setcutoff!(sweeps, precision_cutoff) 
        setnoise!(sweeps, 1E-6, 1E-10, 0.0)

        # Run DMRG
        energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)
        
        # Calculate Entanglement Spectrum
        center_b = N ÷ 2
        orthogonalize!(psi, center_b)
        u, s, v = svd(psi[center_b], (linkind(psi, center_b-1), siteind(psi, center_b)))
        
        sv = Float64[]
        for n in 1:dim(s, 1)
            push!(sv, s[n, n])
        end 
        
        norm_factor = sqrt(sum(sv.^2))
        sv = sv ./ norm_factor
        
        # Save Data
        # We save the singular values (sv). 
        # From 'sv', you can calculate Renyi Entropy for ANY alpha (0 to 1),
        # as well as Von Neumann entropy and truncation errors.
        key = "N=$N/sigma=$σ"
        data[key] = sv
        
        # Save immediately after each run
        save(output_path, data)
        
        println("  > Finished N=$N, σ=$σ. Saved.")
        flush(stdout)
        # Clean up memory
        GC.gc()
    end
end

println("All simulations complete.")