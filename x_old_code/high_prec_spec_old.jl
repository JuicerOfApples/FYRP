using Random, Statistics
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads

# Push DMRG to EPS of the machine -> Cutoff 1E-14
cutoff_val = 1E-14 
maxdim_val = 3000   
num_sweeps = 10
num_graphs_avg = 5

N_values = 10:10:100

target_sigmas = [0.002]

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

function run_simulation(N_vals, sigma_val, n_graphs)
    results = Dict{Int, Vector{Float64}}()
    
    for N in N_vals
        println("Simulating N = $N, σ = $sigma_val (High Precision)")
        
        local_coeffs = Vector{Float64}()
        
        for g in 1:n_graphs
            sites = siteinds("S=1/2", N; conserve_qns=true)
            
            initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:N]
            psi0 = randomMPS(sites, initial_state)
            
            adj = create_weighted_adj_mat(N, sigma_val)
            os = OpSum()
            for i in 1:N, j in (i+1):N
                J_ij = adj[i,j]
                os += -J_ij, "Sz", i, "Sz", j
                os += 0.5 * -J_ij, "S+", i, "S-", j
                os += 0.5 * -J_ij, "S-", i, "S+", j
            end
            H = MPO(os, sites)
            
            sweeps = Sweeps(num_sweeps)
            setmaxdim!(sweeps, 10, 20, 100, maxdim_val)
            setcutoff!(sweeps, 1E-5, 1E-8, 1E-10, cutoff_val) 
            
            energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)
            
            b_idx = N ÷ 2
            orthogonalize!(psi, b_idx)
            U, S, V = svd(psi[b_idx], (linkind(psi, b_idx-1), siteind(psi, b_idx)))
            
            raw_coeffs = Float64[]
            for n in 1:dim(S, 1)
                val = S[n, n]
                push!(raw_coeffs, val)
            end
            
            # Filter small singular values to maintain precision
            filter!(x -> x > 1E-18, raw_coeffs) 
            
            append!(local_coeffs, raw_coeffs)
        end
        
        sort!(local_coeffs, rev=true) 
        results[N] = local_coeffs

        fname = joinpath(@__DIR__, "high_prec_spec_data_$(sigma_val).jld2")
        save(fname, Dict(
            "entanglement_spectrum_results" => results,
            "N_values" => N_vals, 
            "σ_val" => sigma_val,
            "precision" => "1E-14"
        ))
        println("  Checkpoint saved for N = $N")
    end
    return results
end

for sig in target_sigmas
    run_simulation(N_values, sig, num_graphs_avg)
end