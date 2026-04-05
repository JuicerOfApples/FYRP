using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Printf

LinearAlgebra.BLAS.set_num_threads(1)
Random.seed!(1234)


if length(ARGS) < 1
    error("Usage: julia MPO_get_data.jl <sigma>")
end

sigma_str = ARGS[1]
σ = parse(Float64, sigma_str)


N_range = collect(2:2:100)
EPS = 1e-16
num_graphs_avg = 10
μ = 1.0


function create_weighted_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    if σ == 0.0
        A = ones(Float64, N, N)
        A -= Matrix{Float64}(I, N, N)
        return A
    end
    A = zeros(Float64, N, N)
    for i in 1:N, j in (i+1):N
        r = randn()
        while abs(r) > 2.0
            r = randn()
        end
        weight = μ + σ * r
        A[i, j] = A[j, i] = weight
    end
    return A
end

function create_weighted_xxz_mpo(N::Int, adj_mat, sites; J::Float64, Δ::Float64, cutoff::Float64)
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
    H = MPO(ampo, sites)
    return truncate!(H; cutoff=cutoff)
end


flush(stdout)

filename = joinpath(@__DIR__, "mpo_scaling_data_sigma_$(sigma_str).jld2")

mpo_bd_avg = zeros(Float64, length(N_range))
mpo_bd_err = zeros(Float64, length(N_range))

println("Starting evaluation for σ = $σ")

for (n_idx, N) in enumerate(N_range)
    sites = siteinds("S=1/2", N; conserve_qns=true)
    
    bds = zeros(Float64, num_graphs_avg)
    
    for k in 1:num_graphs_avg
        adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
        
        # Create MPO and truncate using the EPS cutoff
        H = create_weighted_xxz_mpo(N, adj_mat, sites; J=-1.0, Δ=-1.0, cutoff=EPS)
        
        bds[k] = maxlinkdim(H)
    end
    
    mpo_bd_avg[n_idx] = mean(bds)
    mpo_bd_err[n_idx] = std(bds)
    
    flush(stdout)
end

println("Completed σ = $σ")

try
    jldopen(filename, "w"; compress=true) do file
        file["mpo_bd_avg"] = mpo_bd_avg
        file["mpo_bd_err"] = mpo_bd_err
        file["N_range"] = N_range
        file["sigma"] = σ
        file["EPS"] = EPS
    end
    println("Saved MPO scaling data successfully to $filename")
catch e
    println("Warning: Failed to save. Error: $e")
end