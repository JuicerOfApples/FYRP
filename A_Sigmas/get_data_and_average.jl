using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads
using Printf

# Hard-coded truncation cutoff as requested
current_cutoff = 1e-16

Random.seed!(1234);
const io_lock = ReentrantLock()

function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = MPS(sites, initial_state) 
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
        r = randn()
        while abs(r) > 2.0
            r = randn()
        end
        weight = μ + σ * r
        A[i, j] = A[j, i] = weight
    end
    return A
end

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
    H = MPO(ampo, sites)
    return truncate!(H; cutoff=1e-15)
end

function get_entropies_and_spectrum(ψ::MPS, N::Int, max_dim::Int)
    center_bond = N ÷ 2
    orthogonalize!(ψ, center_bond)
    
    l_ind = linkind(ψ, center_bond - 1)
    s_ind = siteind(ψ, center_bond)
    inds_to_svd = isnothing(l_ind) ? (s_ind,) : (l_ind, s_ind)
    
    _, S, _ = svd(ψ[center_bond], inds_to_svd...)
    
    sv = [S[i, i] for i in 1:dim(S, 1)]
    p = sv .^ 2
    p ./= sum(p) 
    
    vn = -sum(x -> x > 1e-18 ? x * log(x) : 0.0, p)
    s_05 = (1 / (1 - 0.5)) * log(sum(sqrt.(p)))
    s_0 = log(count(x -> x > 1e-16, p))
 
    padded_p = zeros(Float64, max_dim)
    len = min(length(p), max_dim)
    padded_p[1:len] = p[1:len]
    
    return vn, s_05, s_0, padded_p
end

function run_simulation_avg_err(
    avg_matrix::Matrix{Float64}, err_matrix::Matrix{Float64},
    vn_avg::Matrix{Float64}, vn_err::Matrix{Float64},
    s05_avg::Matrix{Float64}, s05_err::Matrix{Float64},
    s0_avg::Matrix{Float64}, s0_err::Matrix{Float64},
    spectra_avg::Array{Float64, 3},
    N_range, sigma_values, num_graphs_avg::Int,
    num_sweeps::Int, max_bond_dim_limit::Int,
    cutoff::Float64, μ::Float64, filename::String
)
    println("Starting simulation for cutoff $(cutoff) on $(Threads.nthreads()) threads...")
    flush(stdout)

    Threads.@threads for i in 1:length(N_range)
        N = N_range[i]
        for (j, σ) in enumerate(sigma_values)
            if avg_matrix[i, j] != 0.0
                continue
            end

            bond_dims = zeros(Float64, num_graphs_avg)
            vns = zeros(Float64, num_graphs_avg)
            s05s = zeros(Float64, num_graphs_avg)
            s0s = zeros(Float64, num_graphs_avg)
            spec_sum = zeros(Float64, max_bond_dim_limit)
            
            for k in 1:num_graphs_avg
                ψ₀, sites = create_MPS(N)
                adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
                H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=-1.0, Δ=-1.0)

                sweeps = Sweeps(num_sweeps)
                setmaxdim!(sweeps, max_bond_dim_limit)
                setcutoff!(sweeps, cutoff)

                _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)
    
                bond_dims[k] = maxlinkdim(ψ_gs)
                vn, s05, s0, spec = get_entropies_and_spectrum(ψ_gs, N, max_bond_dim_limit)
                vns[k] = vn
                s05s[k] = s05
                s0s[k] = s0
                spec_sum .+= spec
            end

            avg_matrix[i, j] = mean(bond_dims)
            err_matrix[i, j] = std(bond_dims)
            vn_avg[i, j] = mean(vns)
            vn_err[i, j] = std(vns)
            s05_avg[i, j] = mean(s05s)
            s05_err[i, j] = std(s05s)
            s0_avg[i, j] = mean(s0s)
            s0_err[i, j] = std(s0s)
            spectra_avg[i, j, :] .= spec_sum ./ num_graphs_avg
            
            println("Done: N=$N, σ=$σ, EPS=$cutoff | Avg BD: $(avg_matrix[i, j])")
            flush(stdout)
        end

        lock(io_lock) do
            try
                jldsave(filename; 
                    avg_matrix, err_matrix, vn_avg, vn_err, 
                    s05_avg, s05_err, s0_avg, s0_err,
                    spectra_avg, N_range, sigma_values)
                println("Checkpoint saved for N=$N, EPS=$cutoff")
                flush(stdout)
            catch e
                @warn "Checkpoint failed for N=$N: $e"
                flush(stdout)
            end
        end
    end
end

N_range = 2:2:80
sigma_values = [0.0, 1e-7, 1e-6, 1e-5]
num_graphs_avg = 10
num_sweeps = 30
max_bond_dim_limit = 1000
μ = 1.0

# Simplified filename for the single cutoff
filename = joinpath(@__DIR__, "data_more_1e-16.jld2")

avg_matrix = zeros(Float64, length(N_range), length(sigma_values))
err_matrix = zeros(Float64, length(N_range), length(sigma_values))
vn_avg = zeros(Float64, length(N_range), length(sigma_values))
vn_err = zeros(Float64, length(N_range), length(sigma_values))
s05_avg = zeros(Float64, length(N_range), length(sigma_values))
s05_err = zeros(Float64, length(N_range), length(sigma_values))
s0_avg = zeros(Float64, length(N_range), length(sigma_values))
s0_err = zeros(Float64, length(N_range), length(sigma_values))
spectra_avg = zeros(Float64, length(N_range), length(sigma_values), max_bond_dim_limit)

if isfile(filename)
    println("\nResuming from existing file: $filename")
    flush(stdout)
    data = load(filename)
    if haskey(data, "vn_avg") && data["N_range"] == N_range && data["sigma_values"] == sigma_values
        global avg_matrix = data["avg_matrix"]
        global err_matrix = data["err_matrix"]
        global vn_avg = data["vn_avg"]
        global vn_err = data["vn_err"]
        global s05_avg = data["s05_avg"]
        global s05_err = data["s05_err"]
        global s0_avg = data["s0_avg"]
        global s0_err = data["s0_err"]
        global spectra_avg = data["spectra_avg"]
    else
        println("Parameter mismatch detected. Starting fresh for $filename.")
        flush(stdout)
    end
else
    println("\nCreating new file: $filename")
end

run_simulation_avg_err(
    avg_matrix, err_matrix, vn_avg, vn_err,
    s05_avg, s05_err, s0_avg, s0_err,
    spectra_avg, N_range, sigma_values,
    num_graphs_avg, num_sweeps, max_bond_dim_limit, current_cutoff, μ, filename
)

println("Simulation for EPS=$(current_cutoff) complete. Data saved to $filename")
flush(stdout)