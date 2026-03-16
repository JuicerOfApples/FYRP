using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Printf

# Disable BLAS multithreading
LinearAlgebra.BLAS.set_num_threads(1)
Random.seed!(1234)

# ==============================================================================
# 1. Parse Command Line Arguments
# ==============================================================================
if length(ARGS) < 1
    error("Usage: julia get_data_clean.jl <cutoff>")
end

current_cutoff = parse(Float64, ARGS[1])
σ = 0.0 # Clean case strictly forced

# Parameters
N_range = collect(5:5:200)
num_sweeps = 30
max_bond_dim_limit = 2000
μ = 1.0

println("Running Clean State (σ = 0.0) | Cutoff = $current_cutoff")

# ==============================================================================
# 2. Helper Functions
# ==============================================================================
function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = MPS(sites, initial_state) 
    return ψ₀, sites
end

function create_weighted_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    # For σ = 0.0, this safely returns an all-to-all matrix of 1.0s (minus diagonal)
    if σ == 0.0
        A = ones(Float64, N, N)
        A -= Matrix{Float64}(I, N, N)
        return A
    end
    return A # Unreachable in this script, but kept for consistency
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

# ==============================================================================
# 3. Main Simulation Loop
# ==============================================================================
flush(stdout)

sweeps = Sweeps(num_sweeps)
setmaxdim!(sweeps, max_bond_dim_limit)
setcutoff!(sweeps, current_cutoff)

sigma_str = "0.0e+00"
cutoff_str = @sprintf("%.1e", current_cutoff)
filename = joinpath(@__DIR__, "data_fid_$(cutoff_str)_$(sigma_str).jld2")

# Arrays for metrics (Errors will remain 0.0 since there is only 1 graph)
avg_arr = zeros(Float64, length(N_range)); err_arr = zeros(Float64, length(N_range))
vn_avg = zeros(Float64, length(N_range)); vn_err = zeros(Float64, length(N_range))
s05_avg = zeros(Float64, length(N_range)); s05_err = zeros(Float64, length(N_range))
s0_avg = zeros(Float64, length(N_range)); s0_err = zeros(Float64, length(N_range))
energy_avg = zeros(Float64, length(N_range)); energy_err = zeros(Float64, length(N_range))

# Dummy Arrays for Fidelity (Clean state compared to itself is exactly 1.0)
fid_avg = ones(Float64, length(N_range)); fid_err = zeros(Float64, length(N_range))
infid_avg = zeros(Float64, length(N_range)); infid_err = zeros(Float64, length(N_range))

spectra_avg = zeros(Float64, length(N_range), max_bond_dim_limit)

for i in 1:length(N_range)
    N = N_range[i]
    
    # --- Compute the single Clean State ---
    ψ₀_clean, sites = create_MPS(N)
    adj_mat_clean = create_weighted_adj_mat(N, 0.0; μ=μ)
    H_clean = create_weighted_xxz_mpo(N, adj_mat_clean, sites; J=-1.0, Δ=-1.0, cutoff=current_cutoff)
    
    energy_clean, ψ_clean = dmrg(H_clean, ψ₀_clean, sweeps; outputlevel=0)
    
    # --- Extract Metrics ---
    vn, s05, s0, spec = get_entropies_and_spectrum(ψ_clean, N, max_bond_dim_limit)
    
    # Record to arrays
    avg_arr[i] = maxlinkdim(ψ_clean)
    vn_avg[i] = vn
    s05_avg[i] = s05
    s0_avg[i] = s0
    energy_avg[i] = energy_clean
    spectra_avg[i, :] .= spec

    println("Done: N=$N | Clean E: $(energy_avg[i]) | Clean BD: $(avg_arr[i])")
    flush(stdout)

    # --- Robust Save Method ---
    try
        jldopen(filename, "w"; compress=true) do file
            file["avg_arr"] = avg_arr
            file["err_arr"] = err_arr
            file["vn_avg"] = vn_avg
            file["vn_err"] = vn_err
            file["s05_avg"] = s05_avg
            file["s05_err"] = s05_err
            file["s0_avg"] = s0_avg
            file["s0_err"] = s0_err
            file["energy_avg"] = energy_avg
            file["energy_err"] = energy_err
            file["fid_avg"] = fid_avg
            file["fid_err"] = fid_err
            file["infid_avg"] = infid_avg
            file["infid_err"] = infid_err
            file["spectra_avg"] = spectra_avg
            file["N_range"] = N_range
            file["sigma"] = 0.0
            file["cutoff"] = current_cutoff
        end
    catch e
        println("Warning: Failed to save at N=$N. Error: $e")
    end

    GC.gc()
end

println("Completed successfully for Clean State with cutoff=$current_cutoff")