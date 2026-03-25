using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Printf

# Disable BLAS multithreading to avoid clashing with Julia threads
LinearAlgebra.BLAS.set_num_threads(1)
Random.seed!(1234)

# ==============================================================================
# 1. Parse Command Line Arguments from SLURM
# ==============================================================================
if length(ARGS) < 1
    error("Usage: julia get_data_clean_fids.jl <cutoff>")
end

current_cutoff = parse(Float64, ARGS[1])

println("Running Node | Cutoff = $current_cutoff")

# Parameters
N_range = collect(2:2:100)
num_sweeps = 30
max_bond_dim_limit = 100
μ = 1.0

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
    # For the clean state, σ is always 0.0
    A = ones(Float64, N, N)
    A -= Matrix{Float64}(I, N, N)
    # Multiply by μ to ensure the baseline coupling is correctly scaled
    return A .* μ
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

function get_spectrum(ψ::MPS, N::Int, max_dim::Int)
    center_bond = N ÷ 2
    orthogonalize!(ψ, center_bond)
    
    r_ind = linkind(ψ, center_bond)
    s_ind = siteind(ψ, center_bond)
    inds_to_svd = isnothing(r_ind) ? (s_ind,) : (s_ind, r_ind)
    
    _, S, _ = svd(ψ[center_bond], inds_to_svd...)
    
    schmidt_dim = dim(S, 1)
    sv = [S[i, i] for i in 1:schmidt_dim]
    
    p = sort(sv .^ 2, rev=true)
    p ./= sum(p)
 
    padded_p = zeros(Float64, max_dim)
    len = min(length(p), max_dim)
    padded_p[1:len] = p[1:len]
    
    return padded_p
end

# ==============================================================================
# 3. Main Simulation Loop
# ==============================================================================
flush(stdout)

# --- STRICT CLEAN STATE SETTINGS (Reference) ---
const clean_cutoff = 1e-16
sweeps_exact = Sweeps(num_sweeps)
setmaxdim!(sweeps_exact, max_bond_dim_limit)
setcutoff!(sweeps_exact, clean_cutoff)

# --- TRUNCATED STATE SETTINGS ---
sweeps_trunc = Sweeps(num_sweeps)
setmaxdim!(sweeps_trunc, max_bond_dim_limit)
setcutoff!(sweeps_trunc, current_cutoff)

cutoff_str = @sprintf("%.1e", current_cutoff)
filename = joinpath(@__DIR__, "data_clean_fid_$(cutoff_str).jld2")

# Arrays for metrics (no error arrays needed as there is no averaging)
bond_dims = zeros(Float64, length(N_range))
fidelities = zeros(Float64, length(N_range))
spectra = zeros(Float64, length(N_range), max_bond_dim_limit)

for i in 1:length(N_range)
    N = N_range[i]
    
    ψ₀, sites = create_MPS(N)
    adj_mat = create_weighted_adj_mat(N, 0.0; μ=μ)
    
    # 1. Calculate Exact Reference State
    H_exact = create_weighted_xxz_mpo(N, adj_mat, sites; J=-1.0, Δ=-1.0, cutoff=clean_cutoff)
    _, ψ_exact = dmrg(H_exact, ψ₀, sweeps_exact; outputlevel=0)
    
    # 2. Calculate Truncated State
    # We must reset the initial state to ensure a fair DMRG starting point
    ψ₀_trunc = MPS(sites, [isodd(j) ? "Up" : "Dn" for j in 1:N])
    H_trunc = create_weighted_xxz_mpo(N, adj_mat, sites; J=-1.0, Δ=-1.0, cutoff=current_cutoff)
    _, ψ_trunc = dmrg(H_trunc, ψ₀_trunc, sweeps_trunc; outputlevel=0)
    
    # 3. Calculate Fidelity and Extract Metrics
    overlap = inner(ψ_exact', ψ_trunc) 
    fid = abs(overlap)^2
    
    fidelities[i] = fid
    bond_dims[i] = maxlinkdim(ψ_trunc)
    spectra[i, :] .= get_spectrum(ψ_trunc, N, max_bond_dim_limit)

    println("Done: N=$N, Cutoff=$current_cutoff | Fid: $(fidelities[i]) | BD: $(bond_dims[i])")
    flush(stdout)

    try
        jldopen(filename, "w"; compress=true) do file
            file["bond_dims"] = bond_dims
            file["fidelities"] = fidelities
            file["spectra"] = spectra
            file["N_range"] = N_range
            file["cutoff"] = current_cutoff
        end
    catch e
        println("Warning: Failed to save at N=$N. Error: $e")
    end

    # Force garbage collection to keep RAM footprint low
    GC.gc()
end

println("Completed successfully for cutoff=$current_cutoff")