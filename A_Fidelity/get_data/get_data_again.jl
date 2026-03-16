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
if length(ARGS) < 2
    error("Usage: julia get_data_again.jl <cutoff> <sigma_task_id>")
end

current_cutoff = parse(Float64, ARGS[1])
task_id = parse(Int, ARGS[2])

# Parameters
N_range = collect(5:5:200)
sigma_values = Float64[5e-7, 6e-7, 7e-7, 8e-7, 9e-7, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 1e-5, 1e-4, 1e-3, 1e-2, 2e-2, 1e-1, 2e-1, 5e-1, 7e-1, 1.0, 1.5, 2.0, 3.0, 5.0]

if 1 <= task_id <= length(sigma_values)
    σ = sigma_values[task_id]
    println("Running Node Task ID $task_id | Cutoff = $current_cutoff | σ = $σ")
else
    error("Task ID $task_id out of bounds. Must be between 1 and $(length(sigma_values)).")
end

num_graphs_avg = 10
num_sweeps = 30
max_bond_dim_limit = 2000
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

# --- STRICT CLEAN STATE SETTINGS ---
const clean_cutoff = 1e-16
sweeps_clean = Sweeps(num_sweeps)
setmaxdim!(sweeps_clean, max_bond_dim_limit)
setcutoff!(sweeps_clean, clean_cutoff)

# --- NOISY STATE SETTINGS (From SLURM) ---
sweeps_noisy = Sweeps(num_sweeps)
setmaxdim!(sweeps_noisy, max_bond_dim_limit)
setcutoff!(sweeps_noisy, current_cutoff)

sigma_str = @sprintf("%.1e", σ)
cutoff_str = @sprintf("%.1e", current_cutoff)
filename = joinpath(@__DIR__, "data_fid_$(cutoff_str)_$(sigma_str).jld2")

# Arrays for metrics
avg_arr = zeros(Float64, length(N_range)); err_arr = zeros(Float64, length(N_range))
vn_avg = zeros(Float64, length(N_range)); vn_err = zeros(Float64, length(N_range))
s05_avg = zeros(Float64, length(N_range)); s05_err = zeros(Float64, length(N_range))
s0_avg = zeros(Float64, length(N_range)); s0_err = zeros(Float64, length(N_range))
energy_avg = zeros(Float64, length(N_range)); energy_err = zeros(Float64, length(N_range))

# Arrays for Fidelity and Infidelity
fid_avg = zeros(Float64, length(N_range)); fid_err = zeros(Float64, length(N_range))
infid_avg = zeros(Float64, length(N_range)); infid_err = zeros(Float64, length(N_range))

spectra_avg = zeros(Float64, length(N_range), max_bond_dim_limit)

for i in 1:length(N_range)
    N = N_range[i]
    
    ψ₀_clean, sites = create_MPS(N)
    adj_mat_clean = create_weighted_adj_mat(N, 0.0; μ=μ)
    
    # Notice we pass `clean_cutoff` here
    H_clean = create_weighted_xxz_mpo(N, adj_mat_clean, sites; J=-1.0, Δ=-1.0, cutoff=clean_cutoff)
    
    # Notice we pass `sweeps_clean` here
    _, ψ_clean = dmrg(H_clean, ψ₀_clean, sweeps_clean; outputlevel=0)
    
    bond_dims = zeros(Float64, num_graphs_avg)
    vns = zeros(Float64, num_graphs_avg)
    s05s = zeros(Float64, num_graphs_avg)
    s0s = zeros(Float64, num_graphs_avg)
    energies = zeros(Float64, num_graphs_avg)
    fids = zeros(Float64, num_graphs_avg)
    infids = zeros(Float64, num_graphs_avg)
    spec_sum = zeros(Float64, max_bond_dim_limit)

    for k in 1:num_graphs_avg
        # Use exact same 'sites' to ensure states are compatible
        ψ₀_noisy = MPS(sites, [isodd(j) ? "Up" : "Dn" for j in 1:N])
        adj_mat_noisy = create_weighted_adj_mat(N, σ; μ=μ)
        
        # Uses the SLURM cutoff
        H_noisy = create_weighted_xxz_mpo(N, adj_mat_noisy, sites; J=-1.0, Δ=-1.0, cutoff=current_cutoff)
        
        # Uses the SLURM sweeps
        energy_noisy, ψ_noisy = dmrg(H_noisy, ψ₀_noisy, sweeps_noisy; outputlevel=0)
        
        overlap = inner(ψ_clean', ψ_noisy) 
        fid = abs(overlap)^2
        infid = 1.0 - fid
        
        fids[k] = fid
        infids[k] = infid
        
        # Record standard metrics
        energies[k] = energy_noisy
        bond_dims[k] = maxlinkdim(ψ_noisy)
        vn, s05, s0, spec = get_entropies_and_spectrum(ψ_noisy, N, max_bond_dim_limit)
        vns[k] = vn
        s05s[k] = s05
        s0s[k] = s0
        spec_sum .+= spec
    end

    avg_arr[i] = mean(bond_dims); err_arr[i] = std(bond_dims)
    vn_avg[i] = mean(vns); vn_err[i] = std(vns)
    s05_avg[i] = mean(s05s); s05_err[i] = std(s05s)
    s0_avg[i] = mean(s0s); s0_err[i] = std(s0s)
    energy_avg[i] = mean(energies); energy_err[i] = std(energies)
    fid_avg[i] = mean(fids); fid_err[i] = std(fids)
    infid_avg[i] = mean(infids); infid_err[i] = std(infids)
    spectra_avg[i, :] .= spec_sum ./ num_graphs_avg

    println("Done: N=$N, σ=$σ | Avg Fid: $(fid_avg[i]) | Avg BD: $(avg_arr[i])")
    flush(stdout)

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
            file["sigma"] = σ
            file["cutoff"] = current_cutoff
        end
    catch e
        println("Warning: Failed to save at N=$N. Error: $e")
    end

    # Force garbage collection to keep RAM footprint low
    GC.gc()
end

println("Completed successfully for cutoff=$current_cutoff, σ=$σ")