using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads

Random.seed!(1234);

# Lock for thread-safe file writing and counter updates
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
    p ./= sum(p) # Ensure normalization
    
    # Calculate Entropies
    vn = -sum(x -> x > 1e-18 ? x * log(x) : 0.0, p)
    s_05 = (1 / (1 - 0.5)) * log(sum(sqrt.(p)))
    s_0 = log(count(x -> x > 1e-16, p))
    
    # Pad spectrum with zeros up to max_dim for consistent averaging
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
    N_range,
    sigma_values,
    num_graphs_avg::Int,
    num_sweeps::Int,
    max_bond_dim_limit::Int,
    cutoff::Float64,
    μ::Float64,
    filename::String
)
    println("Starting simulation on $(Threads.nthreads()) threads...")
    flush(stdout)

    # Parallelise over N
    Threads.@threads for i in 1:length(N_range)
        N = N_range[i]
        
        for (j, σ) in enumerate(sigma_values)
            # Skip if already computed
            if avg_matrix[i, j] != 0.0
                continue
            end

            # Arrays for averaging over graphs
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
                
                # Get entropies and padded spectrum
                vn, s05, s0, spec = get_entropies_and_spectrum(ψ_gs, N, max_bond_dim_limit)
                vns[k] = vn
                s05s[k] = s05
                s0s[k] = s0
                spec_sum .+= spec
            end

            # Calculate means and stds
            avg_matrix[i, j] = mean(bond_dims)
            err_matrix[i, j] = std(bond_dims)
            
            vn_avg[i, j] = mean(vns)
            vn_err[i, j] = std(vns)
            
            s05_avg[i, j] = mean(s05s)
            s05_err[i, j] = std(s05s)
            
            s0_avg[i, j] = mean(s0s)
            s0_err[i, j] = std(s0s)
            
            # Average the Schmidt spectrum
            spectra_avg[i, j, :] .= spec_sum ./ num_graphs_avg
            
            # Print and flush immediately
            println("Done: N=$N, σ=$σ | Avg BD: $(avg_matrix[i, j])")
            flush(stdout)
        end

        # Thread-safe save
        lock(io_lock) do
            try
                jldsave(filename; 
                    avg_matrix, err_matrix, 
                    vn_avg, vn_err, 
                    s05_avg, s05_err, 
                    s0_avg, s0_err, 
                    spectra_avg,
                    N_range, sigma_values)
                println("Checkpoint saved for N=$N")
                flush(stdout)
            catch e
                @warn "Checkpoint failed for N=$N: $e"
                flush(stdout)
            end
        end
    end
end

# --- Setup ---
N_range = [2:2:30; 35:5:50; 60; 70; 80; 90; 100]
sigma_values = [0.0, 0.00001, 0.002]
num_graphs_avg = 10
num_sweeps = 30
max_bond_dim_limit = 1000
cutoff = 1E-16
μ = 1.0
filename = joinpath(@__DIR__, "EPS_BD_data.jld2")

# Initialize Data Structures
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
    println("Resuming from existing file: $filename")
    flush(stdout)
    data = load(filename)
    # Check if the new entropy variables exist, otherwise we must start fresh
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
        println("Parameter mismatch or old file version detected. Starting fresh.")
        flush(stdout)
    end
end

run_simulation_avg_err(
    avg_matrix, err_matrix,
    vn_avg, vn_err,
    s05_avg, s05_err,
    s0_avg, s0_err,
    spectra_avg,
    N_range, sigma_values,
    num_graphs_avg, num_sweeps, max_bond_dim_limit, cutoff, μ, filename
)

println("Simulation complete. Final data saved to $filename")
flush(stdout)