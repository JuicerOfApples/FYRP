using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads
using Printf


LinearAlgebra.BLAS.set_num_threads(1)

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

# BUG FIX 1: Added `cutoff` to the function signature
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

function run_simulation_avg_err_single_sigma(
    avg_arr::Vector{Float64}, err_arr::Vector{Float64},
    vn_avg::Vector{Float64}, vn_err::Vector{Float64},
    s05_avg::Vector{Float64}, s05_err::Vector{Float64},
    s0_avg::Vector{Float64}, s0_err::Vector{Float64},
    spectra_avg::Matrix{Float64},
    N_range, σ::Float64, num_graphs_avg::Int,
    num_sweeps::Int, max_bond_dim_limit::Int,
    cutoff::Float64, μ::Float64, filename::String
)
    println("Starting simulation for σ=$σ and cutoff $(cutoff) on $(Threads.nthreads()) threads...")
    flush(stdout)

    Threads.@threads for i in 1:length(N_range)
        N = N_range[i]
        
        if avg_arr[i] != 0.0
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
            
            # BUG FIX 1: Pass the cutoff into the MPO creation
            H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=-1.0, Δ=-1.0, cutoff=cutoff)

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

        avg_arr[i] = mean(bond_dims)
        err_arr[i] = std(bond_dims)
        vn_avg[i] = mean(vns)
        vn_err[i] = std(vns)
        s05_avg[i] = mean(s05s)
        s05_err[i] = std(s05s)
        s0_avg[i] = mean(s0s)
        s0_err[i] = std(s0s)
        spectra_avg[i, :] .= spec_sum ./ num_graphs_avg
        
        println("Done: N=$N, σ=$σ, EPS=$cutoff | Avg BD: $(avg_arr[i])")
        flush(stdout)

        lock(io_lock) do
            try
                jldsave(filename; 
                    avg_arr, err_arr, vn_avg, vn_err, 
                    s05_avg, s05_err, s0_avg, s0_err, 
                    spectra_avg, N_range, sigma=σ)
                println("Checkpoint saved for N=$N, σ=$σ, EPS=$cutoff")
                flush(stdout)
            catch e
                @warn "Checkpoint failed for N=$N: $e"
                flush(stdout)
            end
        end
    end
end


N_range = collect(5:5:300)

sigma_values = collect(Float64, vcat(0.0, 1e-7, 1e-6, 1e-5, 1e-4))

num_graphs_avg = 10
num_sweeps = 30
max_bond_dim_limit = 800
μ = 1.0

cutoff_str = @sprintf("%.0e", current_cutoff)


sigmas_to_run = sigma_values
if length(ARGS) > 0
    task_id = parse(Int, ARGS[1])
    if task_id >= 1 && task_id <= length(sigma_values)
        sigmas_to_run = [sigma_values[task_id]]
    else
        error("Task ID $task_id out of bounds. Must be between 1 and $(length(sigma_values)).")
    end
end

for σ in sigmas_to_run
    sigma_str = @sprintf("%.1e", σ)
    filename = joinpath(@__DIR__, "data_$(sigma_str).jld2")

    avg_arr = zeros(Float64, length(N_range))
    err_arr = zeros(Float64, length(N_range))
    vn_avg = zeros(Float64, length(N_range))
    vn_err = zeros(Float64, length(N_range))
    s05_avg = zeros(Float64, length(N_range))
    s05_err = zeros(Float64, length(N_range))
    s0_avg = zeros(Float64, length(N_range))
    s0_err = zeros(Float64, length(N_range))
    spectra_avg = zeros(Float64, length(N_range), max_bond_dim_limit)

    if isfile(filename)
        println("\nResuming from existing file: $filename")
        flush(stdout)
        data = load(filename)
        if haskey(data, "vn_avg") && data["N_range"] == N_range && data["sigma"] == σ
            avg_arr = data["avg_arr"]
            err_arr = data["err_arr"]
            vn_avg = data["vn_avg"]
            vn_err = data["vn_err"]
            s05_avg = data["s05_avg"]
            s05_err = data["s05_err"]
            s0_avg = data["s0_avg"]
            s0_err = data["s0_err"]
            spectra_avg = data["spectra_avg"]
        else
            println("Parameter mismatch detected. Starting fresh for $filename.")
            flush(stdout)
        end
    else
        println("\nCreating new file: $filename")
    end

    run_simulation_avg_err_single_sigma(
        avg_arr, err_arr, vn_avg, vn_err,
        s05_avg, s05_err, s0_avg, s0_err,
        spectra_avg, N_range, σ,
        num_graphs_avg, num_sweeps, max_bond_dim_limit, current_cutoff, μ, filename
    )

    println("Simulation for σ=$(σ), EPS=$(current_cutoff) complete. Data saved to $filename")
    flush(stdout)
end