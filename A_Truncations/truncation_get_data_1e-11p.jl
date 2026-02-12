using Random, Statistics
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads


Random.seed!(1234);
BLAS.set_num_threads(1)
ITensors.Strided.disable_threads()

if length(ARGS) < 1
    error("Please provide sigma index (1-5)")
end
idx = parse(Int, ARGS[1])
sigma_values_all = [0.0, 0.0002, 0.002, 0.02, 0.2]
target_sigma = sigma_values_all[idx]


# --- CORE FUNCTIONS ---

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
        for j in i+1:N
            coupling_strength = adj_mat[i, j]
            if coupling_strength != 0.0
                ampo += coupling_strength * (J / 2), "S+", i, "S-", j
                ampo += coupling_strength * (J / 2), "S-", i, "S+", j
                ampo += coupling_strength * (J * Δ), "Sz", i, "Sz", j
            end
        end
    end
    return MPO(ampo, sites)
end

function get_data(N, σ, J, Δ, μ, num_sweeps, cutoff_val)
    sites = siteinds("S=1/2", N; conserve_qns=true)
    adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
    H = create_weighted_xxz_mpo(N, adj_mat, sites; J=J, Δ=Δ)
    
    initial_state = [isodd(j) ? "Up" : "Dn" for j in 1:N]
    ψ₀ = randomMPS(sites, initial_state)

    maxdims = [10, 20, 50, 100, 200, 400, 800, 1600, 3200, 5000] 
    noises = [1E-6, 1E-7, 1E-8, 0.0]

    # Perform DMRG with keyword arguments
    energy, ψ_gs = dmrg(H, ψ₀; 
                        nsweeps=num_sweeps, 
                        maxdim=maxdims, 
                        cutoff=cutoff_val, 
                        noise=noises, 
                        outputlevel=0) 
    
    magnetization = expect(ψ_gs, "Sz")

    center_bond = N ÷ 2
    orthogonalize!(ψ_gs, center_bond)
    
    U, S, V = svd(ψ_gs[center_bond], (linkind(ψ_gs, center_bond - 1), siteind(ψ_gs, center_bond)))
    coeffs = [S[i, i] for i in 1:dim(S, 1)]
    sort!(coeffs, rev=true)
    
    return coeffs, energy, magnetization
end

# --- MAIN EXECUTION ---

N_values = collect(10:2:90)
J_val = -1.0        
Δ_val = -1.0           
μ_val = 1.0          
num_sweeps = 40       
num_graphs_avg = 8
truncation_cutoff = 1e-11

# Filename specific to the sigma and cutoff
filename = joinpath(@__DIR__, "truncation_$(truncation_cutoff)_sigma_$(target_sigma).jld2")
data_lock = SpinLock() 

# Initialize Structures
entanglement_results = Dict{Int, Vector{Float64}}() 
energy_results = Dict{Int, Float64}()
sz_results = Dict{Int, Vector{Float64}}()

# Resume logic
if isfile(filename)
    println("Resuming from existing file: $filename")
    jldopen(filename, "r") do file
        global entanglement_results = read(file, "entanglement_results")
        global energy_results = read(file, "energy_results")
        global sz_results = read(file, "sz_results")
    end
end

println("\n=== Starting Simulation: σ = $target_sigma, Cutoff = $truncation_cutoff ===")

for N in N_values
    if haskey(entanglement_results, N)
        println("Skipping N=$N (already computed).")
        continue
    end

    println("Running N = $N...")
    raw_coeffs = Vector{Vector{Float64}}(undef, num_graphs_avg)
    raw_energies = Vector{Float64}(undef, num_graphs_avg)
    raw_sz = Vector{Vector{Float64}}(undef, num_graphs_avg)
    
    # Parallelize over the 10 graphs
    Threads.@threads for i in 1:num_graphs_avg
        c, e, sz = get_data(N, target_sigma, J_val, Δ_val, μ_val, num_sweeps, truncation_cutoff)
        raw_coeffs[i] = c
        raw_energies[i] = e
        raw_sz[i] = sz
    end

    # Average Coeffs (Padding for different Schmidt ranks)
    max_len = maximum(length.(raw_coeffs))
    padded_matrix = zeros(Float64, num_graphs_avg, max_len)
    for i in 1:num_graphs_avg
        len = length(raw_coeffs[i])
        padded_matrix[i, 1:len] = raw_coeffs[i]
    end
    avg_coeffs = [mean(padded_matrix[:, j]) for j in 1:max_len]

    # Average Energy and Sz
    avg_energy = mean(raw_energies)
    sz_matrix = hcat(raw_sz...)' 
    avg_sz = vec(mean(sz_matrix, dims=1))

    # Save to memory
    entanglement_results[N] = avg_coeffs
    energy_results[N] = avg_energy
    sz_results[N] = avg_sz

    # Save Checkpoint
    jldsave(filename; 
        entanglement_results, energy_results, sz_results,
        N_values, target_sigma, truncation_cutoff, J_val, Δ_val
    )
    println("Completed N = $N and saved to file.")
    flush(stdout)
end

println("Full simulation for σ=$target_sigma complete.")