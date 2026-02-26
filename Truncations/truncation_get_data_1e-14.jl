using Random, Statistics
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads

Random.seed!(1234);

"""
Creates the initial random MPS (Néel state).
"""
function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = randomMPS(sites, initial_state)
    return ψ₀, sites
end

"""
Creates a weighted adjacency matrix.
"""
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

"""
Creates the MPO for the XXZ Hamiltonian.
"""
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

"""
Internal function: Runs DMRG and extracts:
1. Schmidt Coefficients
2. Ground State Energy
3. Local Magnetization (Sz)
"""
function get_data(N, σ, J, Δ, μ, num_sweeps, cutoff_val)
    sites = siteinds("S=1/2", N; conserve_qns=true)
    
    adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
    H = create_weighted_xxz_mpo(N, adj_mat, sites; J=J, Δ=Δ)
    
    initial_state = [isodd(j) ? "Up" : "Dn" for j in 1:N]
    ψ₀ = randomMPS(sites, initial_state)

    sweeps = Sweeps(num_sweeps)
    setmaxdim!(sweeps, 5000) 
    setcutoff!(sweeps, cutoff_val) 
    setnoise!(sweeps, 1E-6, 1E-7, 1E-8, 0.0)

    energy, ψ_gs = dmrg(H, ψ₀, sweeps; outputlevel=0)
    
    
    # Measure local magnetization <Sz> at every site
    # Returns a Vector{Float64} of length N
    magnetization = expect(ψ_gs, "Sz") 

    center_bond = N ÷ 2
    orthogonalize!(ψ_gs, center_bond)
    
    U, S, V = svd(ψ_gs[center_bond], (linkind(ψ_gs, center_bond - 1), siteind(ψ_gs, center_bond)))
    coeffs = [S[i, i] for i in 1:dim(S, 1)]
    sort!(coeffs, rev=true)
    
    return coeffs, energy, magnetization
end

function run_simulation(
    entanglement_results::Dict,
    energy_results::Dict,
    sz_results::Dict,
    data_lock::SpinLock,
    N_values,
    sigma_values,
    num_graphs_avg,
    num_sweeps,
    μ,
    J_val,
    Δ_val,
    cutoff_val 
)
    filename = joinpath(@__DIR__, "truncation_$cutoff_val.jld2")

    for σ_val in sigma_values
        println("\n=== Processing σ = $σ_val ===")

        # Ensure dictionaries exist for this sigma
        lock(data_lock) do
            if !haskey(entanglement_results, σ_val)
                entanglement_results[σ_val] = Dict{Int, Vector{Float64}}()
                energy_results[σ_val]       = Dict{Int, Float64}()
                sz_results[σ_val]           = Dict{Int, Vector{Float64}}()
            end
        end

        for N in N_values
            
            already_done = false
            lock(data_lock) do
                if haskey(entanglement_results[σ_val], N)
                    already_done = true
                end
            end

            if already_done
                println("Skipping σ=$σ_val, N=$N (already computed).")
                continue
            end

            println("Running for σ = $σ_val, N = $N...")
            
            # Temporary storage for threads
            raw_coeffs = Vector{Vector{Float64}}(undef, num_graphs_avg)
            raw_energies = Vector{Float64}(undef, num_graphs_avg)
            raw_sz = Vector{Vector{Float64}}(undef, num_graphs_avg)
            
            Threads.@threads for i in 1:num_graphs_avg
                c, e, sz = get_data(N, σ_val, J_val, Δ_val, μ, num_sweeps, cutoff_val)
                raw_coeffs[i] = c
                raw_energies[i] = e
                raw_sz[i] = sz
            end

            max_len = maximum(length.(raw_coeffs))
            padded_matrix = zeros(Float64, num_graphs_avg, max_len)
            for i in 1:num_graphs_avg
                len = length(raw_coeffs[i])
                padded_matrix[i, 1:len] = raw_coeffs[i]
            end
            avg_coeffs = [mean(padded_matrix[:, j]) for j in 1:max_len]

            avg_energy = mean(raw_energies)

            # raw_sz is a Vector of Vectors (each length N). We average site-wise.
            # Convert to matrix (num_graphs x N) then take mean column-wise
            sz_matrix = hcat(raw_sz...)' # Transpose to get rows=graphs, cols=sites
            avg_sz = vec(mean(sz_matrix, dims=1))

            lock(data_lock) do
                entanglement_results[σ_val][N] = avg_coeffs
                energy_results[σ_val][N] = avg_energy
                sz_results[σ_val][N] = avg_sz
            end

            println("Completed σ = $σ_val, N = $N")
            flush(stdout)

            # Save checkpoint
            try
                jldsave(filename; 
                    entanglement_results, 
                    energy_results,
                    sz_results,
                    N_values, 
                    sigma_values, 
                    num_graphs_avg, 
                    J_val, 
                    Δ_val, 
                    μ_val=μ,
                    cutoff_val
                )
            catch e
                println("WARNING: Could not save checkpoint for σ=$σ_val, N=$N. Error: $e")
            end
        end
    end
end


N_values = collect(10:2:90)
sigma_values = [0.0, 0.0002, 0.002, 0.02, 0.2]
J_val = -1.0        
Δ_val = -1.0           
μ_val = 1.0          
num_sweeps = 40       
num_graphs_avg = 10    
truncation_cutoff = 1e-14

filename = joinpath(@__DIR__, "truncation_$truncation_cutoff.jld2")
data_lock = SpinLock() 

# Initialize Data Structures
entanglement_results = Dict{Float64, Dict{Int, Vector{Float64}}}() 
energy_results = Dict{Float64, Dict{Int, Float64}}()
sz_results = Dict{Float64, Dict{Int, Vector{Float64}}}()

if isfile(filename)
    println("Found existing data file. Loading progress...")
    try
        loaded_data = jldopen(filename, "r")
        saved_N = read(loaded_data, "N_values")
        saved_sigmas = read(loaded_data, "sigma_values")
        saved_cutoff = read(loaded_data, "cutoff_val")
        
        if saved_N == N_values && saved_sigmas == sigma_values && saved_cutoff == truncation_cutoff
            println("Parameters match. Resuming...")
            global entanglement_results = read(loaded_data, "entanglement_results")
            # Check if older files might verify energy/sz exists, if not initialize empty
            if haskey(loaded_data, "energy_results")
                global energy_results = read(loaded_data, "energy_results")
                global sz_results = read(loaded_data, "sz_results")
            else
                 println("Old file format detected (no observables). Starting fresh to ensure consistency.")
                 global entanglement_results = Dict{Float64, Dict{Int, Vector{Float64}}}() 
                 global energy_results = Dict{Float64, Dict{Int, Float64}}()
                 global sz_results = Dict{Float64, Dict{Int, Vector{Float64}}}()
            end
        else
            println("Parameters differ. Starting fresh.")
        end
        close(loaded_data)
    catch e
        println("Error loading file: $e. Starting fresh.")
    end
end

run_simulation(
    entanglement_results,
    energy_results,
    sz_results,
    data_lock,
    N_values,
    sigma_values,
    num_graphs_avg,
    num_sweeps,
    μ_val,
    J_val,
    Δ_val,
    truncation_cutoff
)

println("Simulation complete. Data saved to $filename")