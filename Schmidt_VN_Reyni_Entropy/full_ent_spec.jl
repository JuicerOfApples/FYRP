using Random, Statistics
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads

Random.seed!(1234);


function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
    ψ₀ = randomMPS(sites, initial_state)
    return ψ₀, sites
end

"""
Creates a weighted adjacency matrix for a completely connected graph.
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
Creates the MPO for the XXZ Hamiltonian on a graph with weighted interactions.
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
Internal function: Runs DMRG and extracts the Schmidt coefficients from the central bond.
"""
function get_schmidt_coeffs(N, σ, J, Δ, μ, num_sweeps)
    sites = siteinds("S=1/2", N; conserve_qns=true)
    
    adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
    H = create_weighted_xxz_mpo(N, adj_mat, sites; J=J, Δ=Δ)
    
    initial_state = [isodd(j) ? "Up" : "Dn" for j in 1:N]
    ψ₀ = randomMPS(sites, initial_state)

    sweeps = Sweeps(num_sweeps)
    setmaxdim!(sweeps, 250)
    setcutoff!(sweeps, 1E-10)

    _, ψ_gs = dmrg(H, ψ₀, sweeps; outputlevel=0)
    
    center_bond = N ÷ 2
    orthogonalize!(ψ_gs, center_bond)
    
    U, S, V = svd(ψ_gs[center_bond], (linkind(ψ_gs, center_bond - 1), siteind(ψ_gs, center_bond)))
    
    coeffs = [S[i, i] for i in 1:dim(S, 1)]
    sort!(coeffs, rev=true)
    
    return coeffs
end





function run_simulation_ent_spec(
    entanglement_spectrum_results::Dict,
    data_lock::SpinLock,
    N_values,
    σ_val,
    num_graphs_avg,
    num_sweeps,
    max_coeffs_to_store,
    μ,
    J_val,
    Δ_val
)
    filename = joinpath(@__DIR__, "full_ent_spec_data_0.002.jld2")
    println("Data will be saved to: $filename")

    for N in N_values
        
        if haskey(entanglement_spectrum_results, N)
            println("Skipping N = $N, results already loaded/computed.")
            continue
        end

        println("Running for N = $N")
        
        N_all_coeffs = zeros(Float64, num_graphs_avg, max_coeffs_to_store)
        
        Threads.@threads for i in 1:num_graphs_avg
            run_coeffs = get_schmidt_coeffs(N, σ_val, J_val, Δ_val, μ, num_sweeps)
            
            num_found = length(run_coeffs)
            
            N_all_coeffs[i, 1:min(num_found, max_coeffs_to_store)] = run_coeffs[1:min(num_found, max_coeffs_to_store)]
        end

        avg_coeffs = [mean(N_all_coeffs[:, j]) for j in 1:max_coeffs_to_store]

        lock(data_lock) do
            entanglement_spectrum_results[N] = avg_coeffs
        end

        println("Completed N = $N")
        try
            jldsave(filename; entanglement_spectrum_results, N_values, σ_val, num_graphs_avg, max_coeffs_to_store)
            println("Checkpoint saved for N = $N")
        catch e
            println("WARNING: Could not save checkpoint for N = $N. Error: $e")
        end
    end
end



N_values = [20, 30, 40, 50, 60, 70, 80, 90]
σ_val = 0.0      
J_val = -1.0        
Δ_val = -1.0          
μ_val = 1.0          
num_sweeps = 30     
num_graphs_avg = 10  
max_coeffs_to_store = 250 

filename = joinpath(@__DIR__, "full_ent_spec_data_0.002.jld2")
data_lock = SpinLock() 


entanglement_spectrum_results = Dict{Int, Vector{Float64}}() 

if isfile(filename)
    println("Found existing data file. Loading progress...")
    try
        loaded_data = jldopen(filename, "r")
        
        # Check if all key parameters match
        params_match = (
            read(loaded_data, "N_values") == N_values &&
            read(loaded_data, "σ_val") == σ_val &&
            read(loaded_data, "num_graphs_avg") == num_graphs_avg &&
            read(loaded_data, "max_coeffs_to_store") == max_coeffs_to_store
        )

        if params_match
            println("Parameters match. Resuming...")
            global entanglement_spectrum_results = read(loaded_data, "entanglement_spectrum_results")
        else
            println("WARNING: Parameters in file do not match current script")
            global entanglement_spectrum_results = Dict{Int, Vector{Float64}}()
        end
        close(loaded_data)
    catch e
        println("WARNING: Could not load existing file. Error: $e")
        global entanglement_spectrum_results = Dict{Int, Vector{Float64}}()
    end
else
    println("No existing data file found.")
    global entanglement_spectrum_results = Dict{Int, Vector{Float64}}()
end


run_simulation_ent_spec(
    entanglement_spectrum_results,
    data_lock,
    N_values,
    σ_val,
    num_graphs_avg,
    num_sweeps,
    max_coeffs_to_store,
    μ_val,
    J_val,
    Δ_val
)

jldsave(filename;
    entanglement_spectrum_results, 
    N_values, 
    σ_val, 
    J_val, 
    Δ_val, 
    μ_val, 
    num_graphs_avg, 
    max_coeffs_to_store
)
println("Data saved successfully.\n")