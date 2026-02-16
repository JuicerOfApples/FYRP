using Random, Statistics
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads

BLAS.set_num_threads(1)

Random.seed!(1234);

function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    # Start with a Néel state (Up, Dn, Up, Dn...)
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
Internal function: Runs DMRG at EPS precision and extracts ALL Schmidt coefficients.
"""
function get_schmidt_coeffs_eps(N, σ, J, Δ, μ, num_sweeps)
    sites = siteinds("S=1/2", N; conserve_qns=true)
    
    adj_mat = create_weighted_adj_mat(N, σ; μ=μ)
    H = create_weighted_xxz_mpo(N, adj_mat, sites; J=J, Δ=Δ)
    
    initial_state = [isodd(j) ? "Up" : "Dn" for j in 1:N]
    ψ₀ = randomMPS(sites, initial_state)

    sweeps = Sweeps(num_sweeps)
    setmaxdim!(sweeps, 5000) 
    setcutoff!(sweeps, 1E-16) 
    setnoise!(sweeps, 1E-6, 1E-7, 1E-8, 0.0) # Helps escape local minima

    _, ψ_gs = dmrg(H, ψ₀, sweeps; outputlevel=0)
    
    center_bond = N ÷ 2
    orthogonalize!(ψ_gs, center_bond)
    
    # Perform SVD at the center bond
    U, S, V = svd(ψ_gs[center_bond], (linkind(ψ_gs, center_bond - 1), siteind(ψ_gs, center_bond)))
    
    # Extract diagonal singular values (Schmidt coefficients)
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
    μ,
    J_val,
    Δ_val
)
    filename = joinpath(@__DIR__, "EPS_Schmidt_data_$(σ_val).jld2")
    println("Data will be saved to: $filename")

    for N in N_values
        
        if haskey(entanglement_spectrum_results, N)
            println("Skipping N = $N, results already loaded/computed.")
            flush(stdout)
            continue
        end

        println("Running for N = $N...")
        flush(stdout)
        
        # Store all raw vectors from the threads
        raw_results = Vector{Vector{Float64}}(undef, num_graphs_avg)
        
        # --- CRITICAL FIX 2: Dynamic Batching with Semaphore ---
        # If N is large, we reduce the number of parallel tasks to save memory.
        
        # Determine safe concurrency limit based on N
        max_concurrent_tasks = if N < 20
            Threads.nthreads()
        elseif N < 30
            div(Threads.nthreads(), 2) # Use half the cores
        else
            max(1, div(Threads.nthreads(), 4)) # Use 1/4 cores (e.g., 2 tasks on 8 cores)
        end
        max_concurrent_tasks = max(1, max_concurrent_tasks)

        println("  > Using concurrency limit of $max_concurrent_tasks to prevent OOM...")

        # Use a Semaphore to enforce the concurrency limit
        task_sem = Base.Semaphore(max_concurrent_tasks)

        Threads.@threads for i in 1:num_graphs_avg
            Base.acquire(task_sem) # Wait until a slot is free
            try
                raw_results[i] = get_schmidt_coeffs_eps(N, σ_val, J_val, Δ_val, μ, num_sweeps)
            finally
                Base.release(task_sem) # Free the slot
                # Suggest GC to free memory after heavy computation
                GC.gc()
            end
        end

        # --- Dynamic Averaging ---
        # 1. Find the maximum bond dimension occurring across all realizations
        max_len = maximum(length.(raw_results))
        
        # 2. Pad shorter vectors with zeros so we can average index-by-index
        padded_matrix = zeros(Float64, num_graphs_avg, max_len)
        for i in 1:num_graphs_avg
            len = length(raw_results[i])
            padded_matrix[i, 1:len] = raw_results[i]
        end
        
        # 3. Compute mean across realizations
        avg_coeffs = [mean(padded_matrix[:, j]) for j in 1:max_len]

        lock(data_lock) do
            entanglement_spectrum_results[N] = avg_coeffs
        end

        avg_chi = mean(length.(raw_results))
        println("Completed N = $N (Max Bond Dim: $max_len, Avg Chi: $avg_chi)")
        flush(stdout)
        
        # Save checkpoint
        try
            jldsave(filename;
                entanglement_spectrum_results, N_values, σ_val, num_graphs_avg, 
                J_val, Δ_val, μ_val=μ)
        catch e
            println("WARNING: Could not save checkpoint for N = $N. Error: $e")
        end
        
        # Full GC sweep between N values
        GC.gc()
    end
end


N_values = collect(10:2:90)

σ_val = 0.00002      
J_val = -1.0        
Δ_val = -1.0           
μ_val = 1.0          

num_sweeps = 40       
num_graphs_avg = 10    

filename = joinpath(@__DIR__, "EPS_Schmidt_data_$(σ_val).jld2")
data_lock = SpinLock() 

# Initialize the results dictionary
entanglement_spectrum_results = Dict{Int, Vector{Float64}}() 

# Check for existing file to resume
if isfile(filename)
    println("Found existing data file. Loading progress...")
    try
        loaded_data = jldopen(filename, "r")
        # Simple check: if N_values match what we want, we resume
        saved_N = read(loaded_data, "N_values")
        saved_sigma = read(loaded_data, "σ_val")
        
        if saved_N == N_values && saved_sigma == σ_val
            println("Parameters match. Resuming...")
            global entanglement_spectrum_results = read(loaded_data, "entanglement_spectrum_results")
        else
            println("WARNING: Parameters in file differ. Starting fresh.")
            global entanglement_spectrum_results = Dict{Int, Vector{Float64}}()
        end
        close(loaded_data)
    catch e
        println("WARNING: Could not load existing file. Error: $e")
        global entanglement_spectrum_results = Dict{Int, Vector{Float64}}()
    end
else
    println("No existing data file found. Starting fresh.")
    global entanglement_spectrum_results = Dict{Int, Vector{Float64}}()
end
flush(stdout)

# Run the simulation
run_simulation_ent_spec(
    entanglement_spectrum_results,
    data_lock,
    N_values,
    σ_val,
    num_graphs_avg,
    num_sweeps,
    μ_val,
    J_val,
    Δ_val
)

println("Simulation complete. Data saved to $filename")