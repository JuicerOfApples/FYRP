using Statistics, Random
using ITensors, ITensorMPS, LinearAlgebra
using JLD2
using Base.Threads

Random.seed!(1234);

function create_MPS(L::Int)
    sites = siteinds("S=1/2", L; conserve_qns=true)
    initial_state = [isodd(i) ? "Up" : "Dn" for i in 1:L] 
    ψ₀ = MPS(sites, initial_state) 
    return ψ₀, sites
end

"""
Creates a weighted adjacency matrix for an 'outlier' graph.
Spin N is connected only to spin 1.
"""
function create_outlier_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    A = zeros(Float64, N, N)

    # Create the (N-1)x(N-1) fully connected subgraph
    for i in 1:(N-1), j in (i+1):(N-1)
        weight = (σ == 0.0) ? μ : (μ + σ * randn()) 
        A[i, j] = A[j, i] = weight
    end

    # Create the single connection for the outlier (spin N) to spin 1
    outlier_weight = (σ == 0.0) ? μ : (μ + σ * randn()) 
    A[1, N] = A[N, 1] = outlier_weight

    return A
end

"""
Creates the MPO for the XXZ Hamiltonian on a graph with weighted interactions.
"""
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
    return MPO(ampo, sites)
end


function run_simulation_avg_err_outlier(
    results::Dict,
    N_range,
    sigma_values,
    num_graphs_avg::Int,
    num_sweeps::Int,
    max_bond_dim_limit::Int,
    cutoff::Float64,
    μ::Float64
)
    filename = joinpath(@__DIR__, "outl_bd_data.jld2")
  
    println("Data will be saved to: $filename")

    Threads.@threads for i in 1:length(N_range)
        N = N_range[i] 

        if results[sigma_values[1]].avg[i] != 0.0
             println("Skipping N = $N, results already loaded/computed.") 
             continue
        end
        
        for σ in sigma_values
            
            bond_dims_for_avg = zeros(Float64, num_graphs_avg)
            
            # Optimization: for sigma=0.0, all "graphs" are identical.
            num_runs = (σ == 0.0) ? 1 : num_graphs_avg 
            
            for k in 1:num_runs
                ψ₀, sites = create_MPS(N) 
                
                # Create the Outlier Graph MPO
                adj_mat = create_outlier_adj_mat(N, σ; μ=μ) 
                H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=-1.0, Δ=-1.0) 

                
                sweeps = Sweeps(num_sweeps) 
                setmaxdim!(sweeps, max_bond_dim_limit)
                setcutoff!(sweeps, cutoff)

                _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)
    
                bond_dims_for_avg[k] = maxlinkdim(ψ_gs) 
            end

            avg_dim = mean(bond_dims_for_avg[1:num_runs]) 
            std_dev = (num_runs > 1) ? std(bond_dims_for_avg[1:num_runs]) : 0.0 
            
            results[σ].avg[i] = avg_dim
            results[σ].err[i] = std_dev
  
        end
        println("Completed N = $N")

        try
            jldsave(filename; results, N_range, sigma_values) 
            println("Checkpoint saved for N = $N")
        catch e
            println("WARNING: Could not save checkpoint for N = $N. Error: $e")
        end

    
    end
    
end


# Parameters 
N_range = 10:1:100
sigma_values = [0.0, 0.002] 
num_graphs_avg = 10
num_sweeps = 30
max_bond_dim_limit = 250
cutoff = 1E-10
μ = 1.0

# New filename 
filename = joinpath(@__DIR__, "outl_bd_data.jld2")

# Data loading structure 
if isfile(filename)
    println("Found existing data file") 
    try
        loaded_data = jldopen(filename, "r")
        N_range_loaded = read(loaded_data, "N_range")
        sigma_values_loaded = read(loaded_data, "sigma_values")
        
        if N_range_loaded == N_range && sigma_values_loaded == sigma_values
            global results = read(loaded_data, "results") 
        else
            println("WARNING: Parameters in file do not match current script.") 
            global results = Dict(σ => (avg=zeros(Float64, length(N_range)),
                                  err=zeros(Float64, length(N_range))) 
                                  for σ in sigma_values) 
        end
        close(loaded_data)
    catch e
        println("WARNING: Could not load existing file. Error: $e")
        global results = Dict(σ => (avg=zeros(Float64, length(N_range)), 
                                  err=zeros(Float64, length(N_range))) 
                                  for σ in sigma_values) 
    end
else
    println("No existing data file found.") 
    global results = Dict(σ => (avg=zeros(Float64, length(N_range)), 
                                err=zeros(Float64, length(N_range))) 
                                for σ in sigma_values)
end

run_simulation_avg_err_outlier(
    results,
    N_range,
    sigma_values,
    num_graphs_avg,
    num_sweeps,
    max_bond_dim_limit,
    cutoff,
    μ
) 

jldsave(filename; results, N_range, sigma_values) 
println("Data saved successfully.\n")