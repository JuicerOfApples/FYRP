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
Creates the MPO for a 1D Disordered XXZ Chain (Linear Graph).
"""
function create_disordered_chain_mpo(N::Int, sites; J::Float64, Δ::Float64, σ::Float64, μ::Float64=1.0)
    ampo = OpSum()
    
    # Create N-1 random couplings for the N-1 bonds
    couplings = [μ + σ * randn() for _ in 1:(N-1)] 

    for i in 1:(N-1)
        # Only add nearest-neighbor terms
        coupling_strength = couplings[i]
        
        ampo += coupling_strength * (J / 2), "S+", i, "S-", i+1
        ampo += coupling_strength * (J / 2), "S-", i, "S+", i+1
        ampo += coupling_strength * (J * Δ), "Sz", i, "Sz", i+1 
    end
    return MPO(ampo, sites)
end


function run_simulation_avg_err_linear(
    results::Dict,
    N_range,
    sigma_values,
    num_graphs_avg::Int,
    num_sweeps::Int,
    max_bond_dim_limit::Int,
    cutoff::Float64,
    μ::Float64
)
    filename = joinpath(@__DIR__, "lin_bd_data.jld2")
  
    println("Data will be saved to: $filename")

    Threads.@threads for i in 1:length(N_range)
        N = N_range[i] 

        if results[sigma_values[1]].avg[i] != 0.0
             println("Skipping N = $N, results already loaded/computed.") 
            continue
        end
        
        for σ in sigma_values
            
            bond_dims_for_avg = zeros(Float64, num_graphs_avg)
            

            num_runs = (σ == 0.0) ? 1 : num_graphs_avg
            
            for k in 1:num_runs
                ψ₀, sites = create_MPS(N) 
                

                H_mpo = create_disordered_chain_mpo(N, sites; J=-1.0, Δ=-1.0, σ=σ, μ=μ) 

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

filename = joinpath(@__DIR__, "lin_bd_data.jld2")

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

run_simulation_avg_err_linear(
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