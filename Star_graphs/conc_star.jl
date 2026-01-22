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

function create_weighted_star_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    A = zeros(Float64, N, N)
    if N < 2
        return A
    end
    for j in 2:N
        weight = (σ == 0.0) ? μ : (μ + σ * randn())
        A[1, j] = A[j, 1] = weight
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
    return MPO(ampo, sites)
end

function calculate_concurrence(rho_matrix::Matrix{ComplexF64})
    if size(rho_matrix) != (4, 4)
        error("Density matrix must be 4x4")
    end
    sy = [0.0 -1.0im; 1.0im 0.0]
    sy_sy = kron(sy, sy)
    rho_tilde = sy_sy * conj(rho_matrix) * sy_sy
    R_matrix = rho_matrix * rho_tilde
    eigvals_R = eigvals(R_matrix)
    lambdas = sort(sqrt.(complex(eigvals_R)), by=real, rev=true)
    C = max(0.0, real(lambdas[1]) - real(lambdas[2]) - real(lambdas[3]) - real(lambdas[4]))
    return C
end


function run_simulation_star_concurrence(
    results::Dict,
    N_range,
    sigma_values,
    num_graphs_avg::Int,
    num_sweeps::Int,
    max_bond_dim_limit::Int,
    cutoff::Float64,
    μ::Float64,
    filename::String
)
    
    ITensors.set_warn_order(18)


    Threads.@threads for i in 1:length(N_range)
        N = N_range[i] 

        # Check if N is already a key in the results dict
        if haskey(results[sigma_values[1]], N)
             println("Skipping N = $N, results already loaded/computed.")
            continue
        end
        
        for σ in sigma_values

            concurrence_for_avg_all_pairs = [zeros(Float64, num_graphs_avg) for _ in 2:N]
            
            for k in 1:num_graphs_avg
                ψ₀, sites = create_MPS(N)
                
                adj_mat = create_weighted_star_adj_mat(N, σ; μ=μ)
                H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=-1.0, Δ=-1.0)

                sweeps = Sweeps(num_sweeps)
                setmaxdim!(sweeps, max_bond_dim_limit)
                setcutoff!(sweeps, cutoff)

                _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)
    
                if N < 2
                    continue
                end
                
                orthogonalize!(ψ_gs, 1)
                s1 = sites[1]
                C_tensor = ψ_gs[1] 
                
                for j_pair in 2:N
                    C_tensor *= ψ_gs[j_pair]
                    s_j = sites[j_pair]

                    rho_1j_tensor = C_tensor * prime(dag(C_tensor), dag(s1), dag(s_j))

                    C_rows = combiner(s1, s_j)
                    C_cols = combiner(prime(s1), prime(s_j))
                    
                    rho_combined = (rho_1j_tensor * C_rows) * dag(C_cols)
                    rho_matrix = matrix(rho_combined)

                    C = calculate_concurrence(complex(rho_matrix))
                 
                    concurrence_for_avg_all_pairs[j_pair - 1][k] = C
                end
            end 

            if N >= 2
                avg_conc_all_pairs = [mean(concurrence_for_avg_all_pairs[j_idx]) for j_idx in 1:(N-1)]
                std_dev_all_pairs = [std(concurrence_for_avg_all_pairs[j_idx]) for j_idx in 1:(N-1)]
                
                results[σ][N] = (avg=avg_conc_all_pairs, err=std_dev_all_pairs)
            else
                results[σ][N] = (avg=Float64[], err=Float64[])
            end
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

"""
Initializes the results dictionary.
The outer Dict keys are sigma values.
The inner Dict keys are N (system size), and the values are a NamedTuple holding the avg (a Vector{Float64}) and err (a Vector{Float64}).
"""
function init_results_structure(sigma_values)
    InnerDictType = Dict{Int, @NamedTuple{avg::Vector{Float64}, err::Vector{Float64}}}
    
    return Dict(σ => InnerDictType() for σ in sigma_values)
end





# Parameters
N_range = 4:2:14
sigma_values = [0.2] 
num_graphs_avg = 10            
num_sweeps = 10
max_bond_dim_limit = 100
cutoff = 1E-10
μ = 1.0                  

filename = joinpath(@__DIR__, "conc_star_data_0.2.jld2") 

if isfile(filename)
    println("Found existing data file")
    try
        loaded_data = jldopen(filename, "r")
        N_range_loaded = read(loaded_data, "N_range")
        sigma_values_loaded = read(loaded_data, "sigma_values")
        
        # check if the parameters match
        if N_range_loaded == N_range && sigma_values_loaded == sigma_values
            global results = read(loaded_data, "results") 
        else
            println("WARNING: Parameters in file do not match current script.")
            global results = init_results_structure(sigma_values)
        end
        close(loaded_data)
    catch e
        println("WARNING: Could not load existing file. Error: $e")
        global results = init_results_structure(sigma_values)
    end
else
    println("No existing data file found.")
    global results = init_results_structure(sigma_values)
end

run_simulation_star_concurrence(
    results,
    N_range,
    sigma_values,
    num_graphs_avg,
    num_sweeps,
    max_bond_dim_limit,
    cutoff,
    μ,
    filename
)

jldsave(filename; results, N_range, sigma_values)
println("Data saved successfully to $filename\n")