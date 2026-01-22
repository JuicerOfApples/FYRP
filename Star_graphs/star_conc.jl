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
Creates a weighted adjacency matrix for a star graph (center spin 1).
Applies disorder 'σ' to the couplings.
"""
function create_weighted_star_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    A = zeros(Float64, N, N)
    if N < 2
        return A
    end
    # Connect center (1) to all outer spins (2...N)
    for j in 2:N
        weight = (σ == 0.0) ? μ : (μ + σ * randn())
        A[1, j] = A[j, 1] = weight
    end
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

"""
Calculates concurrence for a 2-qubit (4x4) density matrix.
"""
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
    
    println("Data will be saved to: $filename")

    Threads.@threads for i in 1:length(N_range)
        N = N_range[i] 

        if results[sigma_values[1]].avg[i] != 0.0
             println("Skipping N = $N, results already loaded/computed.")
            continue
        end
        
        for σ in sigma_values
            
            concurrence_for_avg = zeros(Float64, num_graphs_avg)
            
            for k in 1:num_graphs_avg
                ψ₀, sites = create_MPS(N)
                
                adj_mat = create_weighted_star_adj_mat(N, σ; μ=μ)
                H_mpo = create_weighted_xxz_mpo(N, adj_mat, sites; J=-1.0, Δ=-1.0)

                sweeps = Sweeps(num_sweeps)
                setmaxdim!(sweeps, max_bond_dim_limit)
                setcutoff!(sweeps, cutoff)

                _, ψ_gs = dmrg(H_mpo, ψ₀, sweeps; outputlevel=0)
    
                if N < 2
                    concurrence_for_avg[k] = 0.0
                    continue
                end





                # Calculate 2-site RDM for center (1) and one outer spin (2)
                orthogonalize!(ψ_gs, 1)
                
                s1 = sites[1]
                s2 = sites[2] # Concurrence between center (1) and this spin (2)

                # Contract the first two MPS tensors
                phi = ψ_gs[1] * ψ_gs[2]
                
                # Compute the RDM by contracting with the complex conjugate (bra)
                # This traces over the bond connecting to site 3
                rho_12_tensor = phi * dag(prime(phi, s1, s2))

                # Convert RDM tensor to a standard 4x4 matrix
                C_rows = combiner(s1, s2)
                C_cols = combiner(prime(s1), prime(s2))
                

                # must dag() C_cols to flip the direction of s1' and s2' for correct contraction.
                rho_combined = (rho_12_tensor * C_rows) * dag(C_cols)
                
                rho_matrix = matrix(rho_combined)

                C = calculate_concurrence(complex(rho_matrix))
                concurrence_for_avg[k] = C






            end

            avg_conc = mean(concurrence_for_avg)
            std_dev = std(concurrence_for_avg)
            
            results[σ].avg[i] = avg_conc
            results[σ].err[i] = std_dev
        end
        println("Completed N = $N")

        # Save checkpoint
        try
            jldsave(filename; results, N_range, sigma_values)
            println("Checkpoint saved for N = $N")
        catch e
            println("WARNING: Could not save checkpoint for N = $N. Error: $e")
        end
    end
end



# Parameters
N_range = 4:1:20
sigma_values = [0.0, 0.002] 
num_graphs_avg = 10            
num_sweeps = 10
max_bond_dim_limit = 100
cutoff = 1E-10
μ = 1.0                     

filename = joinpath(@__DIR__, "star_conc_data.jld2")

# Check for existing data and resume
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
println("Data saved successfully.\n")

