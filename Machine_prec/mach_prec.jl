
using Pkg
Pkg.add(["LinearAlgebra", "ITensors", "ITensorMPS", "Statistics", "JLD2", "FileIO"])

using LinearAlgebra
using ITensors
using ITensorMPS
using Statistics
using JLD2, FileIO



J = Δ = -1
N_vals = [10:2:40; 80; 100; 120]
σ_vals = [0.000, 0.001, 0.002, 0.005, 0.1, 0.3, 0.5]
μ = 1.0

NUM_SWEEPS = 30
MAX_BOND_DIM = typemax(Int) 
ACC = eps(Float64) 

AVG_OVER_PLOTS = 1

function generate_fully_connected_wam(N::Int, σ::Float64; μ::Float64=1.0)
    """
    Create a weighted adjacency matrix for a fully connected graph of N nodes. μ should always be one really.
    """
 
    A = zeros(Float64, N, N)
    for i in 1:N
        for j in (i+1):N
            weight = μ + σ * randn() # weight from normal distribution with mean μ and std σ
            A[i, j] = weight
            A[j, i] = weight 
        end
    end
    return A
end 

function create_xxz_hamiltonian_mpo(N, adj_mat, J, Δ, sites)
    """Create the XXZ Hamiltonian as an MPO given an adjacency matrix."""
    ampo = OpSum()
    for i = 1:N-1
        for j = i+1:N
            weight = adj_mat[i, j]
            if weight != 0.0

                ampo += weight * J/2, "S+", i, "S-", j
                ampo += weight * J/2, "S-", i, "S+", j
                ampo += weight * J * Δ, "Sz", i, "Sz", j
            end
        end
    end
    H = MPO(ampo, sites)
    return H
end 

function solve_xxz_hamiltonian_dmrg(H, ψ0, sweeps::Int=10, bond_dim::Int=1000, cutoff::Float64=1E-14)
    """Solves the XXZ Hamiltonian using DMRG with given parameters. Returns the ground state energy and MPS. """
    swps = Sweeps(sweeps)
    setmaxdim!(swps, bond_dim)
    setcutoff!(swps, cutoff)
    E, ψ = dmrg(H, ψ0, swps; outputlevel=0)
    return E, ψ 
end

function create_MPS(L::Int, conserve_qns::Bool=true)
    """Create a random MPS for a spin-1/2 chain of length L with bond dimension Χ."""
    sites = siteinds("S=1/2", L; conserve_qns=conserve_qns) 
    # create a random MPS with bond dimension Χ
    init_state = [isodd(i) ? "Up" : "Dn" for i = 1:L] # antiferromagnetic ground state

    ψ0 = MPS(sites, init_state)
    return ψ0, sites
end

function find_ground_state_mps(N, J, Δ, σ, μ)
    ψ_mps_N, sites_N = create_MPS(N)
    wam_N = generate_fully_connected_wam(N, σ; μ)
    H_N = create_xxz_hamiltonian_mpo(N, wam_N, J, Δ, sites_N)
    _, ψ_gs_N = solve_xxz_hamiltonian_dmrg(H_N, ψ_mps_N, NUM_SWEEPS, MAX_BOND_DIM, ACC)
    return ψ_gs_N
end

function get_entanglement_data(psi, b)
    orthogonalize!(psi, b)
    s = siteind(psi, b)
    
    local U, S, V
    if b==1
        U, S, V = svd(psi[b], (s,))
    else
        l = linkind(psi, b-1)
        U, S, V = svd(psi[b], (l, s))
    end

    schmidt_coefs = diag(S)
    schmidt_spectrum = schmidt_coefs .^ 2 # square all schmidt_coefs
    schmidt_spectrum = schmidt_spectrum[schmidt_spectrum .> 1e-15] # filter out really small p so we don't get NaN entropies
    schmidt_spectrum ./= sum(schmidt_spectrum) # normalise

    vn_entropy = -sum(p * log(p) for p in schmidt_spectrum) # von neuman entropy

    return schmidt_coefs, vn_entropy
end

function main()
    filename = joinpath(@__DIR__, "mach_prec_data.jld2")
    for N in N_vals
        for σ in σ_vals

            # file management
            group_path = "N=$(N)/sigma=$(σ)"

            data_exists = false
            if isfile(filename)   
                jldopen(filename, "r") do file
                    data_exists = haskey(file, group_path)
                end
            end
            if data_exists
                println("Skipping N=$N, σ=$σ (data already exists).")
                continue
            else
                println("Running N=$N, σ=$σ.")
            end

            averaging = σ == 0.00 ? 1 : AVG_OVER_PLOTS
            println("   Averaging $averaging times.")


            bond_dims = []
            all_schmidt_coefs = []
            vn_entropies = []

            for a in 1:averaging
                gs_mps = find_ground_state_mps(N, J, Δ, σ, μ)
                bond_dim = maxlinkdim(gs_mps)
                push!(bond_dims, bond_dim)
                println("       Found bond dimension for run $a.")
                
                # define bipartite cut at N/2
                b = ceil(Int, N / 2)

                coefs, vn_ent = get_entanglement_data(gs_mps, b)

                push!(all_schmidt_coefs, coefs)
                push!(vn_entropies, vn_ent)
                println("       Found Schmidt coefficients and Von-Neumann entropy for run $a.")
            end

            avg_bond_dim, bond_dim_err = 0, 0.0

            # find average bond dim + error on mean
            if length(bond_dims) > 1
                avg_bond_dim = mean(bond_dims)
                bond_dim_err = std(bond_dims) / sqrt(length(bond_dims))
            else
                avg_bond_dim = bond_dims[1] # error is zero, only one plot
            end
            println("   Found average bond dimension $avg_bond_dim ± $bond_dim_err")

            # also want vn entropy
            avg_entropy, entropy_err = 0.0, 0.0
            if length(vn_entropies) > 1
                avg_entropy = mean(vn_entropies)
                entropy_err = std(vn_entropies) / sqrt(length(vn_entropies))
            else
                avg_entropy = vn_entropies[1]
            end
            println("   Found average VN Entropy $avg_entropy ± $entropy_err")

            # averaging schmidt spectrum
            # this contains some problems
            # firstly, run 1 might have bond dimension of 10, but then the next one will have bond dim of 12 say.
            # then, the spectrum lengths are not the same, so we run into problems.
            # so, we can use a different approach, to go through each index of schmit coefficient, 
            #   and find any results that had that, until we run out of indicies with any results

            max_len = maximum(length(coefs) for coefs in all_schmidt_coefs)
            schmidt_coefs_by_index = Vector{Vector{Float64}}(undef, max_len)
            sc_k_averages, sc_k_errors = zeros(Float64, max_len), zeros(Float64, max_len)

            for k in 1:max_len
                # collect the k-th coefficient from EVERY run
                # If a run doesn't have a k-th coefficient, get() returns 0.0
                k_th_coeffs = [get(run_coeffs, k, 0.0) for run_coeffs in all_schmidt_coefs]
                schmidt_coefs_by_index[k] = k_th_coeffs
                # now, average this list
                sc_k_averages[k] = mean(k_th_coeffs)
                if averaging > 1
                    sc_k_errors[k] = std(k_th_coeffs) / sqrt(length(k_th_coeffs))
                end 
            end
            println("    Found average Schmidt coefficients and errors.")

            results_dict = Dict(
                "bond_dims" => bond_dims,
                "avg_bond_dim" => avg_bond_dim,
                "err_bond_dim" => bond_dim_err,

                "vn_entropies" => vn_entropies,
                "avg_entropy" => avg_entropy,
                "err_entropy" => entropy_err,

                "schmidt_coefs" => all_schmidt_coefs,
                "avg_schmidt_spectrum" => sc_k_averages,
                "err_schmidt_spectrum" => sc_k_errors,

                "parameters" => Dict(
                    "N" => N,
                    "sigma" => σ,
                    "J" => J,
                    "Delta" => Δ,
                    "mu" => μ,
                    "num_sweeps" => NUM_SWEEPS,
                    "num_runs_averaged" => averaging
                )
            )

            jldopen(filename, "a") do file
                file[group_path] = results_dict
            end

            println("Save complete...")
            println("\n")

        end
    end
end

main()
