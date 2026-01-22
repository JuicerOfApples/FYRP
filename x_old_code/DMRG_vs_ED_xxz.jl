using Markdown
using InteractiveUtils
using Graphs, Random, GraphPlot, Plots, Colors, GraphRecipes
using ITensors, ITensorMPS, LinearAlgebra

function create_XXZ_Ham_MPO(Nodes, adj_mat, Δ, sites)
    """
    Constructs the MPO (Matrix Product Operator) for the XXZ Hamiltonian on an arbitrary graph.

    # Arguments
    - `Nodes::Integer`: The number of nodes in the graph.
    - `adj_mat`: The adjacency matrix of the graph.
    - `Δ::Number`: The anisotropy parameter of the XXZ model.
    - `sites`: The set of local Hilbert spaces (sites) for the MPO.

    # Returns
    - `H::MPO`: The MPO representing the XXZ Hamiltonian.
    """
    ampo = OpSum()
    for i in 1:Nodes
        for j in i+1:Nodes
            if adj_mat[i, j] != 0
                ampo += 4 * adj_mat[i, j] * Δ, "Sz", i, "Sz", j
                ampo += -2 * adj_mat[i, j], "S+", i, "S-", j
                ampo += -2 * adj_mat[i, j], "S-", i, "S+", j
            end
        end
    end
    H = MPO(ampo, sites)
    return H
end

function create_XXZ_Ham_matrix(Nodes, adj_mat, Δ)
    """
    Creates a 2ᴺ x 2ᴺ matrix representation of the XXZ Hamiltonian.

    # Arguments
    - `Nodes::Integer`: The number of spin sites in the system.
    - `adj_mat`: The adjacency matrix of the graph.
    - `Δ::Number`: The anisotropy parameter.

    # Returns
    - `H::Matrix{ComplexF64}`: A dense matrix representing the XXZ Hamiltonian.
    """

    ⊗(x, y) = kron(x, y)

    I = complex([1 0; 0 1])
    σx = complex([0 1; 1 0])
    σy = complex([0 -im; im 0])
    σz = complex([1 0; 0 -1])

    H = complex(zeros(Float64, 2^Nodes, 2^Nodes))

    for i in 1:Nodes
        for j in i+1:Nodes
            if adj_mat[i, j] != 0
                pauli_x_ops = fill(I, Nodes)
                pauli_y_ops = fill(I, Nodes)
                pauli_z_ops = fill(I, Nodes)

                pauli_x_ops[i] = σx; pauli_x_ops[j] = σx
                pauli_y_ops[i] = σy; pauli_y_ops[j] = σy
                pauli_z_ops[i] = σz; pauli_z_ops[j] = σz

                H -= adj_mat[i, j] * foldl(⊗, pauli_x_ops)
                H -= adj_mat[i, j] * foldl(⊗, pauli_y_ops)
                H += Δ * adj_mat[i, j] * foldl(⊗, pauli_z_ops)
            end
        end
    end
    return H
end

function generate_graph(Nodes, Probability)
    """
    Generates a random graph using the Erdős–Rényi model.

    # Arguments
    - `Nodes::Integer`: The number of spin sites in the system.
    - `Probability::Float`: Probability of an edge in the random graph

    # Returns
    - `adj_mat`: The adjacency matrix of the generated graph.
    - `G`: The SimpleGraph object representing the network.
    """

    G = Graphs.erdos_renyi(Nodes, Probability)
    adj_mat = Graphs.adjacency_matrix(G)
    return adj_mat, G
end


function DMRG_simulation(Nodes, adj_mat, Δ, χ)
# USE A COMBINER TENSOR

    """
    Performs DMRG to find the ground state of the XXZ Hamiltonian.

    # Arguments
    - `Nodes::Integer`: The number of spin sites in the system.
    - `adj_mat`: The adjacency matrix of the graph.
    - `Δ::Number`: The anisotropy parameter.
    - `χ::Integer`: Maximum bond dimension

    # Returns
    - `ground_energy::Float64`: The ground state energy found by DMRG.
    - `ground_state_vector::Vector{ComplexF64}`: The ground state wavefunction as a 2ᴺ vector.
    - `H_mpo::MPO`: The Hamiltonian constructed as a Matrix Product Operator.
    """
    # Initialise site inds and initial state
    sites = siteinds("S=1/2", Nodes; conserve_qns=true)
    state = [isodd(i) ? "Up" : "Dn" for i = 1:Nodes]
    psi0 = randomMPS(sites, state, linkdims=10)

    # Set DMRG sweep parameters
    sweeps = Sweeps(15)
    setmaxdim!(sweeps, χ)
    setcutoff!(sweeps, 1E-12)

    # Create Hamiltonian MPO and run DMRG
    H_mpo = create_XXZ_Ham_MPO(Nodes, adj_mat, Δ, sites)
    ground_energy, DMRG_psi = dmrg(H_mpo, psi0, sweeps; outputlevel=0)   # change outputlevel to view sweeps  
    println("DMRG Ground Energy Per Site: ", ground_energy / Nodes)

    # Default ITensor basis order is reversed compared to kron, so must permute
    itensor_gs = ITensors.contract(DMRG_psi)
    array_gs = array(itensor_gs)
    perm = ntuple(i -> Nodes - i + 1, Nodes)
    permuted_array_gs = permutedims(array_gs, perm)
    ground_state_vector = vec(permuted_array_gs)
   
    return ground_energy, ground_state_vector, H_mpo
end

function ED_simulation(Nodes, adj_mat, Δ)
    """
    Performs Exact Diagonalization for the XXZ Hamiltonian.

    # Arguments
    - `Nodes::Integer`: The number of spin sites in the system.
    - `adj_mat`: The adjacency matrix of the graph.
    - `Δ::Number`: The anisotropy parameter.

    # Returns
    - `eigenvalues::Vector{Float64}`: All eigenvalues (energies), sorted.
    - `eigenvectors::Matrix{ComplexF64}`: All eigenvectors (columns of the matrix).
    - `H_matrix::Matrix{ComplexF64}`: The Hamiltonian matrix.
    """

    H_matrix = create_XXZ_Ham_matrix(Nodes, adj_mat, Δ)
    eigen_decomposition = eigen(H_matrix) # Returns a structure with values and vectors
    vals = real.(eigen_decomposition.values) # Energies should be real
    vecs = eigen_decomposition.vectors
    en0 = vals[1]
    
    println("ED Ground Energy Per Site: ", en0 / Nodes)
    
    return vals, vecs, H_matrix
end





function main()

    # parameters
    Nodes = 8       # Number of nodes in the system
    Probability = 0.4   # Probability of an edge in the random graph
    Δ = 1.0         # Anisotropy parameter, Δ=1 for the Heisenberg model
    χ = 100         # Maximum bond dimension for DMRG


    adj_mat, tensor_network_graph = generate_graph(Nodes, Probability)
    plot_title = "Erdős–Rényi Graph (p = $Probability)"
    display(gplot(tensor_network_graph, layout = circular_layout, nodelabel = 1:Nodes))




    
    # DMRG gives the lowest energy and its corresponding state
    DMRG_energy, DMRG_ground_state, _ = DMRG_simulation(Nodes, adj_mat, Δ, χ)

    # ED gives all energies and states
    ED_eigenvals, ED_eigenvecs, _ = ED_simulation(Nodes, adj_mat, Δ)
    




    
    # Comparison of results of both simulations

    # Ground state from ED results
    ED_ground_energy = ED_eigenvals[1]
    ED_ground_state = ED_eigenvecs[:, 1]
    
    # Comparison of ground state energies
    println("")
    println(repeat("-", 40))
    println("DMRG Ground Energy: ", DMRG_energy)
    println("ED Ground Energy:   ", ED_ground_energy)
    println("Absolute Difference:  ", abs(DMRG_energy - ED_ground_energy))
    println(repeat("-", 40))
    
    # Compare the ground state wavefunctions
    # calculate the squared overlap (fidelity), should be 1.
    fidelity = abs(dot(DMRG_ground_state, ED_ground_state))^2
    println()
    println(repeat("-", 40))
    println("Fidelity (Overlap |<ψ_ED|ψ_DMRG>|²): ", fidelity)
    println("Error bound from fidelity: ", abs(1 - fidelity))

    if isapprox(fidelity, 1.0; atol=1e-6)
        println("The ground state vectors are consistent.")
    else
        println("Warning: The ground state vectors differ significantly.")
    end
    println(repeat("-", 40))
end

# Run the main function
main()

"""
next steps ideas
- make graph showing the ground states for changing the Probability from 0.01 to 0.99
- graph of the fidelity to show how that changes as probability changes
- use sigma = 0 
"""