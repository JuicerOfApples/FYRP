using ITensors, ITensorMPS
using LinearAlgebra
using Plots
using Random
using Statistics

# --- Configuration ---
# Set the range of system sizes to match your report (N = 3 to 13)
N_range = 3:13
σ_disorder = 0.002  # Standard deviation for random weights J_ij (from report)
sweeps = Sweeps(30) # 30 sweeps as specified in your methods


# Function to generate Gaussian random weights for a fully connected graph
function generate_couplings(N, σ)
    # Fully connected adjacency matrix with Gaussian weights
    # J_ij = J_ji, zero diagonal
    J = zeros(Float64, N, N)
    for i in 1:N
        for j in i+1:N
            weight = randn() * σ
            J[i, j] = weight
            J[j, i] = weight
        end
    end
    return J
end

# Arrays to store results
overlaps = Float64[]
errors = Float64[] # To plot 1 - Overlap if desired

println("Starting comparison between DMRG and ED...")

for N in N_range
    println("Processing N = $N ...")
    
    # 1. Define Sites
    sites = siteinds("S=1/2", N)
    
    # 2. Construct Hamiltonian
    J_mat = generate_couplings(N, σ_disorder)
    os = OpSum()
    for i in 1:N
        for j in i+1:N
            val = J_mat[i, j]
            os += val, "Sx", i, "Sx", j
            os += val, "Sy", i, "Sy", j
            os += val, "Sz", i, "Sz", j
        end
    end
    H = MPO(os, sites)

    # 3. Solve with DMRG (Updated Syntax)
    psi0 = randomMPS(sites, 10)
    
    # FIX: Pass parameters directly here instead of using a 'Sweeps' object
    energy_dmrg, psi_dmrg = dmrg(H, psi0; 
                                 nsweeps=30, 
                                 maxdim=[10, 20, 100, 100, 200], 
                                 cutoff=1E-10, 
                                 outputlevel=0)

    # 4. Solve with Exact Diagonalization (ED)
    # Contract the MPO into a single dense tensor to treat as a matrix
    # WARNING: This scales exponentially. N=13 is roughly the limit for laptop RAM.
    H_dense = prod(H) 
    H_mat = Array(H_dense, sites...) 
    H_mat = reshape(H_mat, 2^N, 2^N) # Reshape to square matrix
    
    # Diagonalize
    evals, evecs = eigen(Hermitian(H_mat))
    psi_ed_vec = evecs[:, 1] # Ground state vector
    
    # 5. Calculate Overlap
    # Convert DMRG MPS to a dense vector to compare
    psi_dmrg_dense = prod(psi_dmrg)
    psi_dmrg_vec = reshape(Array(psi_dmrg_dense, sites...), 2^N)
    
    # Calculate overlap (inner product)
    # Taking absolute value handles global phase differences
    ovlp = abs(dot(conj(psi_dmrg_vec), psi_ed_vec))
    
    push!(overlaps, ovlp)
    push!(errors, 1.0 - ovlp)
    
    println("  N=$N | Overlap = $ovlp")
end

# --- Plotting ---
# Replicating the style of Figure 6
p = plot(N_range, overlaps,
    title = "Overlap of DMRG and Exact Diagonalization",
    xlabel = "Number of Nodes (N)",
    ylabel = "Overlap |<ψ_DMRG | ψ_ED>|",
    legend = false,
    marker = :circle,
    color = :blue,
    linewidth = 2,
    framestyle = :box,
    grid = true
)

# Save the figure
savefig(p, "overlap_plots.png")
println("Plot saved to overlap_plots.png")