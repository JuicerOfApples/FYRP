using Graphs
using Plots
using GraphRecipes
using LinearAlgebra
using NetworkLayout
using Random 

Random.seed!(42) 

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
Creates a weighted adjacency matrix for a star graph.
"""
function create_weighted_star_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    A = zeros(Float64, N, N)
    if N < 2
        return A
    end
    for j in 2:N
        weight = (σ == 0.0) ?
μ : (μ + σ * randn())
        A[1, j] = A[j, 1] = weight
    end
    return A
end

"""
Creates a weighted adjacency matrix for a linear chain graph.
"""
function create_weighted_chain_adj_mat(N::Int, σ::Float64; μ::Float64=1.0)
    A = zeros(Float64, N, N)
    if N < 2
        return A
    end
    
    # Create N-1 random couplings for the N-1 bonds 
    couplings = [μ + σ * randn() for _ in 1:(N-1)] 

    for i in 1:(N-1)
        # Only add nearest-neighbor terms 
        coupling_strength = couplings[i]
        A[i, i+1] = A[i+1, i] = coupling_strength
    end
    return A
end


"""
Generic function to generate and save a graph visualization.
"""
function plot_graph_to_file(
    adj_mat::Matrix{Float64}, 
    N::Int, 
    σ::Float64, 
    μ::Float64, 
    plot_title::String, 
    filename::String
)
    
    g = Graphs.Graph(adj_mat) 

    layout_coords = spring(g) 
    
    x_coords = [p[1] for p in layout_coords]
    y_coords = [p[2] for p in layout_coords]
    
    cmap = cgrad(:bwr, [μ - 2σ, μ, μ + 2σ], categorical=false)
    
    edge_width_matrix = (abs.(adj_mat) ./ μ) .* 2

    println("Generating plot for $filename...")

    gr() 

    p = graphplot(
        g,
        # Pass coordinates directly to x and y keywords
        x = x_coords,
        y = y_coords,
        names = 1:N,
        fontsize = 10,
        nodesize = 0.5,
        nodeshape = :circle,
        nodecolor = :lightblue,
        
        # Edge properties
        edge_width = edge_width_matrix, 
        edge_z = adj_mat,           
        seriescolor = cmap,
        
        title = "$plot_title (N = $N, σ = $σ, μ = $μ) \nYellow = Strong, Black = Weak",
        colorbar_title = "Coupling Strength (J_ij)",
        size = (800, 800),
        colorbar = true,
        dpi = 150
    )

    savefig(p, filename)
    println("Plot saved to: $filename")
end


"""
Main function to generate all three plots.
"""
function generate_all_plots()

    N = 5      
    μ = 1.0     
    σ = 0.75   

    # Connected Graph
    adj_con = create_weighted_adj_mat(N, σ; μ=μ)
    filename_con = joinpath(@__DIR__, "dis_graphs_plot_con.png")
    plot_graph_to_file(adj_con, N, σ, μ, "Disordered Connected Graph", filename_con)

    # Linear Chain Graph
    adj_lin = create_weighted_chain_adj_mat(N, σ; μ=μ)
    filename_lin = joinpath(@__DIR__, "dis_graphs_plot_lin.png")
    plot_graph_to_file(adj_lin, N, σ, μ, "Disordered Linear Chain", filename_lin)

    # Star Graph
    adj_star = create_weighted_star_adj_mat(N, σ; μ=μ)
    filename_star = joinpath(@__DIR__, "dis_graphs_plot_star.png")
    plot_graph_to_file(adj_star, N, σ, μ, "Disordered Star Graph", filename_star)
    
end

generate_all_plots()