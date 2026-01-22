using Markdown
using InteractiveUtils
using Graphs, Random, GraphPlot, Plots, Colors, GraphRecipes
using ITensors, ITensorMPS, LinearAlgebra
using Compose

"""
Creates a weighted adjacency matrix for a star graph with N nodes Node 1 is the center.
"""
function weighted_star_adjmat(N::Int, σ::Float64; μ::Float64=1.0)
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

"""
Places node 1 at the center (0,0) and all other nodes in a circle around it.
"""
function star_layout(g::AbstractGraph)
    N = nv(g)
    locs_x = zeros(Float64, N)
    locs_y = zeros(Float64, N)
    
    locs_x[1] = 0.0
    locs_y[1] = 0.0
    
    num_outer = N - 1
    angles = range(0, 2π, length=num_outer + 1)[1:num_outer] 
    
    for (i, node_idx) in enumerate(2:N)
        locs_x[node_idx] = cos(angles[i])
        locs_y[node_idx] = sin(angles[i])
    end
    
    return (locs_x, locs_y)
end



N_nodes = 8
sigma = 0.02
mu = 1

adjmat = weighted_star_adjmat(N_nodes, sigma)
G = Graphs.SimpleGraph(adjmat)


edge_colors = Color[]
for edge in edges(G)

    weight = adjmat[src(edge), dst(edge)]
    
    if weight > mu
        push!(edge_colors, colorant"salmon") # > 1.0
    
    elseif weight < mu
        push!(edge_colors, colorant"violet")  # < 1.0
    else
        push!(edge_colors, colorant"grey") # 1.0
    end
end

layout = star_layout(G)

node_labels = [string(i) for i in 1:N_nodes]

p = GraphPlot.gplot(
    G, 
    layout=star_layout,           
    nodelabel=node_labels,
    nodesize=0.3,
    nodelabelsize=3,
    edgelinewidth=2.0,
    edgestrokec=edge_colors  
)

draw(PNG(joinpath(@__DIR__,"star_graph_plot.png")), p)