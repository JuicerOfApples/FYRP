using JLD2
using Plots
using Statistics

filename = joinpath(@__DIR__, "big_N_scaling_data.jld2")



data = load(filename)
results = data["results"]
N_range = data["N_range"]
sigma_values = data["sigma_values"]

p = plot(
    title = "Bond Dimension Scaling with System Size",
    xlabel = "System Size (N)",
    ylabel = "Max Bond Dimension (χ)",
    legend = :topleft,
    grid = :true,
    minorgrid = :true,
    dpi = 300,
    size = (800, 600)
)


for σ in sort(sigma_values)
    

    res_data = results[σ]
    
    avg_dim = res_data.avg
    std_dev = res_data.err

    plot!(p, 
        N_range, 
        avg_dim, 
        yerror = std_dev, 
        label = "σ = $σ",
        marker = :circle,
        markersize = 4,
        linewidth = 2
    )
end

savefig(p, joinpath(@__DIR__, "big_N_scaling_plot.png"))
