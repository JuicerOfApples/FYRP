using JLD2
using Plots
using Statistics


target_file = joinpath(@__DIR__, "renyi_entropy_data_0.0.jld2")

if !isfile(target_file)
    error("Data file not found: $target_file")
end

data = load(target_file)

entropy_results = data["entropy_results"]   
N_range = data["N_range_used"]              
alpha_val = data["alpha_used"]              


p = plot(
    title = "Rényi Entropy Scaling (α = $alpha_val)",
    xlabel = "System Size (N)",
    ylabel = "Rényi Entropy S_$alpha_val",
    legend = :topleft,
    grid = true,
    guidefontsize = 12,
    tickfontsize = 10,
    margin = 5Plots.mm
)


sigmas = sort(collect(keys(entropy_results)))

for σ in sigmas
    print(σ)
    entropies = entropy_results[σ]
    
    plot!(p, N_range, entropies, 
          label = "σ = $σ", 
          marker = :circle, 
          linewidth = 2,
          markersize = 4)
end

output_filename = joinpath(@__DIR__, "renyi_entropy_alpha_$(alpha_val)_plot.png")
savefig(p, output_filename)
