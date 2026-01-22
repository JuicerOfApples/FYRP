using JLD2
using FileIO
using Plots
using Statistics

filename = joinpath(@__DIR__, "high_prec_spec_data_0.002.jld2")
output_file = joinpath(@__DIR__, "high_prec_spec_plot_0.002.png")

if !isfile(filename)
    error("Data file not found: $filename")
end

data = load(filename)
results = data["entanglement_spectrum_results"] 
N_values = data["N_values"]
sigma_val = data["σ_val"]

entropies = Float64[]
sorted_N = sort(collect(keys(results)))
filter!(n -> n <= 100, sorted_N)

largest_N = maximum(sorted_N)
largest_N_spectrum = Float64[]

for N in sorted_N
    coeffs = results[N]
    
    # Calculate probabilities p = λ^2
    p = coeffs .^ 2
    
    # Filter out numerical zeros
    filter!(x -> x > 1e-20, p)
    
    # Normalization factor
    num_graphs = sum(p) 
    
    # Calculate Von Neumann Entropy
    S = -sum(p .* log.(p)) / num_graphs
    
    push!(entropies, S)
    
    if N == largest_N
        global largest_N_spectrum = coeffs
    end
end


plot_width_px = 1200
plot_height_px = 600

p = plot(
    layout = (1, 2),
    size = (plot_width_px, plot_height_px),
    plot_title = "Entanglement Entropy & Tail Spectrum (σ = $sigma_val)",
    plot_titlefontsize = 20,
    top_margin = 5Plots.mm,
    bottom_margin = 10Plots.mm,
    left_margin = 10Plots.mm,
    right_margin = 5Plots.mm,
    margin = 5Plots.mm
)

plot!(p, subplot=1, 
    sorted_N, 
    entropies, 
    seriestype = :scatter,
    markershape = :circle,
    markersize = 4,
    markerstrokewidth = 0,
    color = :dodgerblue,
    label = "S_VN",
    title = "Von Neumann Entropy Scaling",
    xlabel = "System Size (N)", 
    ylabel = "Von Neumann Entropy",
    legend = :bottomright,
    framestyle = :box,
    grid = true
)
plot!(p, subplot=1, sorted_N, entropies, seriestype=:path, color=:dodgerblue, alpha=0.6, label="")


num_graphs_int = round(Int, sum(largest_N_spectrum .^ 2))
unique_coeffs = largest_N_spectrum[1:num_graphs_int:end]

filter!(x -> x > 1e-20, unique_coeffs)

x_indices = 1:length(unique_coeffs)

plot!(p, subplot=2, 
    x_indices, 
    unique_coeffs, 
    seriestype = :scatter,
    markershape = :circle,
    markersize = 3,
    markerstrokewidth = 0,
    color = :crimson,
    label = "N=$largest_N",
    title = "Schmidt Coefficient Tail (N = $largest_N)",
    xlabel = "Coefficient Index", 
    ylabel = "Schmidt Coefficients (log)",
    yaxis = :log10, 
    framestyle = :box,
    grid = true
)
plot!(p, subplot=2, x_indices, unique_coeffs, seriestype=:path, color=:crimson, alpha=0.5, label="")

savefig(p, output_file)
