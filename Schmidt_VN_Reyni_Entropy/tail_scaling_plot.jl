using JLD2, FileIO
using Plots
using LaTeXStrings

data_file = joinpath(@__DIR__, "tail_analysis_data.jld2")
analysis = load(data_file)

output_filename = joinpath(@__DIR__, "tail_scaling_plot.png")

p = plot(
    size = (800, 600),
    dpi = 300,
    xlabel = "System Size (N)",
    ylabel = "Tail Weight (Truncation Error)",
    title = "Scaling of Tail Weight with System Size",
    margin = 5Plots.mm,
    legend = :bottomleft,
    yaxis = :log10 # Log scale
)

c_00 = :purple
c_002 = :darkorange

for (sig, data) in analysis
    if sig == "0.0" || sig == 0.0
        lbl = "σ=0.0"
        col = c_00
        shp = :rect
    else
        lbl = "σ=0.002"
        col = c_002
        shp = :circle
    end
    
    N_vals = data["N"]
    weights = data["tail_weight"]
    
    # Filter out zero values to avoid log-plot errors 
    mask = weights .> 0
    
    if count(mask) == 0
        println("Warning: All tail weights for $lbl are 0.0. Nothing to plot for this series.")
        continue
    end

    clean_N = N_vals[mask]
    clean_weights = weights[mask]
    
    plot!(p, clean_N, clean_weights,
        label = lbl,
        seriestype = :scatter,
        markershape = shp,
        color = col,
        lw = 2
    )
    plot!(p, clean_N, clean_weights, seriestype=:path, color=col, alpha=0.5, label="")
end

# Add a reference line for Machine Precision
epss = eps(Float64)
hline!(p, [epss], label="Machine Precision", color=:grey, linestyle=:dash)

savefig(p, output_filename)
println("Plot saved to $output_filename")