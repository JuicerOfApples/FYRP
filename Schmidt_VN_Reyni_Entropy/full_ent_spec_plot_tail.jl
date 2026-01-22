using JLD2, Plots

file_00 = joinpath(@__DIR__, "full_ent_spec_data_0.0.jld2")
file_002 = joinpath(@__DIR__, "full_ent_spec_data_0.002.jld2")
output_file = joinpath(@__DIR__, "tail_plot.png")

if !isfile(file_00) || !isfile(file_002)
    error("Data files not found.")
end

d00 = JLD2.load(file_00)
d002 = JLD2.load(file_002)

r00 = d00["entanglement_spectrum_results"]
r002 = d002["entanglement_spectrum_results"]
N_vals = d00["N_values"]
sigma_00 = d00["σ_val"]
sigma_002 = d002["σ_val"]

plot_width_px = 2400
plot_height_px = 1000

p = plot(
    layout = (2, 4),
    size = (plot_width_px, plot_height_px),
    plot_title = "Schmidt Coefficient Tail Behavior (Log Scale)",
    plot_titlefontsize = 20,
    legend = :topright,
    top_margin = 10Plots.mm,
    margin = 10Plots.mm
)

for (i, N) in enumerate(N_vals)
    
    title_str = "$N nodes"
    
    v00 = haskey(r00, N) ? sort(r00[N], rev=true) : nothing
    v002 = haskey(r002, N) ? sort(r002[N], rev=true) : nothing

    label_00 = (i == 1) ? "σ=$sigma_00" : ""
    label_002 = (i == 1) ? "σ=$sigma_002" : ""

    if v002 !== nothing
        mask = v002 .> 0
        y = v002[mask]
        x = (1:length(v002))[mask]
        
        plot!(p, subplot=i, x, y, 
            seriestype=:scatter, 
            markershape=:circle, 
            markersize=3, 
            markerstrokewidth=0, 
            color=:darkorange, 
            alpha=0.8, 
            label=label_002
        )
        plot!(p, subplot=i, x, y, seriestype=:path, color=:darkorange, alpha=0.5, label="")
    end

    if v00 !== nothing
        mask = v00 .> 0
        y = v00[mask]
        x = (1:length(v00))[mask]
        
        plot!(p, subplot=i, x, y, 
            seriestype=:scatter, 
            markershape=:rect, 
            markersize=3, 
            markerstrokewidth=0, 
            color=:purple, 
            alpha=0.4, 
            label=label_00
        )
        plot!(p, subplot=i, x, y, seriestype=:path, color=:purple, alpha=0.5, label="")
    end

    plot!(p, subplot=i,
        title = title_str,
        xlabel = "Schmidt Index",
        ylabel = "Schmidt Coefficients (log)",
        yaxis = :log10,
        framestyle = :box
    )
end

savefig(p, output_file)
println("Plot saved to $output_file")