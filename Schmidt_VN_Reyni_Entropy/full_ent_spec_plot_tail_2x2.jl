using JLD2, Plots, LaTeXStrings

file_00 = joinpath(@__DIR__, "full_ent_spec_data_0.0.jld2")
file_002 = joinpath(@__DIR__, "full_ent_spec_data_0.002.jld2")
output_file = joinpath(@__DIR__, "full_ent_spec_plot_tail_2x2.png")

d00 = JLD2.load(file_00)
d002 = JLD2.load(file_002)
r00 = d00["entanglement_spectrum_results"]
r002 = d002["entanglement_spectrum_results"]
sigma_00, sigma_002 = d00["σ_val"], d002["σ_val"]

target_N = [20, 40, 60, 80]
fnt_title, fnt_guide, fnt_tick, fnt_legend = 24, 20, 14, 14

p = plot(
    layout = (2, 2),
    size = (1000, 900),
    plot_title = "Schmidt Spectrum Tail Comparison",
    plot_titlefontsize = fnt_title,
    legend = :topright,
    legendfontsize = fnt_legend,
    margin = 10Plots.mm
)

for (i, N) in enumerate(target_N)
    v00 = haskey(r00, N) ? sort(r00[N] .^ 2, rev=true) : nothing
    v002 = haskey(r002, N) ? sort(r002[N] .^ 2, rev=true) : nothing

    label_00 = (i == 1) ? "σ=$sigma_00" : ""
    label_002 = (i == 1) ? "σ=$sigma_002" : ""

    # Plot for sigma = 0.002 (Orange Circles)
    if v002 !== nothing
        mask = v002 .> 1e-12
        plot!(p, subplot=i, (1:length(v002))[mask], v002[mask], 
            seriestype=:path, color=:darkorange, alpha=0.6, label="")

        plot!(p, subplot=i, (1:length(v002))[mask], v002[mask], 
            seriestype=:scatter, markershape=:circle, markersize=4, 
            markerstrokewidth=0.2, markerstrokecolor=:black,
            color=:darkorange, alpha=0.9, label=label_002)
    end

    # Plot for sigma = 0.0 (Purple Squares)
    if v00 !== nothing
        mask = v00 .> 1e-12
        plot!(p, subplot=i, (1:length(v00))[mask], v00[mask], 
            seriestype=:path, color=:purple, alpha=0.6, label="")

        plot!(p, subplot=i, (1:length(v00))[mask], v00[mask], 
            seriestype=:scatter, markershape=:circle, markersize=4, 
            markerstrokewidth=0.2, markerstrokecolor=:black,
            color=:purple, alpha=0.6, label=label_00)
    end

    plot!(p, subplot=i, title = "$N nodes", titlefontsize = fnt_guide,
        xlabel = "Schmidt Index" * L"\: j", ylabel = L"λ_j^2 ",
        guidefontsize = fnt_guide, tickfontsize = fnt_tick,
        yaxis = :log10, framestyle = :axes, 
        grid = true, gridalpha = 0.15)
end

savefig(p, output_file)
println("Refined plot saved to $output_file")