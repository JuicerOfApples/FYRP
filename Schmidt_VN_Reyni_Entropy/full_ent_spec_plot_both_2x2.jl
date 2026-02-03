using JLD2, Plots, LaTeXStrings

data_filename_1 = joinpath(@__DIR__, "full_ent_spec_data_0.0.jld2")
data_filename_2 = joinpath(@__DIR__, "full_ent_spec_data_0.002.jld2")
output_filename = joinpath(@__DIR__, "full_ent_spec_plot_both_2x2.png")

try
    data_00 = JLD2.load(data_filename_1)
    results_00 = data_00["entanglement_spectrum_results"]
    sigma_00 = data_00["σ_val"]

    data_0002 = JLD2.load(data_filename_2)
    results_0002 = data_0002["entanglement_spectrum_results"]
    sigma_0002 = data_0002["σ_val"]

    target_N = [20, 40, 60, 80]
    fnt_title, fnt_guide, fnt_tick, fnt_legend = 24, 20, 14, 14

    p_layout = plot(
        layout = (2, 2), 
        size = (1000, 900), 
        plot_title = "Schmidt Spectrum Comparison",
        plot_titlefontsize = fnt_title,
        legend = :topright, 
        legendfontsize = fnt_legend,
        margin = 10Plots.mm      
    )
    
    x_lims = (0, 30)

    for (i, N) in enumerate(target_N)
        label_00 = (i == 1) ? "σ=$sigma_00" : ""
        label_0002 = (i == 1) ? "σ=$sigma_0002" : ""

        if haskey(results_0002, N)
            bar!(p_layout, subplot = i, results_0002[N] .^ 2,
                title = "$N nodes", titlefontsize = fnt_guide,
                xlabel = "Schmidt Index" * L"\: j", ylabel = L"\lambda_j",
                guidefontsize = fnt_guide, tickfontsize = fnt_tick,
                xlims = x_lims, label = label_0002,
                seriescolor = :darkorange, linecolor = :darkorange,
                bar_width = 1, gap = 0, alpha = 0.8, framestyle = :axes)
        end
        
        if haskey(results_00, N)
            bar!(p_layout, subplot = i, results_00[N] .^ 2,
                label = label_00, seriescolor = :purple, linecolor = :purple,
                bar_width = 1, gap = 0, alpha = 0.4, framestyle = :axes)
        end
    end
    savefig(p_layout, output_filename)
catch e
    showerror(stdout, e)
end