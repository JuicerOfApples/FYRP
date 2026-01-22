using JLD2
using Plots

data_filename_1 = joinpath(@__DIR__, "full_ent_spec_data_0.0.jld2")
data_filename_2 = joinpath(@__DIR__, "full_ent_spec_data_0.002.jld2")
output_filename = joinpath(@__DIR__, "full_ent_spec_plot_both_small.png")


try
    println("Loading data from $data_filename_1...")
    if !isfile(data_filename_1)
        println("ERROR: File not found: $data_filename_1")
        throw(SystemError("File not found", 2))
    end
    data_00 = JLD2.load(data_filename_1)
    results_00 = data_00["entanglement_spectrum_results"]
    N_values = data_00["N_values"]
    sigma_00 = data_00["σ_val"]
    max_coeffs_00 = data_00["max_coeffs_to_store"]




    println("Loading data from $data_filename_2...")
    if !isfile(data_filename_2)
        println("ERROR: File not found: $data_filename_2")
        throw(SystemError("File not found", 2))
    end
    data_0002 = JLD2.load(data_filename_2)
    results_0002 = data_0002["entanglement_spectrum_results"]
    N_values_0002 = data_0002["N_values"]
    sigma_0002 = data_0002["σ_val"]
    max_coeffs_0002 = data_0002["max_coeffs_to_store"]



    plot_width_px = 2400
    plot_height_px = 1000
    
    p_layout = plot(
        layout = (2, 4),
        size = (plot_width_px, plot_height_px),
        plot_title = "Full Entanglement Spectrum Comparison (σ=$(sigma_00) vs σ=$(sigma_0002))",
        plot_titlefontsize = 20,
        legend = :topright, 
        top_margin = 10Plots.mm,
        margin = 10Plots.mm      
    )
    
    x_axis_label = "Schmidt Coefficients"
    y_axis_label = "Coefficient Values"

    # y_lims = (0, 0.6) 
    # x_lims = (0, 50)

    y_lims = (0, 0.002) 
    x_lims = (10, 50)

    println("Generating 8 subplots...")
    for (i, N) in enumerate(N_values)
        
        title_str = "$N nodes"
        
        has_data_00 = haskey(results_00, N)
        has_data_0002 = haskey(results_0002, N)

        if !has_data_00
            println("  WARNING: No data found for N = $N in file 00.")
            title_str *= " (Missing σ=0.0)"
        end
        if !has_data_0002
            println("  WARNING: No data found for N = $N in file 0002.")
            title_str *= " (Missing σ=0.002)"
        end
        
        # Define labels for the first subplot (i=1)
        # This creates a single legend for the whole figure
        label_00 = (i == 1) ? "σ=$sigma_00" : ""
        label_0002 = (i == 1) ? "σ=$sigma_0002" : ""

        
        # Plot data for sigma = 0.002 (orange)
        if has_data_0002
            coeffs_0002 = results_0002[N]
            bar!(
                p_layout,
                subplot = i,
                coeffs_0002,
                title = title_str, 
                xlabel = x_axis_label,
                ylabel = y_axis_label,
                ylims = y_lims,
                xlims = x_lims,
                label = label_0002,
                seriescolor = :darkorange, 
                linecolor = :darkorange,
                bar_width = 1,
                gap = 0,
                alpha = 0.8 
                )
            end
            
            # Plot data for sigma = 0.0 (purple) on top
            if has_data_00
                coeffs_00 = results_00[N]
                bar!(
                    p_layout,
                    subplot = i,
                    coeffs_00,
                    title = title_str,
                    xlabel = x_axis_label,
                    ylabel = y_axis_label,
                    ylims = y_lims,
                    xlims = x_lims,
                    label = label_00,
                    seriescolor = :purple, 
                    linecolor = :purple,
                    bar_width = 1,
                    gap = 0,
                    alpha = 0.4 
                )
            end

        if !has_data_00 && !has_data_0002
            plot!(p_layout, subplot = i, title = "$N nodes (No Data)", framestyle=:box)
        end
    end

    savefig(p_layout, output_filename)
    println("\nPlot saved successfully to $output_filename")

catch e
    showerror(stdout, e)
    println()
end