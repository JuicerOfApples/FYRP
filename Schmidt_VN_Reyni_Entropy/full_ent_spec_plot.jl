using JLD2
using Plots
# using HDF5

data_filename = joinpath(@__DIR__,"full_ent_spec_data_0.0.jld2")
output_filename = joinpath(@__DIR__,"full_ent_spec_plot_0.0.png")

println("Loading data from $data_filename...")

try
    jldopen(data_filename, "r") do file
        
        required_keys = ["entanglement_spectrum_results", "N_values", "σ_val", "max_coeffs_to_store"]
        if !all(haskey(file, k) for k in required_keys)
            println("ERROR: The JLD2 file '$data_filename' is missing one or more required keys.")
            println("It must contain: $required_keys")
            return
        end

        entanglement_spectrum_results = read(file, "entanglement_spectrum_results")
        N_values = read(file, "N_values")
        σ_val = read(file, "σ_val")
        max_coeffs_to_store = read(file, "max_coeffs_to_store")

        println("Data loaded successfully.")
        
        sort!(N_values)

        plot_width_px = 2400  
        plot_height_px = 1000 
        
        p_layout = plot(
            layout = (2, 4),
            size = (plot_width_px, plot_height_px),
            plot_title = "Full Entanglement Spectrum for σ=$(σ_val)",
            plot_titlefontsize = 20,
            legend = false,
            top_margin = 10Plots.mm,
            margin = 10Plots.mm      
        )
        
        x_axis_label = "Schmidt Coefficients"
        y_axis_label = "Coefficient Values"
        y_lims = (0, 0.6)
        x_lims = (0,  50)

        for (i, N) in enumerate(N_values)
            
            if !haskey(entanglement_spectrum_results, N)
                println("  WARNING: No data found for N = $N. Plotting an empty subplot.")
                plot!(p_layout, subplot = i, title = "$N nodes (No Data)", framestyle=:box)
                continue
            end
            
            coeffs = entanglement_spectrum_results[N]
            
            bar!(
                p_layout,
                subplot = i,
                coeffs,
                title = "$N nodes",
                xlabel = x_axis_label,
                ylabel = y_axis_label,
                ylims = y_lims,
                xlims = x_lims,
                seriescolor = :orange,
                linecolor = :darkorange,
                bar_width = 1,
                gap = 0,
            )
        end

        savefig(p_layout, output_filename)
        println("\nPlot saved successfully to $output_filename")

    end
catch e
    println("\nAn error occurred while trying to read the file or generate the plot:")
end