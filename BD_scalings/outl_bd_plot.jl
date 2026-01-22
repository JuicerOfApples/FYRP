using JLD2, Plots

function create_comparison_plot()
    
    connected_data_file = joinpath(@__DIR__, "full_bd_data.jld2")
    outlier_data_file = joinpath(@__DIR__, "outl_bd_data.jld2")
    
    plot_filename = joinpath(@__DIR__, "outl_full_bd_plot.png")

    if !isfile(connected_data_file)
        println("ERROR: Connected data file not found: $connected_data_file")
        return
    end
    if !isfile(outlier_data_file)
         println("ERROR: Outlier data file not found: $outlier_data_file")
        return
    end

    local results_connected, N_range_connected, sigma_values_connected
    local results_outlier, N_range_outlier, sigma_values_outlier

    try
        jldopen(connected_data_file, "r") do file
            results_connected = read(file, "results")
            N_range_connected = read(file, "N_range")
            sigma_values_connected = read(file, "sigma_values")
        end
        println("Loaded connected data from $connected_data_file")
    catch e
        println("ERROR: Could not load data from $connected_data_file. Error: $e")
        return
    end

    try
        jldopen(outlier_data_file, "r") do file 
            results_outlier = read(file, "results") 
            N_range_outlier = read(file, "N_range")
            sigma_values_outlier = read(file, "sigma_values")
        end
        println("Loaded outlier data from $outlier_data_file")
    catch e
        println("ERROR: Could not load data from $outlier_data_file. Error: $e")
        return
    end

    if sigma_values_connected != sigma_values_outlier
        println("WARNING: Sigma values do not match between files.")
        println("Connected Sigmas: $sigma_values_connected")
        println("Outlier Sigmas: $sigma_values_outlier")
        println("Plotting may be misleading. Using connected sigma values for loop.")
    end
    
    sigma_values = sigma_values_connected

    gr()
    
    p = plot(
        title = "Bond Dimension Comparison: Connected vs. Outlier Models", 
        xlabel = "System Size (N)",
        ylabel = "Average Max Bond Dimension",
        legend = :topleft,
        size = (1000, 600)
    )

    colors = [:blue, :purple, :red, :green] 
    
    for (idx, σ) in enumerate(sigma_values)
        color = colors[idx]
        
        if haskey(results_connected, σ)
             plot!(p,
                N_range_connected,
                results_connected[σ].avg,
                ribbon = results_connected[σ].err,
                label = "Connected (σ=$σ)",
                color = color,
                linestyle = :solid,
                fillalpha = 0.15 
            )
        end
        
        if haskey(results_outlier, σ)
            plot!(p,
                 N_range_outlier,
                results_outlier[σ].avg,
                ribbon = results_outlier[σ].err,
                label = "Outlier (σ=$σ)",
                color = color,
                linestyle = :dash,
                 fillalpha = 0.15 
            )
        end
    end

    try
        savefig(p, plot_filename)
        println("Plot saved successfully to $plot_filename") 
    catch e
        println("ERROR: Could not save plot. Error: $e") 
    end
end

create_comparison_plot()