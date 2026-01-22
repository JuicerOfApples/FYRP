using JLD2, Plots

function create_comparison_plot()
    
    connected_data_file = joinpath(@__DIR__, "full_bd_data.jld2")
    linear_data_file = joinpath(@__DIR__, "lin_bd_data.jld2")
    
    plot_filename = joinpath(@__DIR__, "lin_full_bd_plot.png")

    if !isfile(connected_data_file)
        println("ERROR: Connected data file not found: $connected_data_file")
        return
    end
    if !isfile(linear_data_file)
        println("ERROR: Linear data file not found: $linear_data_file")
        return
    end

    local results_connected, N_range_connected, sigma_values_connected
    local results_linear, N_range_linear, sigma_values_linear

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
        jldopen(linear_data_file, "r") do file 
            results_linear = read(file, "results") 
            N_range_linear = read(file, "N_range")
            sigma_values_linear = read(file, "sigma_values")
        end
        println("Loaded linear data from $linear_data_file")
    catch e
        println("ERROR: Could not load data from $linear_data_file. Error: $e")
        return
    end

    if sigma_values_connected != sigma_values_linear
        println("WARNING: Sigma values do not match between files.")
        println("Connected Sigmas: $sigma_values_connected")
        println("Linear Sigmas: $sigma_values_linear")
        println("Plotting may be misleading. Using connected sigma values for loop.")
    end
    
    sigma_values = sigma_values_connected

    gr()
    
    p = plot(
        title = "Bond Dimension Comparison: Connected vs. Linear Models", 
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
        
        if haskey(results_linear, σ)
            plot!(p,
                N_range_linear,
                results_linear[σ].avg,
                ribbon = results_linear[σ].err,
                label = "Linear (σ=$σ)",
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