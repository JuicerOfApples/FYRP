using Plots, JLD2
"""
Plots the average concurrence vs. N from the simulation data.
"""
function plot_concurrence_results(filename::String)
    println("Loading data from $filename...")
    
    # Check if the file exists
    if !isfile(filename)
        println("ERROR: Data file '$filename' not found.")
        println("Please run 'star_conc.jl' first to generate the data.")
        return
    end
    
    # Load the results
    data = jldopen(filename, "r")
    results = read(data, "results")
    N_range = read(data, "N_range")
    sigma_values = read(data, "sigma_values")
    close(data)

    
    p1 = plot(
        title = "Star Graph Concurrence vs. System Size",
        xlabel = "System Size (N)",
        ylabel = "Concurrence C(ρ₁₂)",
        legend = :topright,
        # xaxis = :log10, # Log scale for N
        # yaxis = :log10, # Log scale for Concurrence
        minorgrid = true
    )
    
    sorted_sigmas = sort(sigma_values)
    
    for σ in sorted_sigmas
        avg_concurrence = results[σ].avg
        error_std = results[σ].err 
        
        plot!(
            p1,
            N_range, 
            avg_concurrence, 
            ribbon = error_std,  
            label = "σ = $σ", 
            marker = :circle,
            markersize = 4
        )
    end 
    
    
    # Save the plot
    output_filename = joinpath(@__DIR__, "star_conc_plot.png")
    savefig(p1, output_filename)
    println("Saved concurrence plot to $output_filename")
end


data_filename = joinpath(@__DIR__, "star_conc_data.jld2")

plot_concurrence_results(data_filename)

println("Plotting script finished.")