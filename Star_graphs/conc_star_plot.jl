using JLD2
using Plots
using Statistics


data_filename = joinpath(@__DIR__, "conc_star_data_0.002.jld2")
plot_filename = joinpath(@__DIR__, "conc_star_plot_0.002.png")

# Check if data file exists
if !isfile(data_filename)
    println("ERROR: Data file not found: $data_filename")
else
    println("Loading data from $data_filename")
    
    # Load the data
    file = jldopen(data_filename, "r")
    results = read(file, "results")
    N_range = read(file, "N_range")
    sigma_values = read(file, "sigma_values")
    close(file)
    
    
    if isempty(sigma_values)
        error("No sigma values found in data file.")
    end
    
    σ = sigma_values[1] 
    data_for_sigma = results[σ] 
    

    all_plots = Plots.Plot[]

    for N in N_range
        if haskey(data_for_sigma, N)
            data_N = data_for_sigma[N]
            concurrence_vector = data_N.avg 
            std_dev_vector = data_N.err
            
            if !isempty(concurrence_vector)
                num_pairs = length(concurrence_vector)
                pair_labels = ["(1, $(j+1))" for j in 1:num_pairs]


                max_y_value = maximum(concurrence_vector .+ std_dev_vector)
                
                upper_limit = max_y_value + 0.2

                # Create the bar chart
                plt = bar( 
                    pair_labels,
                    concurrence_vector,
                    # yerror = std_dev_vector,
                    ylims = (0.0, upper_limit), 
                    label = "Pair Concurrence",
                    title = "Concurrence Spread for N=$N", 
                    xlabel = "Pair (Center, Outer)",
                    ylabel = "Concurrence C(1,j)",
                    legend = :topleft,
                    xrotation = 60,
                    bottom_margin = 20Plots.mm,
                    left_margin = 10Plots.mm,  
                    tickfont = 8 
                )
                
                # Calculate the average concurrence for this N
                avg_C = mean(concurrence_vector)

                hline!(
                    [avg_C],
                    label = "Equal Concurrence",
                    color = :red,
                    linestyle = :dash,
                    linewidth = 2
                )
                
                push!(all_plots, plt)
            end
        else
            println("No data found for N=$N. Skipping.")
        end
    end

    if isempty(all_plots)
        println("No plots were generated. Exiting.")
    else
        println("Combining $(length(all_plots)) plots into one file")
        

        final_layout = (2, 3) 
        
        combined_plot = plot(
            all_plots..., 
            layout = final_layout,
            size = (1800, 1200),
            plot_title = "Star Graph Concurrence of Centre to Outer Nodes, (σ=$σ)"
        )
        
        savefig(combined_plot, plot_filename)
        println("saved successfully to $plot_filename")
    end
end