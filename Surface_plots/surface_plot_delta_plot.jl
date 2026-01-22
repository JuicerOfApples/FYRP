using JLD2, Plots, Colors


const J_val = -1.0 

filename = joinpath(@__DIR__, "surface_plot_delta_data(-1.0).jld2")

function load_data_and_plot(data_file)
    println("Loading data from $data_file...")
    if !isfile(data_file)
        println("Error: Data file '$data_file' not found.")
        return 
    end
    
    data = load(data_file)
    
    avg_bond_dims = data["avg_bond_dims"]
    N_range = data["N_range"]
    delta_range = data["delta_range"]
    sigma_values = data["sigma_values"]

    N_values = collect(N_range)
    delta_values = collect(delta_range) 

    plotlyjs() 
    

    for (idx, sigma) in enumerate(sigma_values)
        println("  Plotting for σ = $sigma")

        data_slice = avg_bond_dims[:, :, idx]
        
        plot_title = "Max Bond Dimension χ vs. (N, Δ) (σ = $sigma)"
         
        plt = plot(N_values, -delta_values, data_slice', # Transpose is needed
            st=:surface,
            title=plot_title,
            xlabel="System Size N",
            ylabel="Anisotropy Δ",  
            zlabel="Max Bond Dimension χ",
            camera=(-25, 25),      
            c=cgrad(:inferno),    
            legend=false,
            automargin = true,
            autosize = true,
        )
        
        output_filename = joinpath(@__DIR__, "surface_plot_delta_plot(-1.0)($sigma).png")
        
        try
            savefig(plt, output_filename)
            println("  Plot saved successfully to $output_filename")
        catch e
            println("  ERROR saving plot: $e")
        end
    end
end

load_data_and_plot(filename)
