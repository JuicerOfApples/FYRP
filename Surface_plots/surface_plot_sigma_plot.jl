using JLD2, Plots, Colors
using Measures 

gr() 

filename = joinpath(@__DIR__, "surface_plot_sigma_data(-1.0)(-1.0).jld2")

function load_data_and_plot(data_file)
    println("Loading data from $data_file...")
    if !isfile(data_file)
        println("Error: Data file '$data_file' not found.")
        return 
    end
    
    data = load(data_file)
    avg_bond_dims = data["avg_bond_dims"]
    N_range = data["N_range"]
    sigma_range = data["sigma_range"]

    N_values = collect(N_range)
    sigma_values = collect(sigma_range)

    N_slice = 1:91
    
    plt = plot(N_values[N_slice], sigma_values, avg_bond_dims'[:, N_slice],
        st = :surface,
        
        title = "Average Bond Dimension against \nSystem Size and Disorder",
        xlabel = "System Size (N)",   
        ylabel = "\n       Disorder (σ)",      
        zlabel = "\nAvg Max Bond Dim",  
        
        camera = (25, 35),
        
        c = cgrad(:inferno),
        legend = false,
        lw = 0.1,             
        fillalpha = 1.0,     
        
        tickfontsize = 14,    
        guidefontsize = 16,   
        titlefontsize = 22,
        margin = 3mm,       
        
        size = (1000, 800),   
        dpi = 300        
    )
     
    output_filename = joinpath(@__DIR__, "surface_plot_sigma_plot(-1.0)(-1.0).png")
    savefig(plt, output_filename)
    println("Plot saved successfully to $output_filename")
end

load_data_and_plot(filename)