using JLD2, Plots


gr()

filename = joinpath(@__DIR__, "avg_err_bd_data(-1.0)(-1.0).jld2")






function load_data_and_plot(data_file)
    println("Loading data from $data_file...")
    if !isfile(data_file)
        println("Error: Data file '$data_file' not found.")
        return
    end
    
    data = load(data_file)
    results_data = data["results"]
    N_range_data = data["N_range"]

    N_values = collect(N_range_data)
    sigma_values = [0.0, 0.001, 0.002]



    plot_attrs = Dict(
        0.0 => (
            color = :gold,
            shape = :circle
        ),
        0.001 => (
            color = :darkviolet,
            shape = :square
        ),
        0.002 => (
            color = :firebrick,
            shape = :diamond
        )
    )

    plt = plot(
        title="Average Bond Dimension against System Size",
        xlabel="System Size (N)",
        ylabel="Avgerage Max Bond Dimension",
        legend=:bottomright,     
        framestyle=:box,     
        size=(800, 500),    
        dpi=300             
    )
    
    N_slice = 1:91

    for σ in sigma_values
        
        attrs = plot_attrs[σ]

        plot!(plt, N_values[N_slice], results_data[σ].avg[N_slice],
            ribbon=results_data[σ].err[N_slice],
            fillalpha=0.2,                   
            
            label="σ = $σ",
            lw=2,                              
            marker=attrs.shape,               
            markersize=4,                       
            color=attrs.color
        )
    end
    
    output_filename = joinpath(@__DIR__, "avg_err_bd_plot(-1.0)(-1.0).png")
    savefig(plt, output_filename)
    println("Plot saved successfully to $output_filename")

    println(maximum(results_data[0.002]))

end



load_data_and_plot(filename)