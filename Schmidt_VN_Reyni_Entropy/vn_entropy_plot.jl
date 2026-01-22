using JLD2, Plots, Colors

function plot_entropy_linear(N_range, results)
    plotlyjs() 
    N_values = collect(N_range)

    plt = plot(title="Average Von Neumann Entropy",
               xlabel="Number of Nodes (N)",
               ylabel="Average Entropy",
               legend=:topleft,
               grid=true)

    colors = Dict(0.0 => :purple, 0.002 => :darkorange)
    sigmas_sorted = sort(collect(keys(results)))

    for σ in sigmas_sorted
        label_str = "σ=$σ"
        color = get(colors, σ, :auto)
        entropy_values = results[σ]

        if length(entropy_values) != length(N_values)
            println("Warning (Linear Plot): Mismatch in lengths for σ=$σ.")
        end


        plot!(plt, N_values, entropy_values, 
              label=label_str, 
              color=color, 
              linewidth=1.5,
              marker=:circle,  
              markersize=4)    
    end
    return plt
end


function plot_entropy_logscale(N_range, results)
    plotlyjs() 
    N_values_full = collect(N_range)
    N_log_values = log2.(N_values_full)

    plt = plot(title="Average Von Neumann Entropy (Log Scale)",
               xlabel="Number of Nodes (log₂ scale)",
               ylabel="Average Entropy",
               legend=:topleft,
               grid=true)

    colors = Dict(0.0 => :purple, 0.002 => :darkorange)
    sigmas_sorted = sort(collect(keys(results)))

    for σ in sigmas_sorted
        label_str = "σ=$σ"
        color = get(colors, σ, :auto)
        entropy_values = results[σ]

        if length(entropy_values) != length(N_log_values)
            println("Warning (Log Plot): Mismatch in lengths for σ=$σ.")
        end

        plot!(plt, N_log_values, entropy_values, 
              label=label_str, 
              color=color, 
              linewidth=1.5,
              marker=:circle, 
              markersize=4)   
    end
    
    xticks_vals = [log2(v) for v in [16, 32, 64]] 
    xtick_labels = ["2^$(Int(log2(v)))" for v in [16, 32, 64]]
    valid_ticks = [(t, l) for (t, l) in zip(xticks_vals, xtick_labels) if minimum(N_log_values) <= t <= maximum(N_log_values)]
    if !isempty(valid_ticks)
        plot!(plt, xticks=(first.(valid_ticks), last.(valid_ticks)))
    end

    return plt
end


function load_and_generate_plots()
    filename = joinpath(@__DIR__, "vn_entropy_data(-1.0)(-1.0).jld2")
    println("Loading data from $filename...")

    if !isfile(filename)
        println("Error: Data file '$filename' not found.")
        return
    end
    
    data = load(filename)
    entropy_results = data["entropy_results"]
    N_range_used = data["N_range_used"]

    println("Generating linear plot")
    plt_linear = plot_entropy_linear(N_range_used, entropy_results)
    output_linear = joinpath(@__DIR__, "vn_entropy_plot_lin(-1.0)(-1.0).png")
    savefig(plt_linear, output_linear)
    println("Saved linear plot to $output_linear")

    println("Generating log scale plot")
    plt_log = plot_entropy_logscale(N_range_used, entropy_results)
    output_log = joinpath(@__DIR__, "vn_entropy_plot_log(-1.0)(-1.0).png")
    savefig(plt_log, output_log)
    println("Saved log scale plot to $output_log")
end

load_and_generate_plots()
