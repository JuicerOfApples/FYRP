using JLD2
using Plots
using LinearAlgebra

gr()

filename = joinpath(@__DIR__, "full_bd_data.jld2")
output_filename = joinpath(@__DIR__, "log_sqrt_scaling_plot.png")

function perform_fitting_and_plot(data_file, output_file)
    println("Loading data from $data_file...")
    if !isfile(data_file)
        println("Error: Data file '$data_file' not found.")
        return
    end
    
    data = load(data_file)
    results_data = data["results"]
    N_values = collect(data["N_range"])
    
    σ_target = 0.0
    if !haskey(results_data, σ_target)
        println("Error: Sigma = 0.0 not found in results.")
        return
    end


    x_data = Float64.(N_values)
    y_data = results_data[σ_target].avg

    # Subsample only the even indices (Julia is 1-based, so indices 2,4,6,...)
    min_len = min(length(x_data), length(y_data))
    even_inds = 2:2:min_len
    if isempty(even_inds)
        println("No even indices available after subsampling; aborting.")
        return
    end
    x_data = x_data[even_inds]
    y_data = y_data[even_inds]

    # Logarithmic Fit: y = a * ln(x) + b
    A_log = [log.(x_data) ones(length(x_data))]
    coeffs_log = A_log \ y_data
    a_log, b_log = coeffs_log[1], coeffs_log[2]
    
    # Square Root Fit: y = c * sqrt(x) + d
    A_sqrt = [sqrt.(x_data) ones(length(x_data))]
    coeffs_sqrt = A_sqrt \ y_data
    c_sqrt, d_sqrt = coeffs_sqrt[1], coeffs_sqrt[2]

    println("Fits calculated:")
    println("  Logarithmic: y ≈ $(round(a_log, digits=3)) ln(N) + $(round(b_log, digits=3))")
    println("  Square Root: y ≈ $(round(c_sqrt, digits=3)) √N + $(round(d_sqrt, digits=3))")

    x_smooth = range(minimum(x_data), maximum(x_data), length=200)
    y_smooth_log = a_log .* log.(x_smooth) .+ b_log
    y_smooth_sqrt = c_sqrt .* sqrt.(x_smooth) .+ d_sqrt

    plt = plot(
        title="Scaling of Bond Dimension (σ=0.0)",
        xlabel="System Size (N)",
        ylabel="Average Max Bond Dimension",
        legend=:bottomright,
        framestyle=:box,
        size=(800, 500),
        dpi=300
    )

    plot!(plt, x_data, y_data,
        label="Scaling for σ=0.0",
        color=:black,
        linewidth = 4
    )

    plot!(plt, x_smooth, y_smooth_log,
        label="Log Fit (~ ln N)",
        color=:blue,
        linewidth=2
    )

    plot!(plt, x_smooth, y_smooth_sqrt,
        label="Sqrt Fit (~ √N)",
        color=:red,
        linewidth=2
    )

    savefig(plt, output_file)
    println("Plot saved successfully to $output_file")
end

perform_fitting_and_plot(filename, output_filename)