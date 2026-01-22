using JLD2
using FileIO
using Plots
using LinearAlgebra
using Statistics






function sorted_spec(N, σ)

    jldopen(datafile, "r") do file
        group_path_σ = "N=$(N)/sigma=$(σ)"
        
        if !haskey(file, group_path_σ)
            return nothing
        end
        
        all_run_coefs_σ = file[group_path_σ]["schmidt_coefs"]
        
        first_run_σ = all_run_coefs_σ[1]
        if first_run_σ isa Vector{Float64}
            run_coefs_σ = first_run_σ
        else
            run_coefs_σ = first_run_σ.storage.data
        end
        
        spectrum_σ = run_coefs_σ .^ 2
        spectrum_σ ./= sum(spectrum_σ)
        sorted_spec_σ = sort(spectrum_σ, rev=true)

        return sorted_spec_σ
    end
end

function find_tail_divergence(clean_spec, disordered_spec; ratio_threshold=5.0, min_val=1e-20)
    len = min(length(clean_spec), length(disordered_spec))
    
    for i in 1:len
        val_clean = clean_spec[i]
        val_disordered = disordered_spec[i]
        
        if val_clean < min_val
            if val_disordered > min_val * 10
                return i
            else
                continue 
            end
        end
        
        ratio = val_disordered / val_clean
        
        if ratio > ratio_threshold
            return i
        end
    end
    
    return len # Fallback if no deviation found
end

function truncation_error(sorted_spec, dev_idx)
    if dev_idx > length(sorted_spec)
        return 0.0
    end
    truncated_sorted_spec = sorted_spec[dev_idx : end]
    return sum(truncated_sorted_spec)
end

function safe_xlogx(x)
    if x <= 1e-20
        return 0.0
    end
    return x * log(x)
end

function find_body_entropy_contribution(sorted_spec, deviation_idx)
    idx = min(deviation_idx, length(sorted_spec) + 1)
    truncated_sorted_spec = sorted_spec[1 : idx - 1]
    return sum(safe_xlogx(x) for x in truncated_sorted_spec)
end

function find_tail_entropy_contribution(sorted_spec, deviation_idx)
    if deviation_idx > length(sorted_spec)
        return 0.0
    end
    truncated_sorted_spec = sorted_spec[deviation_idx : end]
    return sum(safe_xlogx(x) for x in truncated_sorted_spec)
end

function analyze_spectral_fidelity(spec1::Vector{T}, spec2::Vector{T}) where T <: Number
    len = max(length(spec1), length(spec2))
    p = zeros(T, len)
    q = zeros(T, len)
    
    p[1:length(spec1)] = spec1
    q[1:length(spec2)] = spec2
    
    if sum(p) > 0; p ./= sum(p); end
    if sum(q) > 0; q ./= sum(q); end

    bc_coeff = sum(sqrt.(p .* q))
    classical_fidelity = bc_coeff^2

    trace_dist = 0.5 * sum(abs.(p .- q))
    observable_bound = 2 * trace_dist

    return classical_fidelity, trace_dist, observable_bound
end








function tail_contribution_with_system_size(N_vals, σ_vals)
    results = Vector{Dict{String, Any}}()
    println("Processing data...")

    for N in N_vals
        sorted_spec_0 = sorted_spec(N, 0.00)
        
        if sorted_spec_0 === nothing
            println("Skipping N=$N: No σ=0 data found.")
            continue
        end

        for σ in σ_vals
            # Skip 0.0 if it's in the list
            if σ == 0.0
                continue 
            end

            sorted_spec_σ = sorted_spec(N, σ)

            if sorted_spec_σ === nothing
                println("Skipping N=$N, σ=$σ: No data found.")
                continue
            end

            dev_idx = find_tail_divergence(sorted_spec_0, sorted_spec_σ)
            trunc_err = truncation_error(sorted_spec_σ, dev_idx)
            
            # Calculate other metrics to match original dictionary structure
            body_cnt_σ = find_body_entropy_contribution(sorted_spec_σ, dev_idx)
            tail_cnt_σ = find_tail_entropy_contribution(sorted_spec_σ, dev_idx)
            fidelity, trace_dist, observable_bound = analyze_spectral_fidelity(sorted_spec_0, sorted_spec_σ)

            results_dict = Dict(
                "N" => N,
                "σ" => σ,
                "dev_idx" => dev_idx,
                "truncation_err" => trunc_err,
                "body_contribution" => body_cnt_σ,
                "tail_contribution" => tail_cnt_σ,
                "total_entropy" => body_cnt_σ + tail_cnt_σ,
                "tail_fraction" => tail_cnt_σ / (body_cnt_σ + tail_cnt_σ),
                "fidelity" => fidelity,
                "trace_dist" => trace_dist,
                "observable_bound" => observable_bound
            )

            push!(results, results_dict)
            
        end # σ loop
    end # N loop
    
    println("Processing complete. Found $(length(results)) data points.")
    return results
end







function plot_truncation_err_vs_N(results)
    unique_sigmas = sort(unique([d["σ"] for d in results]))
    
    p = plot(
        title = "Truncation Error vs System Size (N)",
        xlabel = "System Size (N)",
        ylabel = "Tail Entropy (S_tail)",
        legend = :bottomleft,
        grid = true,
        bg_legend = :transparent,
        yscale = :log,
        yaxis=:log
    )
    
    for σ in unique_sigmas
        sigma_data = filter(d -> d["σ"] == σ, results)
    
        sort!(sigma_data, by = d -> d["N"])
        
        x_vals = [d["N"] for d in sigma_data]
        y_vals = [d["truncation_err"] for d in sigma_data]
        
        plot!(p, x_vals, y_vals, 
            label = "σ = $σ", 
            marker = :circle, 
            linewidth = 2
        )
    end
    
    return p
end


function main()
    global datafile = joinpath(@__DIR__, "mach_prec_data.jld2")

    # N_vals = vcat(10:2:40, 80, 100, 120) 
    # σ_vals = [0.001, 0.002, 0.005, 0.1, 0.3, 0.5]

    N_vals = vcat(14:2:28) 
    σ_vals = [0.001, 0.002, 0.005]

    results = tail_contribution_with_system_size(N_vals, σ_vals)

    plt = plot_truncation_err_vs_N(results)

    savefig(plt, joinpath(@__DIR__, "truncation_vs_N_small.png"))


end

main()


