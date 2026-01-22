using JLD2, FileIO
using Statistics, Printf

# Function to find where two spectra diverge
function find_divergence_index(s_clean::Vector{Float64}, s_disorder::Vector{Float64}; tol=1e-4)
    # Compare only up to the length of the shorter vector
    len = min(length(s_clean), length(s_disorder))
    
    for i in 1:len
        # Calculate relative difference
        diff = abs(s_clean[i] - s_disorder[i])
        avg = (s_clean[i] + s_disorder[i]) / 2.0
        
        # Avoid division by zero f or very small numbers
        if avg < 1e-14
            if diff > 1e-14 
                return i 
            end
            continue
        end
        
        rel_diff = diff / avg
        
        # If relative difference exceeds tolerance, we found the split
        if rel_diff > tol
            return i
        end
    end
    return len # If no divergence found, return the end
end

file_clean = joinpath(@__DIR__, "high_prec_spec_data_0.0.jld2")
file_disorder = joinpath(@__DIR__, "high_prec_spec_data_0.002.jld2")



data_clean = load(file_clean)["entanglement_spectrum_results"]
data_disorder = load(file_disorder)["entanglement_spectrum_results"]


common_keys = intersect(keys(data_clean), keys(data_disorder))
N_values = sort(collect(common_keys))

analysis_results = Dict{String, Any}()
output_file_path = joinpath(@__DIR__, "tail_analysis_data.jld2")

results_storage = Dict(
    "N" => Int[],
    "divergence_index" => Int[],
    "tail_weight_disorder" => Float64[],
    "tail_weight_clean" => Float64[]
)

println("Comparative Analysis: Divergence of σ=0.0 vs σ=0.002")
println("N   | Div Index | Tail Wgt (Disorder) | Tail Wgt (Clean)")
println("----|-----------|---------------------|-----------------")

for N in N_values
    # Get coefficients
    s_clean = sort(data_clean[N], rev=true)
    s_disorder = sort(data_disorder[N], rev=true)
    
    # Find Divergence Index
    # We use a tolerance of 1% (0.01) relative difference to define "divergence"
    idx = find_divergence_index(s_clean, s_disorder, tol=0.01)
    
    # Calculate Tail Weight based on this dynamic index
    # The tail is everything FROM this index onwards
    
    # Safety check: ensure index is within bounds
    idx_clean = min(idx, length(s_clean))
    idx_disorder = min(idx, length(s_disorder))
    
    tail_clean = s_clean[idx_clean:end]
    tail_disorder = s_disorder[idx_disorder:end]
    
    wgt_clean = sum(tail_clean .^ 2)
    wgt_disorder = sum(tail_disorder .^ 2)
    
    @printf("%-3d | %-9d | %-19.2e | %-15.2e\n", N, idx, wgt_disorder, wgt_clean)
    
    push!(results_storage["N"], N)
    push!(results_storage["divergence_index"], idx)
    push!(results_storage["tail_weight_disorder"], wgt_disorder)
    push!(results_storage["tail_weight_clean"], wgt_clean)
end

save(output_file_path, Dict(
    "0.002" => Dict(
        "N" => results_storage["N"], 
        "tail_weight" => results_storage["tail_weight_disorder"]
    ),
    "0.0" => Dict( 
        "N" => results_storage["N"], 
        "tail_weight" => results_storage["tail_weight_clean"]
    )
))

println("\nAnalysis complete. Saved to $output_file_path")