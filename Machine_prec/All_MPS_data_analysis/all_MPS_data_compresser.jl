using ITensors
using ITensorMPS
using JLD2
using LinearAlgebra
using Printf

function compress_mps_data()
    input_file = joinpath(@__DIR__,"all_MPS_data.jld2")
    output_file = joinpath(@__DIR__,"all_MPS_data_compressed.jld2")
    
    if !isfile(input_file)
        println("Error: $input_file not found.")
        return
    end
    
    println("Loading original results from $input_file...")
    data_db = load(input_file, "results_db")
    
    # 1. Cache Clean Baselines for calculations
    println("Caching Clean System baselines...")
    clean_baselines = Dict()
    for (key, data) in data_db
        N, σ = key
        if σ == 0.0
            clean_baselines[N] = data["mps"]
        end
    end
    
    compressed_db = Dict()
    println("\nCompressing and Extracting Features...")
    println("Strategy: Keep Clean MPS (Key), Discard Noisy MPS (Bloat)")
    println("-"^70)
    @printf("%-4s | %-8s | %-10s | %-25s\n", "N", "Sigma", "BD", "Action")
    println("-"^70)

    for (key, data) in data_db
        N, σ = key
        
        # --- Feature Extraction ---
        spectrum = data["spectrum"]
        
        if haskey(data, "mps")
            psi = data["mps"]
            chi = maxlinkdim(psi)
            
            # Compute Truncated Fidelity (Heavy calculation done NOW so we can drop the noisy MPS)
            fidelity_truncated = 1.0
            
            if σ > 0.0 && haskey(clean_baselines, N)
                psi_clean = clean_baselines[N]
                chi_clean = maxlinkdim(psi_clean)
                
                psi_test = copy(psi)
                truncate!(psi_test; maxdim=chi_clean, cutoff=1E-14)
                ov = abs(inner(psi_clean, psi_test))
                fidelity_truncated = ov^2
            end
            
            # Compute Entropy
            S_vn = 0.0
            for p in spectrum
                if p > 1E-20
                    S_vn -= p * log(p)
                end
            end
        else
            # Handle cases where data might already be partial
            chi = get(data, "maxlinkdim", 0)
            fidelity_truncated = get(data, "fidelity", 0.0)
            S_vn = get(data, "entropy", 0.0)
        end
        
        # --- Build Compressed Entry ---
        entry = Dict(
            "spectrum" => spectrum,
            "maxlinkdim" => chi,
            "fidelity" => fidelity_truncated,
            "entropy" => S_vn,
            "sigma" => σ,
            "N" => N
        )
        
        # --- CRITICAL CHANGE: KEEP KEY MPS ---
        # If this is a Clean System (Sigma=0), we KEEP the full MPS object.
        # This allows future recalculations or new analysis using the baseline.
        action_msg = ""
        if σ == 0.0
            entry["mps"] = data["mps"]
            action_msg = "KEPT (Reference)"
        else
            # For noisy systems, we drop the MPS to save space
            action_msg = "DROPPED (Stats Extracted)"
        end
        
        compressed_db[key] = entry
        
        @printf("%-4d | %-8.3f | %-10d | %-25s\n", N, σ, chi, action_msg)
    end
    
    println("-"^70)
    println("Saving compressed database to $output_file...")
    jldsave(output_file; results_db=compressed_db)
    
    # Calculate stats
    s_in = filesize(input_file) / 1024^2
    s_out = filesize(output_file) / 1024^2
    println("\nCompression Complete!")
    @printf("Original Size:   %.2f MB\n", s_in)
    @printf("Compressed Size: %.2f MB\n", s_out)
    @printf("Reduction:       %.1f%%\n", 100 * (1 - s_out/s_in))
end

compress_mps_data()