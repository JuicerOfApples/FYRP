using ITensors
using ITensorMPS
using JLD2
using LinearAlgebra
using Printf

function compress_noise_vs_clean()
    input_file = joinpath(@__DIR__, "noise_vs_clean_data.jld2")
    output_file = joinpath(@__DIR__, "noise_vs_clean_data_analysis.jld2")
    
    if !isfile(input_file)
        println("Error: Input file '$input_file' not found.")
        return
    end

    println("Loading raw simulation data...")
    # Using IOStream to avoid file locking issues
    data = jldopen(input_file, "r"; iotype=IOStream)
    
    if !haskey(data, "results_clean") || !haskey(data, "results_noise")
        println("Error: Raw data file is missing 'results_clean' or 'results_noise'.")
        close(data)
        return
    end

    results_clean = read(data, "results_clean")
    results_noise = read(data, "results_noise")
    close(data)

    # Dictionary to store lightweight analysis results
    analysis_db = Dict{Int, Dict{Symbol, Any}}()
    
    println("\nExtracting Analysis Data (Spectra, Renyi, Entropy, Fidelity)...")
    println("-"^100)
    @printf("%-4s | %-10s | %-12s | %-12s | %-15s | %-10s\n", 
            "N", "Fidelity", "SvN(Cln)", "SvN(Ns)", "Chi(Cln/Ns)", "Status")
    println("-"^100)

    all_Ns = sort(collect(keys(results_clean)))

    for N in all_Ns
        if !haskey(results_noise, N); continue; end

        psi_clean = results_clean[N]
        psi_noise = results_noise[N]

        # ---------------------------------------------------------
        # 1. Bond Dimension
        # ---------------------------------------------------------
        chi_clean = maxlinkdim(psi_clean)
        chi_noise = maxlinkdim(psi_noise)

        # ---------------------------------------------------------
        # 2. Fidelity
        # ---------------------------------------------------------
        sites_clean = siteinds(psi_clean)
        replace_siteinds!(psi_noise, sites_clean)
        fidelity = abs(inner(psi_clean, psi_noise))^2

        # ---------------------------------------------------------
        # 3. Entanglement (Von Neumann, Renyi, Spectrum)
        # ---------------------------------------------------------
        function analyze_entanglement(psi, bond; alpha=2.0)
            psi_c = copy(psi)
            orthogonalize!(psi_c, bond)
            
            # SVD to get singular values (Schmidt coefficients)
            _, S, _ = svd(psi_c[bond], uniqueinds(psi_c[bond], psi_c[bond+1]))
            
            # Extract standard Vector{Float64} of singular values
            # S is a diagonal tensor, so we extract diagonal elements
            sv = [S[n, n] for n in 1:dim(S, 1)]
            
            # Normalize probabilities (p = s^2)
            p = sv.^2
            p = p ./ sum(p)
            
            # Filter small values to avoid NaN in log
            p_nz = p[p .> 1e-16]

            # Von Neumann Entropy: - sum(p * log(p))
            SvN = -sum(p_nz .* log.(p_nz))

            # Renyi Entropy: 1/(1-alpha) * log(sum(p^alpha))
            S_renyi = (1.0 / (1.0 - alpha)) * log(sum(p_nz.^alpha))
            
            return SvN, S_renyi, sv
        end
        
        # Calculate at half-chain cut
        mid_b = N ÷ 2
        SvN_c, Renyi_c, spec_c = analyze_entanglement(psi_clean, mid_b; alpha=2.0)
        SvN_n, Renyi_n, spec_n = analyze_entanglement(psi_noise, mid_b; alpha=2.0)

        # ---------------------------------------------------------
        # 4. Spin Correlations
        # ---------------------------------------------------------
        C_clean = Matrix{Float64}(correlation_matrix(psi_clean, "Sz", "Sz"))
        C_noise = Matrix{Float64}(correlation_matrix(psi_noise, "Sz", "Sz"))

        # ---------------------------------------------------------
        # 5. Store Stats
        # ---------------------------------------------------------
        analysis_db[N] = Dict(
            :fidelity       => fidelity,
            :S_clean        => SvN_c,
            :S_noise        => SvN_n,
            :renyi_clean    => Renyi_c,      # New: Renyi entropy
            :renyi_noise    => Renyi_n,      # New: Renyi entropy
            :spectrum_clean => spec_c,       # New: Full singular value spectrum
            :spectrum_noise => spec_n,       # New: Full singular value spectrum
            :chi_clean      => chi_clean,
            :chi_noise      => chi_noise,
            :C_clean        => C_clean,
            :C_noise        => C_noise
        )

        chi_str = "$chi_clean / $chi_noise"
        @printf("%-4d | %-10.6f | %-12.4f | %-12.4f | %-15s | %-10s\n", 
                N, fidelity, SvN_c, SvN_n, chi_str, "Done")
    end

    println("-"^100)
    println("Saving detailed analysis file to $output_file...")

    if isfile(output_file); rm(output_file; force=true); end
    jldsave(output_file; iotype=IOStream, analysis_results=analysis_db)
    
    println("Done.")
end

compress_noise_vs_clean()