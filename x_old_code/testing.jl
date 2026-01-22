using Random, Markdown
using JLD2 

function main()

    σ  = 0.002
    weight = 1 + σ * randn() 

    filename = "testing.jld2"
    jldsave(filename; weight)
end

main()