
using JLD2 

function load_data_and_plot(filename)
    println("Loading data from $filename...")
    if !isfile(filename)
        println("Error: Data file '$filename' not found")
        return nothing
    end
    
    # Load the data from the JLD2 file
    data = load(filename)
    testdataforfile = data["testdataforfile"]
    
    println(testdataforfile)
    
    
end


testdataforfile = [0,1,2,3,4,5,6,7,8,9,10]

filename = "testfilefordata.jld2"
jldsave(filename; testdataforfile)
println("Data saved successfully.\n")

load_data_and_plot(filename)