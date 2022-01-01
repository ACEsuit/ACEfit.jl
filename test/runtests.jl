using ACEfit
using Test

@testset "ACEfit.jl" begin
    
    @testset "Iterate" begin include("test_iterate.jl"); end 
    
end
