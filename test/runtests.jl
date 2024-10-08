using ACEfit
using Test

@testset "ACEfit.jl" begin
    @testset "Iterate" begin include("test_iterate.jl") end

    @testset "Bayesian Linear" begin include("test_bayesianlinear.jl") end

    @testset "Linear Solvers" begin include("test_linearsolvers.jl") end

    @testset "ASP" begin include("test_asp.jl") end

    @testset "MLJ Solvers" begin include("test_mlj.jl") end
end
