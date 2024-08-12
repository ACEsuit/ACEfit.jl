using ACEfit
using LinearAlgebra
using MLJ
using MLJScikitLearnInterface

@info("Test MLJ interface on overdetermined system")
Nobs = 10_000
Nfeat = 100
A = randn(Nobs, Nfeat) / sqrt(Nobs)
y = randn(Nobs)
P = Diagonal(1.0 .+ rand(Nfeat))


@info(" ... MLJLinearModels LinearRegressor")
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
solver = LinearRegressor()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)


@info(" ... MLJLinearModels LassoRegressor")
LassoRegressor = @load LassoRegressor pkg=MLJLinearModels
solver = LassoRegressor()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)


@info(" ... MLJ SKLearn ARD")
ARDRegressor = @load ARDRegressor pkg=MLJScikitLearnInterface
solver = ARDRegressor(
    max_iter = 300,
    tol = 1e-3,
    threshold_lambda = 10000
)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
