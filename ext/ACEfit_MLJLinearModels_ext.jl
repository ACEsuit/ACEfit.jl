module ACEfit_MLJLinearModels_ext

using MLJ
using ACEfit
using MLJLinearModels

"""
    ACEfit.solve(solver, A, y)

Overloads `ACEfit.solve` to use MLJLinearModels solvers,
when `solver` is [MLJLinearModels](https://github.com/JuliaAI/MLJLinearModels.jl) solver.

# Example
```julia
using MLJ
using ACEfit

# Load Lasso solver
LassoRegressor = @load LassoRegressor pkg=MLJLinearModels

# Create the solver itself and give it parameters
solver = LassoRegressor(
    lambda = 0.2,
    fit_intercept = false
    # insert more fit params
)

# fit ACE model
linear_fit(training_data, basis, solver)

# or lower level
ACEfit.fit(solver, A, y)
```
"""
function  ACEfit.solve(solver::Union{
            MLJLinearModels.ElasticNetRegressor,
            MLJLinearModels.HuberRegressor,
            MLJLinearModels.LADRegressor,
            MLJLinearModels.LassoRegressor,
            MLJLinearModels.LinearRegressor,
            MLJLinearModels.QuantileRegressor,
            MLJLinearModels.RidgeRegressor,
            MLJLinearModels.RobustRegressor,
        },
        A, y)
    Atable = MLJ.table(A)
    mach = machine(solver, Atable, y)
    MLJ.fit!(mach)
    params = fitted_params(mach)
    return Dict{String, Any}("C" => map( x->x.second,  params.coefs) )
end

end