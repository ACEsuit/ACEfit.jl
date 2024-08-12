module ACEfit_MLJScikitLearnInterface_ext

using ACEfit
using MLJ
using MLJScikitLearnInterface
using PythonCall


"""
    ACEfit.solve(solver, A, y)

Overloads `ACEfit.solve` to use scikitlearn solvers from MLJ.

# Example
```julia
using MLJ
using ACEfit

# Load ARD solver
ARDRegressor = @load ARDRegressor pkg=MLJScikitLearnInterface

# Create the solver itself and give it parameters
solver = ARDRegressor(
    max_iter = 300,
    tol = 1e-3,
    threshold_lambda = 10000
    # more params
)

# fit ACE model
linear_fit(training_data, basis, solver)

# or lower level
ACEfit.fit(solver, A, y)
```
"""
function  ACEfit.solve(solver, A, y)
    Atable = MLJ.table(A)
    mach = machine(solver, Atable, y)
    MLJ.fit!(mach)
    params = fitted_params(mach)
    c = params.coef
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

end
