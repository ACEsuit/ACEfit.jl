```@meta
CurrentModule = ACEfit
```

# ACEfit

Documentation for [ACEfit](https://github.com/ACEsuit/ACEfit.jl).

## Scikit-learn solvers

To use Python based Scikit-learn solvers you need to load PythonCall in addition to ACEfit.

```julia
using ACEfit
using PythonCall
```

## MLJ solvers

To use [MLJ](https://github.com/alan-turing-institute/MLJ.jl) solvers you need to load MLJ in addition to ACEfit

```julia
using ACEfit
using MLJ
```

After that you need to load an appropriate MLJ solver. Take a look on available MLJ [solvers](https://alan-turing-institute.github.io/MLJ.jl/dev/model_browser/). Note that only [MLJScikitLearnInterface.jl](https://github.com/JuliaAI/MLJScikitLearnInterface.jl) and [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) have extension available. To use other MLJ solvers please file an issue.

You need to load the solver and then create a solver structure

```julia
# Load ARD solver
ARDRegressor = @load ARDRegressor pkg=MLJScikitLearnInterface

# Create the solver itself and give it parameters
solver = ARDRegressor(
    max_iter = 300,
    tol = 1e-3,
    threshold_lambda = 10000
)
```

After this you can use the MLJ solver like any other solver.

## Index

```@index
```

```@autodocs
Modules = [ACEfit]
```
