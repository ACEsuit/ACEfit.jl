module ACEfit_MLJLinearModels_ext

using MLJ
using ACEfit
using MLJLinearModels

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
    return Dict{String, Any}("C" => params.coef )
end

end