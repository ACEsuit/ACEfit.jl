module ACEfit_MLJScikitLearnInterface_ext

using ACEfit
using MLJ
using MLJScikitLearnInterface
using PythonCall

function  ACEfit.solve(solver, A, y)
    Atable = MLJ.table(A)
    mach = machine(solver, Atable, y)
    MLJ.fit!(mach)
    params = fitted_params(mach)
    c = params.coef
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

end