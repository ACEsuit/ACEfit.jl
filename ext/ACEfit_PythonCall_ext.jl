module ACEfit_PythonCall_ext

using ACEfit
using PythonCall


function ACEfit.solve(solver::ACEfit.SKLEARN_BRR, A, y)
    @info "Entering SKLEARN_BRR"
    BRR = pyimport("sklearn.linear_model")."BayesianRidge"
    clf = BRR(max_iter = solver.max_iter, tol = solver.tol, fit_intercept = true,
              compute_score = true)
    clf.fit(A, y)
    if length(clf.scores_) < solver.max_iter
        @info "BRR converged to tol=$(solver.tol) after $(length(clf.scores_)) iterations."
    else
        @warn "\nBRR did not converge to tol=$(solver.tol) after max_iter=$(solver.max_iter) iterations.\n"
    end
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end


function ACEfit.solve(solver::ACEfit.SKLEARN_ARD, A, y)
    ARD = pyimport("sklearn.linear_model")."ARDRegression"
    clf = ARD(max_iter = solver.max_iter, threshold_lambda = solver.threshold_lambda,
              tol = solver.tol,
              fit_intercept = true, compute_score = true)
    clf.fit(A, y)
    if length(clf.scores_) < solver.max_iter
        @info "ARD converged to tol=$(solver.tol) after $(length(clf.scores_)) iterations."
    else
        @warn "\n\nARD did not converge to tol=$(solver.tol) after max_iter=$(solver.max_iter) iterations.\n\n"
    end
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array,c) )
end

end
