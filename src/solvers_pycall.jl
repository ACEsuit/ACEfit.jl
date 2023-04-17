@doc raw"""
SKLEARN_BRR
"""
struct SKLEARN_BRR
    tol::Number
    n_iter::Integer
end
SKLEARN_BRR(; tol=1e-3, n_iter=300) = SKLEARN_BRR(tol, n_iter)

function linear_solve(solver::SKLEARN_BRR, A, y)
   @info "Entering SKLEARN_BRR"
   BRR = pyimport("sklearn.linear_model")."BayesianRidge"
   clf = BRR(n_iter=solver.n_iter, tol=solver.tol, fit_intercept=true, compute_score=true)
   clf.fit(A, y)
   if length(clf.scores_) < solver.n_iter
      @info "BRR converged to tol=$(solver.tol) after $(length(clf.scores_)) iterations."
   else
      @warn "\nBRR did not converge to tol=$(solver.tol) after n_iter=$(solver.n_iter) iterations.\n"
   end
   c = clf.coef_
   return Dict{String,Any}("C" => c)
end

@doc raw"""
SKLEARN_ARD
"""
struct SKLEARN_ARD
    n_iter::Integer
    tol::Number
    threshold_lambda::Number
end
SKLEARN_ARD(; n_iter=300, tol=1e-3, threshold_lambda=10000) = SKLEARN_ARD(n_iter, tol, threshold_lambda)

function linear_solve(solver::SKLEARN_ARD, A, y)
   ARD = pyimport("sklearn.linear_model")."ARDRegression"
   clf = ARD(n_iter=solver.n_iter, threshold_lambda=solver.threshold_lambda, tol=solver.tol,
             fit_intercept=true, compute_score=true)
   clf.fit(A, y)
   if length(clf.scores_) < solver.n_iter
      @info "ARD converged to tol=$(solver.tol) after $(length(clf.scores_)) iterations."
   else
      @warn "\n\nARD did not converge to tol=$(solver.tol) after n_iter=$(solver.n_iter) iterations.\n\n"
   end
   c = clf.coef_
   return Dict{String,Any}("C" => c)
end
