using LinearAlgebra: qr, I, norm
using LowRankApprox: pqrfact
using IterativeSolvers
using PyCall
using .BayesianLinear

@doc raw"""
create_solver(params::Dict)

Convenience function for creating a solver. The `params` should contain
a `type`, whose value is a solver type. The remaining `params` are passed
as keyword arguments to the solver's constructor.

Valid solver types: "QR, LSQR, RRQR, SKLEARN_BRR, SKLEARN_ARD"
"""
function create_solver(params::Dict)
    params = copy(params)
    solver = uppercase(pop!(params, "type"))
    params = Dict(Symbol(k)=>v for (k,v) in pairs(params))
    if solver == "QR"
        return QR(; params...)
    elseif solver == "LSQR"
        return LSQR(; params...)
    elseif solver == "RRQR"
        return RRQR(; params...)
    elseif solver == "SKLEARN_BRR"
        return SKLEARN_BRR(; params...)
    elseif solver == "SKLEARN_ARD"
        return SKLEARN_ARD(; params...)
    elseif solver == "BLR"
        return BayesianLinearRegressionSVD(; params...)
    else
        @error "ACEfit.create_solver does not recognize $(solver)."
    end
end

@doc raw"""
`struct QR` : linear least squares solver, using standard QR factorisation; 
this solver computes 
```math 
 θ = \arg\min \| A \theta - y \|^2 + \lambda \| P \theta \|^2
```
Constructor
```julia
ACEfit.QR(; lambda = 0.0, P = nothing)
``` 
where 
* `λ` : regularisation parameter 
* `P` : right-preconditioner / tychonov operator
"""
struct QR
   lambda::Number
   P
end

QR(; lambda = 0.0, P = I) = QR(lambda, P)
         
function linear_solve(solver::QR, A, y)
   if solver.lambda == 0 
      AP = A 
      yP = y 
   else 
      AP = [A; solver.lambda * solver.P]
      yP = [y; zeros(eltype(y), size(A, 2))]
   end 
   return Dict{String,Any}("C" => qr(AP) \ yP)
end

@doc raw"""
`struct RRQR` : linear least squares solver, using rank-revealing QR 
factorisation, which can sometimes be more robust / faster than the 
standard regularised QR factorisation. This solver first transforms the 
parameters ``\theta_P = P \theta``, then solves
```math 
 θ = \arg\min \| A P^{-1} \theta_P - y \|^2
```
where the truncation tolerance is given by the `rtol` parameter, and 
finally reverses the transformation. This uses the `pqrfact` of `LowRankApprox.jl`; 
For further details see the documentation of 
[`LowRankApprox.jl`](https://github.com/JuliaMatrices/LowRankApprox.jl#qr-decomposition).

Crucially, note that this algorithm is *not deterministic*; the results can change 
slightly between applications.

Constructor
```julia
ACEfit.RRQR(; rtol = 1e-15, P = I)
``` 
where 
* `rtol` : truncation tolerance
* `P` : right-preconditioner / tychonov operator
"""
struct RRQR
   rtol::Number 
   P
end

RRQR(; rtol = 1e-15, P = I) = RRQR(rtol, P) 

function linear_solve(solver::RRQR, A, y)
   AP = A / solver.P 
   θP = pqrfact(AP, rtol = solver.rtol) \ y 
   return Dict{String,Any}("C" => solver.P \ θP)
end

@doc raw"""
LSQR
"""
struct LSQR
   damp::Number
   atol::Number
   conlim::Number
   maxiter::Integer
   verbose::Bool
   P
end

LSQR(; damp=5e-3, atol=1e-6, conlim=1e8, maxiter=100000, verbose=false, P=nothing) = LSQR(damp, atol, conlim, maxiter, verbose, P)

function linear_solve(solver::LSQR, A, y)
   @warn "Need to apply preconditioner in LSQR."
   println("damp  ", solver.damp)
   println("atol  ", solver.atol)
   println("maxiter  ", solver.maxiter)
   c, ch = lsqr(A, y; damp=solver.damp, atol=solver.atol, conlim=solver.conlim,
                      maxiter=solver.maxiter, verbose=solver.verbose, log=true)
   println(ch)
   println("relative RMS error  ", norm(A*c - y) / norm(y))
   return Dict{String,Any}("C" => c)
end

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
   BRR = pyimport("sklearn.linear_model")["BayesianRidge"]
   clf = BRR(n_iter=solver.n_iter, tol=solver.tol, fit_intercept=true, normalize=true, compute_score=true)
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
   ARD = pyimport("sklearn.linear_model")["ARDRegression"]
   clf = ARD(n_iter=solver.n_iter, threshold_lambda=solver.threshold_lambda, tol=solver.tol,
             fit_intercept=true, normalize=true, compute_score=true)
   clf.fit(A, y)
   if length(clf.scores_) < solver.n_iter
      @info "ARD converged to tol=$(solver.tol) after $(length(clf.scores_)) iterations."
   else
      @warn "\n\nARD did not converge to tol=$(solver.tol) after n_iter=$(solver.n_iter) iterations.\n\n"
   end
   c = clf.coef_
   return Dict{String,Any}("C" => c)
end

@doc raw"""
Bayesian Linear Regression
"""
struct BL
end

function linear_solve(solver::BL, A, y)
   c, _, _, _ = bayesian_fit(y, A; verbose=false)
   return Dict{String,Any}("C" => c)
end

@doc raw"""
Bayesian ARD
"""
struct BARD
end

function linear_solve(solver::BARD, A, y)
   c, _, _, _, _ = ard_fit(y, A; verbose=false)
   return Dict{String,Any}("C" => c)
end

@doc raw"""
Bayesian Linear Regression SVD
"""
struct BayesianLinearRegressionSVD
    verbose::Bool
    committee_size
end
BayesianLinearRegressionSVD(; verbose=false, committee_size=0) =
    BayesianLinearRegressionSVD(verbose, committee_size)

function linear_solve(solver::BayesianLinearRegressionSVD, A, y)
   blr = bayesian_linear_regression_svd(A, y; verbose=solver.verbose, committee_size=solver.committee_size)
   results = Dict{String,Any}("C" => blr["c"])
   haskey(blr, "committee") && (results["committee"] = blr["committee"])
   return results
end
