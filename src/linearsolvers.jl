using LinearAlgebra: qr, I, norm
using LowRankApprox: pqrfact
using IterativeSolvers
using PyCall

include("bayesianlinear.jl")

# TODO: 
#   - read_dict, write_dict 

@doc raw"""
create_solver
"""
function create_solver(params::Dict)
    solver = uppercase(params["type"])
    delete!(params, "type")
    params = Dict(Symbol(k)=>v for (k,v) in pairs(params))
    if solver == "QR"
        #return QR(; params...)
        return QR(; params...)
    elseif solver == "LSQR"
        #return LSQR(; params...)
        return LSQR(; damp=params[:lsqr_damp], atol=params[:lsqr_atol], conlim=params[:lsqr_conlim], maxiter=params[:lsqr_maxiter], verbose=params[:lsqr_verbose])
    elseif solver == "RRQR"
        #return RRQR(; params...)
        return RRQR(; tol=params[:rrqr_tol])
    elseif solver == "SKLEARN_BRR"
        #return SKLEARN_BRR(; params...)
        return SKLEARN_BRR(; n_iter=params[:brr_n_iter], tol=params[:brr_tol])
    elseif solver == "SKLEARN_ARD"
        #return SKLEARN_ARD(; params...)
        return SKLEARN_ARD(; n_iter=params[:ard_n_iter], tol=params[:ard_tol], threshold_lambda=params[:ard_threshold_lambda])
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
ACEfit.QR(; λ = 0.0, P = nothing)
``` 
where 
* `λ` : regularisation parameter 
* `P` : right-preconditioner / tychonov operator
"""
struct QR
   λ::Number
   P
end

QR(; λ = 0.0, P = I) = QR(λ, P)
         
function solve_llsq(solver::QR, A, y)
   if solver.λ == 0 
      AP = A 
      yP = y 
   else 
      AP = [A; solver.λ * solver.P]
      yP = [y; zeros(eltype(y), size(A, 2))]
   end 
   return qr(AP) \ yP
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

function solve_llsq(solver::RRQR, A, y)
   AP = A / solver.P 
   θP = pqrfact(AP, rtol = solver.rtol) \ y 
   return solver.P \ θP
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

LSQR(; damp=5e-3, atol=1e-6, conlim=1e8, maxiter=100000, verbose=false) = LSQR(damp, atol, conlim, maxiter, verbose, I)

function solve_llsq(solver::LSQR, A, y)
   println("damp  ", solver.damp)
   println("atol  ", solver.atol)
   println("maxiter  ", solver.maxiter)
   # preconditioning
   #if !(solver.P == I)
   #   @info("LSQR: Using preconditioning")
   #   D_inv = pinv(P)
   #   mul!(A,A,D_inv)
   #end
   c, ch = lsqr(A, y; damp=solver.damp, atol=solver.atol, conlim=solver.conlim,
                      maxiter=solver.maxiter, verbose=solver.verbose, log=true)
   println(ch)
   println("relative RMS error  ", norm(A*c - y) / norm(y))
   return c
end

function solve_llsq(solver::LSQR, A::DArray, y::DArray)
   println("hello from solve_llsq dist")
   println("damp  ", solver.damp)
   println("atol  ", solver.atol)
   c = dzeros((size(A,2),), [1])
   c, ch = lsqr!(c, A, y; damp=solver.damp, atol=solver.atol, conlim=solver.conlim,
                          maxiter=solver.maxiter, verbose=solver.verbose, log=true)
   println(ch)
   println("relative RMS error  ", norm(A*c - y) / norm(y))
   return convert(Vector, c)
end

@doc raw"""
SKLEARN_BRR
"""
struct SKLEARN_BRR
    tol::Number
    n_iter::Integer
end
SKLEARN_BRR(; tol=1e-3, n_iter=300) = SKLEARN_BRR(tol, n_iter)

function solve_llsq(solver::SKLEARN_BRR, A, y)
   BRR = pyimport("sklearn.linear_model")["BayesianRidge"]
   clf = BRR(n_iter=solver.n_iter, tol=solver.tol, fit_intercept=true, normalize=true, compute_score=true)
   clf.fit(A, y)
   if length(clf.scores_) < solver.n_iter
      @info "BRR converged to tol=$(solver.tol) after $(length(clf.scores_)) iterations."
   else
      @warn "\nBRR did not converge to tol=$(solver.tol) after n_iter=$(solver.n_iter) iterations.\n"
   end
   c = clf.coef_
   return c
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

function solve_llsq(solver::SKLEARN_ARD, A, y)
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
   return c
end

@doc raw"""
Bayesian Linear
"""
struct BL
end

function solve_llsq(solver::BL, A, y)
   c, _, _, _ = BayesianRegression.bayesian_fit(y, A; verbose=false)
   return c
end

@doc raw"""
Bayesian ARD
"""
struct BARD
end

function solve_llsq(solver::BARD, A, y)
   c, _, _, _, _ = BayesianRegression.ard_fit(y, A; verbose=false)
   return c
end

@doc raw"""
Bayesian Ridge Regression SVD
"""
struct BayesianRidgeRegressionSVD
    verbose::Bool
end
BayesianRidgeRegressionSVD(; verbose=false) = BayesianRidgeRegressionSVD(verbose)

function solve_llsq(solver::BayesianRidgeRegressionSVD, A, y)
   c = BayesianRegression.bayesian_ridge_regression_svd(y, A; verbose=solver.verbose)
   return c
end
