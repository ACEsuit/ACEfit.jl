using LinearAlgebra: qr, I 
using LowRankApprox: pqrfact
using IterativeSolvers
using PyCall

include("bayesianlinear.jl")

# TODO: 
#   - read_dict, write_dict 
#   - LSQR 
#   - various scikit-learn solvers 

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

QR(; λ = 0.0, P = nothing) = QR(λ, P)

         
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
end

LSQR(; damp=0, atol=1e-6) = LSQR(damp, atol)

function solve_llsq(solver::LSQR, A, y)
   c = lsqr(A, y, damp=solver.damp, atol=solver.atol)
   return c
end

function solve_llsq(solver::LSQR, A::DArray, y::DArray)
   c = dzeros((size(A,2),), [1])
   lsqr!(c, A, y, damp=solver.damp, atol=solver.atol)
   return convert(Vector, c)
end

@doc raw"""
SKLEARN_BRR
"""
struct SKLEARN_BRR
end

function solve_llsq(solver::SKLEARN_BRR, A, y)
   BRR = pyimport("sklearn.linear_model")["BayesianRidge"]
   clf = BRR()
   clf.fit(A, y)
   c = clf.coef_
   return c
end

@doc raw"""
SKLEARN_ARD
"""
struct SKLEARN_ARD
end

function solve_llsq(solver::SKLEARN_ARD, A, y)
   ARD = pyimport("sklearn.linear_model")["ARDRegression"]
   clf = ARD(n_iter=100000, compute_score=true, verbose=true)
   clf.fit(A, y)
   println("scores length  ", length(clf.scores_))
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
