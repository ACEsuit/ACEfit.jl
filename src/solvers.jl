using LinearAlgebra: qr, I, norm
using LowRankApprox: pqrfact
using IterativeSolvers
using .BayesianLinear

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
    P::Any
end

QR(; lambda = 0.0, P = I) = QR(lambda, P)

function solve(solver::QR, A, y)
    if solver.lambda == 0
        AP = A
        yP = y
    else
        AP = [A; solver.lambda * solver.P]
        yP = [y; zeros(eltype(y), size(A, 2))]
    end
    return Dict{String, Any}("C" => qr(AP) \ yP)
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
    P::Any
end

RRQR(; rtol = 1e-15, P = I) = RRQR(rtol, P)

function solve(solver::RRQR, A, y)
    AP = A / solver.P
    θP = pqrfact(AP, rtol = solver.rtol) \ y
    return Dict{String, Any}("C" => solver.P \ θP)
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
    P::Any
end

function LSQR(; damp = 5e-3, atol = 1e-6, conlim = 1e8, maxiter = 100000, verbose = false,
              P = nothing)
    LSQR(damp, atol, conlim, maxiter, verbose, P)
end

function solve(solver::LSQR, A, y)
    @warn "Need to apply preconditioner in LSQR."
    println("damp  ", solver.damp)
    println("atol  ", solver.atol)
    println("maxiter  ", solver.maxiter)
    c, ch = lsqr(A, y; damp = solver.damp, atol = solver.atol, conlim = solver.conlim,
                 maxiter = solver.maxiter, verbose = solver.verbose, log = true)
    println(ch)
    println("relative RMS error  ", norm(A * c - y) / norm(y))
    return Dict{String, Any}("C" => c)
end

@doc raw"""
`struct BLR` : Bayesian linear regression

    Refer to bayesianlinear.jl (for now) for kwarg definitions.
"""
struct BLR
    kwargs::Dict{Any,Any}
end

function BLR(; kwargs...)
    return BLR(Dict(kwargs))
end

function solve(solver::BLR, A, y)
    return bayesian_linear_regression(A, y; solver.kwargs...)
end
