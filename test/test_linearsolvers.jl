
using ACEfit 
using LinearAlgebra

@info("Test Solver on overdetermined system")
Nobs = 10_000 
Nfeat = 100 
A = randn(Nobs, Nfeat) / sqrt(Nobs)
y = randn(Nobs)
P = Diagonal(1.0 .+ rand(Nfeat))

@info(" ... QR")
solver = ACEfit.QR()
θ = ACEfit.solve_llsq(solver, A, y)
@show norm(A * θ - y)
@show norm(θ)

@info(" ... regularised QR, λ = 1.0")
solver = ACEfit.QR(λ = 1e0, P = P )
θ = ACEfit.solve_llsq(solver, A, y)
@show norm(A * θ - y)
@show norm(θ)

@info(" ... regularised QR, λ = 10.0")
solver = ACEfit.QR(λ = 1e1, P = P)
θ = ACEfit.solve_llsq(solver, A, y)
@show norm(A * θ - y)
@show norm(θ)

@info(" ... RRQR, rtol = 1e-15")
solver = ACEfit.RRQR(rtol = 1e-15, P = P)
θ = ACEfit.solve_llsq(solver, A, y)
@show norm(A * θ - y)
@show norm(θ)

@info(" ... RRQR, rtol = 0.5")
solver = ACEfit.RRQR(rtol = 0.5, P = P)
θ = ACEfit.solve_llsq(solver, A, y)
@show norm(A * θ - y)
@show norm(θ)

@info(" ... RRQR, rtol = 0.99")
solver = ACEfit.RRQR(rtol = 0.99, P = P)
θ = ACEfit.solve_llsq(solver, A, y)
@show norm(A * θ - y)
@show norm(θ)

@info(" ... SKLEARN_BRR")
solver = ACEfit.SKLEARN_BRR()
θ = ACEfit.solve_llsq(solver, A, y)
@show norm(A * θ - y)
@show norm(θ)

@info(" ... SKLEARN_ARD")
solver = ACEfit.SKLEARN_ARD()
θ = ACEfit.solve_llsq(solver, A, y)
@show norm(A * θ - y)
@show norm(θ)
