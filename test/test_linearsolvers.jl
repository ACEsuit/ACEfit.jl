
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
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)

@info(" ... regularised QR, λ = 1.0")
solver = ACEfit.QR(lambda = 1e0, P = P)
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)

@info(" ... regularised QR, λ = 10.0")
solver = ACEfit.QR(lambda = 1e1, P = P)
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)

@info(" ... RRQR, rtol = 1e-15")
solver = ACEfit.RRQR(rtol = 1e-15, P = P)
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)

@info(" ... RRQR, rtol = 0.5")
solver = ACEfit.RRQR(rtol = 0.5, P = P)
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)

@info(" ... RRQR, rtol = 0.99")
solver = ACEfit.RRQR(rtol = 0.99, P = P)
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)

@info(" ... LSQR")
solver = ACEfit.LSQR(damp=0, atol=1e-6)
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)

@info(" ... SKLEARN_BRR")
solver = ACEfit.SKLEARN_BRR()
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)

@info(" ... SKLEARN_ARD")
solver = ACEfit.SKLEARN_ARD()
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)

@info(" ... Bayesian Linear")
solver = ACEfit.BL()
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)

@info(" ... Bayesian ARD")
solver = ACEfit.BARD()
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)

@info(" ... Bayesian Linear Regression SVD")
solver = ACEfit.BayesianLinearRegressionSVD()
C = ACEfit.linear_solve(solver, A, y)
@show norm(A * C - y)
@show norm(C)
