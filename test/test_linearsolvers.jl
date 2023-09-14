
using ACEfit
using LinearAlgebra
using PythonCall

@info("Test Solver on overdetermined system")
Nobs = 10_000
Nfeat = 100
A = randn(Nobs, Nfeat) / sqrt(Nobs)
y = randn(Nobs)
P = Diagonal(1.0 .+ rand(Nfeat))

@info(" ... QR")
solver = ACEfit.QR()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... regularised QR, λ = 1.0")
solver = ACEfit.QR(lambda = 1e0, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... regularised QR, λ = 10.0")
solver = ACEfit.QR(lambda = 1e1, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... RRQR, rtol = 1e-15")
solver = ACEfit.RRQR(rtol = 1e-15, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... RRQR, rtol = 0.5")
solver = ACEfit.RRQR(rtol = 0.5, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... RRQR, rtol = 0.99")
solver = ACEfit.RRQR(rtol = 0.99, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... LSQR")
solver = ACEfit.LSQR(damp = 0, atol = 1e-6)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... SKLEARN_BRR")
solver = ACEfit.SKLEARN_BRR()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... SKLEARN_ARD")
solver = ACEfit.SKLEARN_ARD()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... BLR")
solver = ACEfit.BLR()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... FSBL")
solver = ACEfit.FSBL()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
