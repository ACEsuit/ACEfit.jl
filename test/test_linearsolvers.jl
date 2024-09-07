
using ACEfit
using LinearAlgebra
using Random
using PythonCall

@info("Test Solver on overdetermined system")
Nobs = 10_000
Nfeat = 100
A1 = randn(Nobs, Nfeat) / sqrt(Nobs)
U, S1, V = svd(A1)
S = 1e-4 .+ ((S1 .- S1[end]) / (S1[1] - S1[end])).^2
A = U * Diagonal(S) * V'
c_ref = randn(Nfeat)
y = A * c_ref + 1e-3 * randn(Nobs) / sqrt(Nobs)
P = Diagonal(1.0 .+ rand(Nfeat))

@info(" ... QR")
solver = ACEfit.QR()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@info(" ... regularised QR, λ = 1e-5")
solver = ACEfit.QR(lambda = 1e-5, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@info(" ... regularised QR, λ = 1e-2")
solver = ACEfit.QR(lambda = 1e-2, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@info(" ... RRQR, rtol = 1e-15")
solver = ACEfit.RRQR(rtol = 1e-15, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)


@info(" ... RRQR, rtol = 1e-5")
solver = ACEfit.RRQR(rtol = 1e-5, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@info(" ... RRQR, rtol = 1e-3")
solver = ACEfit.RRQR(rtol = 1e-3, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@info(" ... LSQR")
solver = ACEfit.LSQR(damp = 0, atol = 1e-6)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@info(" ... SKLEARN_BRR")
solver = ACEfit.SKLEARN_BRR()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@info(" ... SKLEARN_ARD")
solver = ACEfit.SKLEARN_ARD()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@info(" ... BLR")
solver = ACEfit.BLR()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@info(" ... TruncatedSVD(; rtol = 1e-5)")
solver = ACEfit.TruncatedSVD(; rtol = 1e-5)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@info(" ... TruncatedSVD(; rtol = 1e-4)")
solver = ACEfit.TruncatedSVD(; rtol=1e-4)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)


@info(" ... ASP")
shuffled_indices = shuffle(1:length(y))
train_indices = shuffled_indices[1:round(Int, 0.85 * length(y))]
val_indices = shuffled_indices[round(Int, 0.85 * length(y)) + 1:end]
At = A[train_indices,:]
Av = A[val_indices,:]
yt = y[train_indices]
yv = y[val_indices]

for (sel, mod) in [((:final,nothing),:basic ),( (:byerror,1.3),:basic ),((:bysize,73),:basic )
    ,((:val,nothing),:smart ),((:byerror,1.3),:smart ),((:bysize,73),:smart )]
    solver = ACEfit.ASP(P=I, select= sel, mode=mod ,params = (loglevel=0, traceFlag=true))
    if mod == :basic
        results = ACEfit.solve(solver, A, y)
        C = results["C"]
        full_path = results["path"]
        @show results["nnzs"]
        @show norm(A * C - y)
        @show norm(C)
        @show norm(C - c_ref)
    elseif mod == :smart
        results = ACEfit.solve(solver, At, yt, Av, yv)
        C = results["C"]
        full_path = results["path"]
        @show results["nnzs"]
        @show norm(Av * C - yv)
        @show norm(C)
        @show norm(C - c_ref)
    end
end
