
using ACEfit
using LinearAlgebra, Random, Test 
using Random
using PythonCall

##

@info("Test Solver on overdetermined system")

Random.seed!(1234)
Nobs = 10_000
Nfeat = 100
A1 = randn(Nobs, Nfeat) / sqrt(Nobs)
U, S1, V = svd(A1)
S = 1e-4 .+ ((S1 .- S1[end]) / (S1[1] - S1[end])).^2
A = U * Diagonal(S) * V'
c_ref = randn(Nfeat)
epsn = 1e-5 
y = A * c_ref + epsn * randn(Nobs) / sqrt(Nobs)
P = Diagonal(1.0 .+ rand(Nfeat))

##

@info(" ... QR")
solver = ACEfit.QR()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)
@test norm(A * C - y) < 10 * epsn 
@test norm(C - c_ref) < 100 * epsn 

##

@info(" ... regularised QR, λ = 1e-5")
solver = ACEfit.QR(lambda = 1e-5, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)
@test norm(A * C - y) < 10 * epsn 
@test norm(C - c_ref) < 1000 * epsn 


##

@info(" ... regularised QR, λ = 1e-2")
solver = ACEfit.QR(lambda = 1e-2, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@test norm(A * C - y) < 1
@test norm(C - c_ref) < 10

##

@info(" ... RRQR, rtol = 1e-15")
solver = ACEfit.RRQR(rtol = 1e-15, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@test norm(A * C - y) < 10 * epsn 
@test norm(C - c_ref) < 100 * epsn 

##

@info(" ... RRQR, rtol = 1e-5")
solver = ACEfit.RRQR(rtol = 1e-5, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)
@test norm(A * C - y) < 10 * epsn 
@test norm(C - c_ref) < 100 * epsn 

##

@info(" ... RRQR, rtol = 1e-3")
solver = ACEfit.RRQR(rtol = 1e-3, P = P)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@test norm(A * C - y) < 1
@test norm(C - c_ref) < 1

##

@info(" ... LSQR")
solver = ACEfit.LSQR(damp = 0, atol = 1e-6)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@test norm(A * C - y) < 10 * epsn 
@test norm(C - c_ref) < 1

##

@info(" ... SKLEARN_BRR")
solver = ACEfit.SKLEARN_BRR()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

##

@info(" ... SKLEARN_ARD")
solver = ACEfit.SKLEARN_ARD()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

##

@info(" ... BLR")
solver = ACEfit.BLR()
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@test norm(A * C - y) < 10 * epsn 
@test norm(C - c_ref) < 1

##

@info(" ... TruncatedSVD(; rtol = 1e-5)")
solver = ACEfit.TruncatedSVD(; rtol = 1e-5)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@test norm(A * C - y) < 10 * epsn 
@test norm(C - c_ref) < 100 * epsn 

##

@info(" ... TruncatedSVD(; rtol = 1e-4)")
solver = ACEfit.TruncatedSVD(; rtol=1e-4)
results = ACEfit.solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
@show norm(C - c_ref)

@test norm(A * C - y) < 10 * epsn 
@test norm(C - c_ref) < 1

##

@info(" ... ASP")
shuffled_indices = shuffle(1:length(y))
train_indices = shuffled_indices[1:round(Int, 0.85 * length(y))]
val_indices = shuffled_indices[round(Int, 0.85 * length(y)) + 1:end]
At = A[train_indices,:]
Av = A[val_indices,:]
yt = y[train_indices]
yv = y[val_indices]

for (select, tolr, tolc) in [ (:final, 10*epsn, 1), 
                             ( (:byerror,1.3), 10*epsn, 1), 
                            ( (:bysize,73), 1, 10) ]
    @show select 
    local solver, results, C 
    solver = ACEfit.ASP(P=I, select = select, loglevel=0, traceFlag=true)
    # without validation 
    results = ACEfit.solve(solver, A, y)
    C = results["C"]
    full_path = results["path"]
    @show results["nnzs"]
    @show norm(A * C - y)
    @show norm(C)
    @show norm(C - c_ref)

    @test norm(A * C - y) < tolr
    @test norm(C - c_ref) < tolc
    

    # with validation 
    results = ACEfit.solve(solver, At, yt, Av, yv)
    C = results["C"]
    full_path = results["path"]
    @show results["nnzs"]
    @show norm(Av * C - yv)
    @show norm(C)
    @show norm(C - c_ref)

    @test norm(Av * C - yv) < tolr
    @test norm(C - c_ref) < tolc
end
