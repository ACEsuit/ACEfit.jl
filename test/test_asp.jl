using ACEfit
using LinearAlgebra, Random, Test 
using Random

##

@info("Test Solver on overdetermined system")

Random.seed!(1234)
Nobs = 10_000
Nfeat = 300
A1 = randn(Nobs, Nfeat) / sqrt(Nobs)
U, S1, V = svd(A1)
S = 1e-4 .+ ((S1 .- S1[end]) / (S1[1] - S1[end])).^2
A = U * Diagonal(S) * V'
c_ref = randn(Nfeat)
epsn = 1e-5 
y = A * c_ref + epsn * randn(Nobs) / sqrt(Nobs)
P = Diagonal(1.0 .+ rand(Nfeat))

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

##

# Experimental Implementation of tsvd postprocessing 


using SparseArrays

function solve_tsvd(At, yt, Av, yv) 
   Ut, Σt, Vt = svd(At); zt = Ut' * yt
   Qv, Rv = qr(Av); zv = Matrix(Qv)' * yv
   @assert issorted(Σt, rev=true)

   Rv_Vt = Rv * Vt

   θv = zeros(size(Av, 2))
   θv[1] = zt[1] / Σt[1] 
   rv = Rv_Vt[:, 1] * θv[1] - zv 

   tsvd_errs = Float64[] 
   push!(tsvd_errs, norm(rv))

   for k = 2:length(Σt)
      θv[k] = zt[k] / Σt[k]
      rv += Rv_Vt[:, k] * θv[k]
      push!(tsvd_errs, norm(rv))
   end

   imin = argmin(tsvd_errs)
   θv[imin+1:end] .= 0
   return Vt * θv, Σt[imin]
end

function post_asp_tsvd(path, At, yt, Av, yv) 
   Qt, Rt = qr(At); zt = Matrix(Qt)' * yt
   Qv, Rv = qr(Av); zv = Matrix(Qv)' * yv

   post = [] 
   for (θ, λ) in path
      if isempty(θ.nzind); push!(post, (θ = θ, λ = λ, σ = Inf)); continue; end  
      inz = θ.nzind 
      θ1, σ = solve_tsvd(Rt[:, inz], zt, Rv[:, inz], zv)
      θ2 = copy(θ); θ2[inz] .= θ1
      push!(post, (θ = θ2, λ = λ, σ = σ))
   end 
   return identity.(post)
end   

solver = ACEfit.ASP(P=I, select = :final, loglevel=0, traceFlag=true)
result = ACEfit.solve(solver, At, yt); 
post = post_asp_tsvd(result["path"], At, yt, Av, yv);
