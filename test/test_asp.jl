using ACEfit
using LinearAlgebra, Random, Test 

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

@info(" ... ASP")
shuffled_indices = shuffle(1:length(y))
train_indices = shuffled_indices[1:round(Int, 0.85 * length(y))]
val_indices = shuffled_indices[round(Int, 0.85 * length(y)) + 1:end]
At = A[train_indices,:]
Av = A[val_indices,:]
yt = y[train_indices]
yv = y[val_indices]

for (nstore, n1) in [ (20, 21), (100, 101), (200, 165)]
   solver = ACEfit.ASP(P=I, select = :final, nstore = nstore, loglevel=0, traceFlag=true)
   results = ACEfit.solve(solver, A, y)
   @test length(results["path"]) == n1 
end

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


# I didn't wanna add more tsvd tests to yours so I just wrote this one
# I only wanted to naïvely demonstrate that tsvd actually does make a difference! :)

for (select, tolr, tolc) in [ (:final, 20*epsn, 1.5), 
   ( (:byerror,1.3), 20*epsn, 1.5), 
  ( (:bysize,73), 1, 10) ]
   @show select 
   local solver, results, C 
   solver_tsvd = ACEfit.ASP(P=I, select=select, mode=:train, tsvd=true, 
   nstore=100, loglevel=0, traceFlag=true)

   solver = ACEfit.ASP(P=I, select=select, mode=:train, tsvd=false, 
   nstore=100, loglevel=0, traceFlag=true)
   # without validation 
   results_tsvd = ACEfit.solve(solver_tsvd, A, y)
   results = ACEfit.solve(solver, A, y)
   C_tsvd = results_tsvd["C"]
   C = results["C"]

   @show results["nnzs"]
   @show norm(A * C - y)
   @show norm(A * C_tsvd - y)
   if norm(A * C_tsvd - y)< norm(A * C - y)
      @info "tsvd made improvements!"
   else
      @warn "tsvd did NOT make any improvements!"
   end


   # with validation 
   results_tsvd = ACEfit.solve(solver_tsvd, At, yt, Av, yv)
   results = ACEfit.solve(solver, At, yt, Av, yv)
   C_tsvd = results_tsvd["C"]
   C = results["C"]
   @show results["nnzs"]
   @show norm(A * C - y)
   @show norm(A * C_tsvd - y)

   if norm(A * C_tsvd - y)< norm(A * C - y)
      @info "tsvd made improvements!"
   else
      @warn "tsvd did NOT make any improvements!"
   end
end

