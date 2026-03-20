# ACEfit solver test suite: ASP and OMP with various selection strategies

using ACEfit
using LinearAlgebra, Random, Test
using SparseArrays

@info("Test Solver on overdetermined system")

# Problem setup
Random.seed!(1234)
Nobs, Nfeat = 10_000, 400
A1 = randn(Nobs, Nfeat) / sqrt(Nobs)
U, S1, V = svd(A1)
S = 1e-4 .+ ((S1 .- S1[end]) / (S1[1] - S1[end])).^2
A = U * Diagonal(S) * V'
c_ref = randn(Nfeat)
epsn = 1e-5
y = A * c_ref + epsn * randn(Nobs) / sqrt(Nobs)
# P = I
P = Diagonal(1.0 .+ rand(Nfeat))


# Train-test split
shuffled = shuffle(1:Nobs)
train_idx = shuffled[1:round(Int, 0.85 * Nobs)]
val_idx = shuffled[round(Int, 0.85 * Nobs) + 1:end]
At, Av = A[train_idx, :], A[val_idx, :]
yt, yv = y[train_idx], y[val_idx]

# Path length test
for solvertype in (ACEfit.ASP, ACEfit.OMP)
    for (nstore, expected_len) in [(20, 21), (100, 101), (200, 201)]
        solver =  solvertype(; P=P, select=:final, nstore=nstore, loglevel=0)
        results = ACEfit.solve(solver, A, y)
        print(@test length(results["path"]) == expected_len)
    end
end

# Accuracy test (with/without validation)
function test_accuracy(solvertype)
    for (select, tolr, tolc) in [(:final, 15*epsn, 1), ((:byerror, 1.3), 15*epsn, 1), ((:bysize, 360), 1, 15)]
        solver = solvertype(P=P, select=select, loglevel=0)

        # Without validation
        C = ACEfit.solve(solver, A, y)["C"]
        @test norm(A * C - y) < tolr
        @test norm(C - c_ref) < tolc

        # With validation
        C = ACEfit.solve(solver, At, yt, Av, yv)["C"]
        @test norm(Av * C - yv) < tolr
        @test norm(C - c_ref) < tolc
    end
end

test_accuracy(ACEfit.ASP)
test_accuracy(ACEfit.OMP)

# Tracer select test
function test_tracer_select(solvertype)
    solver = solvertype(P=P, select=:final, nstore=100, loglevel=0)
    tracer = ACEfit.solve(solver, At, yt, Av, yv)["path"]

    C_select, _ = ACEfit.asp_select(tracer, (:bysize, 73))
    @test length(C_select.nzind) <= 73

    solver_check = solvertype(P=P, select=(:bysize, 73), nstore=100, loglevel=0)
    C_direct = ACEfit.solve(solver_check, At, yt, Av, yv)["C"]
    @test C_select == C_direct
end

test_tracer_select(ACEfit.ASP)
test_tracer_select(ACEfit.OMP)




function test_solver(name, SolverType, A, y, At, yt, Av, yv, P)
    settings = [
        (:final, 20epsn, 1.5),
        ((:byerror, 1.3), 20epsn, 1.5),
        ((:bysize, 73), 1.0, 10.0)
    ]
 
    err_std = err_tsvd = err_val_std = err_val_tsvd = Inf

    for (select, tolr, tolc) in settings
        @show select
        for tsvd in (false, true)
            solver = SolverType(P=P, select=select, tsvd=tsvd, nstore=100, loglevel=0)
            
            # Solve without validation
            results = ACEfit.solve(solver, A, y)
            C = results["C"]
            err = norm(A * C - y)
            if tsvd == true
                err_tsvd = err
            else
                err_std = err
            end

            # Solve with validation
            results = ACEfit.solve(solver, At, yt, Av, yv)
            C = results["C"]
            err_val = norm(A * C - y)
            if tsvd == true
                err_val_tsvd = err_val
            else
                err_val_std = err_val
            end
        end

        if err_tsvd < err_std
            @info "$name: tsvd made improvements!"
        else
            @warn "$name: tsvd did NOT make any improvements!"
        end

        if err_val_tsvd < err_val_std
            @info "$name (val): tsvd made improvements!"
        else
            @warn "$name (val): tsvd did NOT make any improvements!"
        end
    end
end

test_solver("ASP", ACEfit.ASP, A, y, At, yt, Av, yv, P)
test_solver("OMP", ACEfit.OMP, A, y, At, yt, Av, yv, P)



@info "All tests completed for ASP and OMP solvers."





