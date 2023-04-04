using Distributed
using LinearAlgebra
using ProgressMeter
using SharedArrays

function linear_fit(data::AbstractVector, basis, solver=QR(); P=nothing)
    @info "Entering linear_assemble"
    A, Y, W = linear_assemble(data, basis)
    @info "After linear_assemble"
    flush(stdout); flush(stderr)
    lmul!(Diagonal(W),A)
    Y = W.*Y
    !isnothing(P) && (A = A*pinv(P))
    GC.gc()
    @info "Entering linear_solve"
    results = linear_solve(solver, A, Y)
    C = results["C"]
    @info "After linear_solve"
    if !isnothing(P)
        A = A*P
        C = pinv(P)*C
        # TODO: deapply preconditioner to committee
    end
    lmul!(inv(Diagonal(W)),A)
    Y = (1.0./W).*Y
    fit = Dict{String,Any}("C" => C)
    haskey(results, "committee") && (fit["committee"] = results["committee"])
    return fit
end

function linear_assemble(data, basis)

   @info "Assembling linear problem."
   row_start, row_count = row_info(data)

   @info "  - Creating feature matrix with size ($(sum(row_count)), $(length(basis)))."
   A = SharedArray(zeros(sum(row_count),length(basis)))
   Y = SharedArray(zeros(size(A,1)))
   W = SharedArray(zeros(size(A,1)))

   f = i -> linear_fill!(A, Y, W, data[i], basis; row_start=row_start[i])
   if nprocs() == 1
       @info "  - Beginning assembly in serial mode."
       @showprogress map(f, 1:length(data))
   else
       @info "  - Beginning assembly in distributed mode with $(nprocs()) processes."
       @showprogress pmap(f, 1:length(data))
   end

   @info "  - Assembly completed."
   flush(stdout); flush(stderr)
   return Array(A), Array(Y), Array(W)
end

function linear_fill!(A, Y, W, dat, basis; row_start=1)
   i1 = row_start
   i2 = row_start + count_observations(dat) - 1
   A[i1:i2,:] .= feature_matrix(dat, basis)
   Y[i1:i2] .= target_vector(dat)
   W[i1:i2] .= weight_vector(dat)
   GC.gc()  # not sure if this is necessary?
   return nothing
end
