using Distributed
using LinearAlgebra
using ProgressMeter
using SharedArrays

function linear_fit(data::AbstractVector, basis, solver=QR(), mode=:serial)
    A, Y, W = linear_assemble(data, basis, mode)
    C = linear_solve(solver, Diagonal(W)*A, Diagonal(W)*Y)
    return A, Y, W, C
end

function linear_assemble(data, basis, mode=:serial)
   @info "Assembling linear problem."
   row_start, row_count = row_info(data)

   @info "  - Creating feature matrix with size ($(sum(row_count)), $(length(basis)))."
   A = SharedArray(zeros(sum(row_count),length(basis)))
   Y = SharedArray(zeros(size(A,1)))
   W = SharedArray(zeros(size(A,1)))

   f = i -> linear_fill!(A, Y, W, data[i], basis; row_start=row_start[i])
   if mode == :serial
       @info "  - Beginning assembly in serial mode."
       @showprogress map(f, 1:length(data))
   elseif mode == :distributed
       @info "  - Beginning assembly in distributed mode with $(nworkers()) workers."
       @showprogress pmap(f, 1:length(data))
   else
       @error "In linear_assemble, mode $mode is invalid."
   end

   @info "  - Assembly completed."
   return Array(A), Array(Y), Array(W)
end

function linear_fill!(A, Y, W, dat, basis; row_start=1)
      i1 = row_start
      i2 = row_start + count_observations(dat) - 1
      A[i1:i2,:] .= feature_matrix(dat, basis)
      Y[i1:i2] .= target_vector(dat)
      W[i1:i2] .= weight_vector(dat)
end
