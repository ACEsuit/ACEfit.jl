using Distributed
using LinearAlgebra
using SharedArrays

function llsq(data::AbstractVector, basis; solver=QR())
    A, Y, W = llsq_assemble(data, basis, :distributed)
    C = llsq_solve(solver, Diagonal(W)*A, Diagonal(W)*Y)
    return A, Y, W, C
end

function llsq_assemble(data, basis, mode=:serial)
    if mode == :serial
        return llsq_assemble_serial(data, basis)
    elseif mode == :distributed
        return llsq_assemble_distributed(data, basis)
    else
        @error "In llsq_assemble, mode $mode is invalid."
    end
end

function llsq_assemble_serial(data, basis)
   firstrow = ones(Int,length(data))
   rowcount = ones(Int,length(data))
   for (i,d) in enumerate(data)
      rowcount[i] = countobservations(d)
      i<length(data) && (firstrow[i+1] = firstrow[i] + rowcount[i])
   end

   @info "Creating matrix with size ($(sum(rowcount)), $(length(basis)))"
   A = zeros(sum(rowcount),length(basis))
   Y = zeros(size(A,1))
   W = zeros(size(A,1))

   for (i,d) in enumerate(data)
      llsq_fill!(A, Y, W, d, basis; firstrow=firstrow[i])
   end

   return A, Y, W
end

function llsq_assemble_distributed(data, basis)
   @info "Assembling least squares problem in distributed mode."
   firstrow = ones(Int,length(data))
   rowcount = ones(Int,length(data))
   for (i,d) in enumerate(data)
      rowcount[i] = countobservations(d)
      i<length(data) && (firstrow[i+1] = firstrow[i] + rowcount[i])
   end

   @info "Creating matrix with size ($(sum(rowcount)), $(length(basis)))"
   A = SharedArray(zeros(sum(rowcount),length(basis)))
   Y = SharedArray(zeros(size(A,1)))
   W = SharedArray(zeros(size(A,1)))

   f = x -> llsq_fill!(A, Y, W, x[1], basis; firstrow=x[2])
   pmap(f, zip(data,firstrow))

   return Array(A), Array(Y), Array(W)
end

function llsq_fill!(A, Y, W, dat, basis; firstrow=1)
      i1 = firstrow
      i2 = firstrow + countobservations(dat) - 1
      A[i1:i2,:] .= designmatrix(dat, basis)
      Y[i1:i2] .= targetvector(dat)
      W[i1:i2] .= weightvector(dat)
end
