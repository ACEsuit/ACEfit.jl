using LinearAlgebra

function llsq(data::AbstractVector, basis; solver = QR())
    A, Y, W = assemble_llsq(data, basis)
    C = solve_llsq(solver, Diagonal(W)*A, Diagonal(W)*Y)
    return A, Y, W, C
end

function assemble_llsq(data, basis)

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
      fill_llsq!(A, Y, W, d, basis; firstrow=firstrow[i])
   end

   return A, Y, W

end

function fill_llsq!(A, Y, W, dat, basis; firstrow=1)
      i1 = firstrow
      i2 = firstrow + countobservations(dat) - 1
      A[i1:i2,:] .= designmatrix(dat, basis)
      Y[i1:i2] .= targetvector(dat)
      W[i1:i2] .= weightvector(dat)
end
