using LinearAlgebra

function llsq(basis, data::AbstractVector, Vref, par = :serial; solver = QR())
   if par == :serial
      _iterate = siterate
      A, y, w = assemble_llsq(basis, data, _iterate)
      c = solve_llsq(solver, A, y)
      return A, y, w, c
   else 
      error("unknown assembly type")
   end
end

function llsq_new(data::AbstractVector, basis; solver = QR(), par = :serial)
    assemble_llsq_new(data, basis)
end

function get_lsq_indices(data)
   # count the number of observations and assign indices in the lsq matrix
   # we do this always in serial since it should take essentially no time
   # (but what if the data lives distributed???)
   Nobs = 0
   firstidx = zeros(Int, length(data))
   function count_Nobs(i, dat)
      firstidx[i] = Nobs + 1
      for o in ACEfit.observations(dat)
         Nobs += length(ACEfit.vec_obs(o))
      end
   end
   ACEfit.siterate(count_Nobs, data)
   return firstidx, Nobs
end

function assemble_llsq_new(data, basis)

   firstrow = ones(Int,length(data))
   rowcount = ones(Int,length(data))
   for (i,d) in enumerate(data)
      rowcount[i] = countrows(d)
      i<length(data) && (firstrow[i+1] = firstrow[i] + rowcount[i])
   end

   @info "Creating design matrix with size ($(sum(rowcount)), $(length(basis)))"
   A = zeros(sum(rowcount),length(basis))
   Y = zeros(size(A,1))
   W = zeros(size(A,1))

   for (i,d) in enumerate(data)
      i1, i2 = firstrow[i], firstrow[i]+rowcount[i]-1
      a = designmatrix(d, basis)
      y = targetvector(d)
      w = weightvector(d)
      A[i1:i2,:] .= a
      Y[i1:i2] .= y
      W[i1:i2] .= w
   end

end

function assemble_llsq(basis, data, _iterate)

   _, num_obs = get_lsq_indices(data)

   println("Creating design matrix with size (", num_obs, ", ", length(basis), ")")
   A = zeros(num_obs, length(basis))
   Y = zeros(num_obs)
   W = zeros(num_obs)
   
   # inner assembly (this knows about A and Y)
   idx = 1
   function asm_lsq_inner(i0, dat)
      for o in observations(dat)
         # TODO: this isn't type stable; for very cheap models, this inner 
         #       loop could be a bottleneck, can it be fixed? 
         oB = basis_obs(typeof(o), basis, dat.config)
         y = vec_obs(o)
         w = get_weight(o)
         # TODO: make this an input parameter eventually
         if hasproperty(o, :E) || hasproperty(o, :V)
            w = w ./ sqrt(length(dat.config))
         end
         inds = idx:idx+length(y)-1
         Y[inds] .= w.*y[:]
         W[inds] .= w.*ones(length(y))
         for ib = 1:length(basis) 
            ovec = vec_obs(oB[ib])
            A[inds, ib] .= w.*ovec[:]
         end
         idx += length(y)
      end
      return nothing 
   end

   _iterate(asm_lsq_inner, data)

   return A, Y, W
end
