using Distributed, DistributedArrays
@everywhere using Distributed, DistributedArrays

using ACEbase: ACEBasis

# DO WE EVEN NEED THIS? 
#
# struct LLSQ
#    basis 
#    configs
#    solver 
#    meta::Dict{String, Any}
# end

# LLSQ(; basis = nothing, configs = nothing, 
#        solver = QR(), meta = Dict{String, Any}()) = 
#                         LLSQ(basis, configs, solver, meta)


# todo - move this to ACEbase I think 

function get_basis end 


# TODO: Maybe we can provide more fine-tuned execution models 
#    struct Distributed parameters end  struct Serial ... end and so forth 
#    rather than those symbols :serial, :mt, :dist


function llsq!(model, data::AbstractVector, par = :serial; solver = QR())
   basis = get_basis(model) # should return an ACEBasis 
   θ = llsq(basis, data, par; solver = solver)
   set_params!(model, θ)
   return model 
end


function llsq(basis, data::AbstractVector, par = :serial; solver = QR())
   if par == :serial
      _iterate = siterate
      A, y = asm_llsq(basis, data, _iterate)
   elseif par == :mt
      _iterate = titerate
      A, y = asm_llsq(basis, data, _iterate)
   elseif par == :dist
      _iterate = siterate
      A, y = asm_llsq_dist(basis, data, _iterate)
   else 
      error("unknown assembly type")
   end

   return solve_llsq(solver, A, y)
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

function asm_llsq(basis, data, _iterate)
   firstidx, Nobs = get_lsq_indices(data)

   # allocate - maybe we need to check somewhere what the 
   #            eltypes of A, y should be? real, complex, precision?
   A = zeros(Nobs, length(basis))
   Y = zeros(Nobs)
   
   # inner assembly (this knows about A and Y)
   idx = 1
   # TODO: i0 is not used anymore, should revisit
   function asm_lsq_inner(i0, dat)
      for o in observations(dat)
         # TODO: this isn't type stable; for very cheap models, this inner 
         #       loop could be a bottleneck, can it be fixed? 
         oB = basis_obs(typeof(o), basis, dat.config)
         y = vec_obs(o)
         w = get_weight(o)
         inds = idx:idx+length(y)-1
         Y[inds] .= w .* y[:]
         for ib = 1:length(basis) 
            ovec = vec_obs(oB[ib])
            A[inds, ib] .= w .* ovec[:]
         end
         idx += length(y)
      end
      return nothing 
   end

   _iterate(asm_lsq_inner, data)

   return A, Y 
end

function asm_llsq_dist(basis, data, _iterate)

   # for now, all data is sent to all workers, should revisit
   firstidx, Nobs = get_lsq_indices(data)

   A = dzeros(Nobs, length(basis))
   Y = dzeros(Nobs)

   idx = 1
   # TODO: i0 is not used anymore, should revisit
   function asm_lsq_inner(i0, dat)
      for o in observations(dat)
         oB = basis_obs(typeof(o), basis, dat.config)
         y = vec_obs(o)
         w = get_weight(o)
         inds = idx:idx+length(y)-1
         idx += length(y)
         # check whether row indices are within local bounds
         localrows = localindices(Y)[1]
         if inds[1]>localrows[end] || inds[end]<localrows[1]
            continue
         # TODO: cases with partial overlap
         #elseif
         end
         # fill rows of Y and A
         localpart(Y)[inds.-localrows[1].+1] .= w .* y[:]
         for ib = 1:length(basis)
            ovec = vec_obs(oB[ib])
            localpart(A)[inds.-localrows[1].+1, ib] .= w .* ovec[:]
         end
      end
      return nothing
   end

   @sync [@spawnat w _iterate(asm_lsq_inner, data) for w in workers()]
   A = convert(Array, A)  # send data to main processs and build full matrix

   return A, Y
end
