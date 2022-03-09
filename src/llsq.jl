
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
   elseif par == :mt
      _iterate = titerate
   elseif par == :dist
      error("distributed llsq not yet implemented")   
   else 
      error("unknown assembly type")
   end
      
   A, y = asm_llsq(basis, data, _iterate)

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
   function asm_lsq_inner(i0, dat)
      println("idx:  ", idx)
      for o in observations(dat)
         # TODO: this isn't type stable; for very cheap models, this inner 
         #       loop could be a bottleneck, can it be fixed? 
         oB = basis_obs(typeof(o), basis, dat.config)
         y = vec_obs(o)
         w = get_weight(o)
         inds = idx:idx+length(y)-1
         println("  inds:  ", inds, "\t  max ind should reach length(Y):  ", length(Y))
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

