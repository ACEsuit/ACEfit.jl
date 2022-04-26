using Distributed, DistributedArrays
using LinearAlgebra

# [CO] This can't be called here, otherwise can't call ACEfit from another 
# package that doesn't have Distributed, DistributedArrays in its dependencies
# note sure yet how to resolve this 
# @everywhere using Distributed, DistributedArrays

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
# wcw: fix these hacks
function get_basis(m::Any)
    return m
end
function set_params!(m::Any, x::Any)
    return m
end

# TODO: Maybe we can provide more fine-tuned execution models 
#    struct Distributed parameters end  struct Serial ... end and so forth 
#    rather than those symbols :serial, :mt, :dist


function llsq!(model, data::AbstractVector, par = :serial; solver = QR())
   basis = get_basis(model) # should return an ACEBasis 
   θ, errors = llsq(basis, data, par; solver = solver)
   set_params!(model, θ)
   return θ, errors
end

function llsq(basis, data::AbstractVector, par = :serial; solver = QR())
   if par == :serial
      _iterate = siterate
      A, y, w = asm_llsq(basis, data, _iterate)
   elseif par == :mt
      _iterate = titerate
      A, y = asm_llsq(basis, data, _iterate)
   elseif par == :dist
      _iterate = siterate
      A, y = asm_llsq_dist(basis, data, _iterate)
   else 
      error("unknown assembly type")
   end

   coef = solve_llsq(solver, Diagonal(w)*A, w.*y)

   config_errors = error_llsq(data, A*coef-y)

   return coef, config_errors

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
   _, Nobs = get_lsq_indices(data)

   # allocate - maybe we need to check somewhere what the 
   #            eltypes of A, y should be? real, complex, precision?
   A = zeros(Nobs, length(basis))
   Y = zeros(Nobs)
   W = zeros(Nobs)
   
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
         # TODO: make this an input parameter eventually
         if hasproperty(o, :E) || hasproperty(o, :V)
            w = w ./ sqrt(length(dat.config))
         end
         inds = idx:idx+length(y)-1
         Y[inds] .= y[:]
         W[inds] .= w*ones(length(y))
         for ib = 1:length(basis) 
            ovec = vec_obs(oB[ib])
            A[inds, ib] .= ovec[:]
         end
         idx += length(y)
      end
      return nothing 
   end

   _iterate(asm_lsq_inner, data)

   return A, Y, W
end

function asm_llsq_dist(basis, data, _iterate)

   # for now, all data is sent to all workers, should revisit
   _, Nobs = get_lsq_indices(data)

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

   # send data to main process for use with any solver
   #A = convert(Array, A)
   #Y = convert(Vector, Y)

   return A, Y
end

function error_llsq(data, errors)

   config_types = String[]
   config_counts = Dict("set"=>Dict("E"=>0,   "F"=>0,   "V"=>0))
   config_errors = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   for dat in data
       if !(dat.configtype in config_types)
          push!(config_types, dat.configtype)
          merge!(config_counts, Dict(dat.configtype=>Dict("E"=>0,   "F"=>0,   "V"=>0)))
          merge!(config_errors, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
       end
   end

   i = 1
   for dat in data
      for o in observations(dat)
         obs_len = length(vec_obs(o))
         obs_errors = errors[i:i+obs_len-1]
         if hasproperty(o, :E) || hasproperty(o, :V)
            obs_errors = obs_errors ./ length(dat.config)
         end
         obs_error = sum(obs_errors.^2)
         if hasproperty(o, :E)
            config_counts["set"]["E"] += obs_len
            config_errors["set"]["E"] += obs_error
            config_counts[dat.configtype]["E"] += obs_len
            config_errors[dat.configtype]["E"] += obs_error
         elseif hasproperty(o, :F)
            config_counts["set"]["F"] += obs_len
            config_errors["set"]["F"] += obs_error
            config_counts[dat.configtype]["F"] += obs_len
            config_errors[dat.configtype]["F"] += obs_error
         elseif hasproperty(o, :V)
            config_counts["set"]["V"] += obs_len
            config_errors["set"]["V"] += obs_error
            config_counts[dat.configtype]["V"] += obs_len
            config_errors[dat.configtype]["V"] += obs_error
         else
            println("something is wrong")
         end
         i += obs_len
      end
   end

   for i in keys(config_errors)
      for j in keys(config_errors[i])
         config_errors[i][j] /= config_counts[i][j]
         config_errors[i][j] = sqrt(config_errors[i][j])
      end

   end

   return config_errors

end
