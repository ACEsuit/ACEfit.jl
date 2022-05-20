using Distributed, DistributedArrays
using LinearAlgebra

# [CO] This can't be called here, otherwise can't call ACEfit from another 
# package that doesn't have Distributed, DistributedArrays in its dependencies
# note sure yet how to resolve this 
# @everywhere using Distributed, DistributedArrays

using ACEbase: ACEBasis
using JuLIP
using ExtXYZ

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

function llsq!(model, data::AbstractVector, Vref, par = :serial; solver = QR())
   basis = get_basis(model) # should return an ACEBasis 
   IP, errors = llsq(basis, data, Vref, par; solver = solver)
   #set_params!(model, θ)
   return IP, errors
end

function llsq(basis, data::AbstractVector, Vref, par = :serial; solver = QR())

   if par == :serial
      _iterate = siterate
      A, y, w = asm_llsq(basis, data, _iterate)
      c = solve_llsq(solver, A, y)
      config_errors = error_llsq(data, (A*c)./w, y./w)
      # wcw, this is the old way, should revisit
      IP = JuLIP.MLIPs.combine(basis, c)
      if Vref != nothing
         IP = JuLIP.MLIPs.SumIP(Vref, IP)
      end
      return IP, config_errors
   elseif par == :mt
      _iterate = titerate
      A, y = asm_llsq(basis, data, _iterate)
   elseif par == :dist
      _iterate = siterate
      A, y, w = asm_llsq_dist(basis, data, _iterate)
   else 
      error("unknown assembly type")
   end
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

   println("Creating design matrix with size (", Nobs, ", ", length(basis), ")")
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

function _atoms_to_data(atoms, params)
    # create one body potential
    # conversion prevents error in OneBody
    params["e0"] = convert(Dict{String, Any}, params["e0"])
    Vref = OneBody(params["e0"])
    energy_key = params["data"]["energy_key"]
    force_key = params["data"]["force_key"]
    virial_key = params["data"]["virial_key"]
    weights = params["weights"]

    energy = nothing
    forces = nothing
    virial = nothing
    config_type = "default"
    for key in keys(atoms.data)
        if lowercase(key)=="config_type"; config_type=atoms.data[key].data; end
    end
    for key in keys(atoms.data)
        if lowercase(key) == lowercase(energy_key)
            w = (config_type in keys(weights)) ? weights[config_type]["E"] : weights["default"]["E"]
            energy_ref = JuLIP.energy(Vref, atoms)
            energy = atoms.data[key].data - energy_ref
            energy = ObsPotentialEnergy(energy, w, energy_ref)
        end
        if lowercase(key) == lowercase(force_key)
            w = (config_type in keys(weights)) ? weights[config_type]["F"] : weights["default"]["F"]
            forces = ObsForces(atoms.data[key].data[:], w)
        end
        if lowercase(key) == lowercase(virial_key)
            w = (config_type in keys(weights)) ? weights[config_type]["V"] : weights["default"]["V"]
            m = SMatrix{3,3}(atoms.data[key].data)
            virial = ObsVirial(m, w)
        end
    end
    obs = Any[energy]
    if !isnothing(forces)
        push!(obs, forces)
    end
    if !isnothing(virial)
        insert!(obs, 1, virial)
    end
    return ACEfit.Dat(atoms, config_type, obs)
end

function assemble_dist_new!(A, Y, W, params, basis)

    row = 1
    localrows = localindices(Y)[1]
    function asm_lsq_inner(dat)
       for o in observations(dat)
          y = vec_obs(o)
          if row>localrows[end] || (row-1+length(y))<localrows[1]
             row += length(y)    # increment and move on, because
             continue            # there is no local work to do
          end
          oB = basis_obs(typeof(o), basis, dat.config)
          w = get_weight(o)
          # TODO: make this an input parameter eventually
          if hasproperty(o, :E) || hasproperty(o, :V)
             w = w ./ sqrt(length(dat.config))
          end
          # fill rows of Y and A
          for i in 1:length(y)
             if row>=localrows[1] && row<=localrows[end]
                localpart(Y)[row-localrows[1]+1] = w*y[i]
                localpart(W)[row-localrows[1]+1] = w
                for ib = 1:length(basis)
                   localpart(A)[row-localrows[1]+1, ib] = w*vec_obs(oB[ib])[i]
                end
             end
             row += 1
          end
       end
       return nothing
    end

    for dict in ExtXYZ.iread_frames(params["data"]["fname"])
        atoms = JuLIP._extxyz_dict_to_atoms(dict)
        data = _atoms_to_data(atoms, params)
        asm_lsq_inner(data)
    end
 
    return nothing
end

function assemble_dist!(A, Y, W, data, basis)

   row = 1
   localrows = localindices(Y)[1]
   function asm_lsq_inner(dat)
      for o in observations(dat)
         y = vec_obs(o)
         if row>localrows[end] || (row-1+length(y))<localrows[1]
            row += length(y)    # increment and move on, because
            continue            # there is no local work to do
         end
         oB = basis_obs(typeof(o), basis, dat.config)
         w = get_weight(o)
         # TODO: make this an input parameter eventually
         if hasproperty(o, :E) || hasproperty(o, :V)
            w = w ./ sqrt(length(dat.config))
         end
         # fill rows of Y and A
         for i in 1:length(y)
            if row>=localrows[1] && row<=localrows[end]
               localpart(Y)[row-localrows[1]+1] = w*y[i]
               localpart(W)[row-localrows[1]+1] = w
               for ib = 1:length(basis)
                  localpart(A)[row-localrows[1]+1, ib] = w*vec_obs(oB[ib])[i]
               end
            end
            row += 1
         end
      end
      return nothing
   end

   for dat in data
      asm_lsq_inner(dat)
   end

   return nothing
end

function asm_llsq_dist(basis, data, _iterate)

   println("Creating distributed design matrix with size (", Nobs, ", ", length(basis), ")")
   A = dzeros(Nobs, length(basis))
   Y = dzeros(Nobs)
   W = dzeros(Nobs)

   # send data to main process for use with any solver
   #A = convert(Array, A)
   #Y = convert(Vector, Y)
   #W = convert(Vector, W)

   return A, Y, W
end

function error_llsq(data, approx, exact)

   errors = approx - exact
   config_types = String[]
   config_counts = Dict("set"=>Dict("E"=>0, "F"=>0, "V"=>0))
   config_errors = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   config_norms = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   for dat in data
       if !(dat.configtype in config_types)
          push!(config_types, dat.configtype)
          merge!(config_counts, Dict(dat.configtype=>Dict("E"=>0,   "F"=>0,   "V"=>0)))
          merge!(config_errors, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_norms, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
       end
   end

   i = 1
   for dat in data
      for o in observations(dat)
         obs_len = length(vec_obs(o))
         obs_errors = errors[i:i+obs_len-1]
         obs_values = exact[i:i+obs_len-1]
         # TODO: we store the ref energy because it is used for the relrmse
         #       calculation ... but does it make sense to use the total energy?
         if hasproperty(o, :E)
            obs_values = obs_values .+ o.E_ref
         end
         if hasproperty(o, :E) || hasproperty(o, :V)
            obs_errors = obs_errors ./ length(dat.config)
            obs_values = obs_values ./ length(dat.config)
         end
         obs_error = sum(obs_errors.^2)
         obs_norm = sum(obs_values.^2)
         if hasproperty(o, :E)
            config_counts["set"]["E"] += obs_len
            config_errors["set"]["E"] += obs_error
            config_norms["set"]["E"] += obs_norm
            config_counts[dat.configtype]["E"] += obs_len
            config_errors[dat.configtype]["E"] += obs_error
            config_norms[dat.configtype]["E"] += obs_norm
         elseif hasproperty(o, :F)
            config_counts["set"]["F"] += obs_len
            config_errors["set"]["F"] += obs_error
            config_norms["set"]["F"] += obs_norm
            config_counts[dat.configtype]["F"] += obs_len
            config_errors[dat.configtype]["F"] += obs_error
            config_norms[dat.configtype]["F"] += obs_norm
         elseif hasproperty(o, :V)
            config_counts["set"]["V"] += obs_len
            config_errors["set"]["V"] += obs_error
            config_norms["set"]["V"] += obs_norm
            config_counts[dat.configtype]["V"] += obs_len
            config_errors[dat.configtype]["V"] += obs_error
            config_norms[dat.configtype]["V"] += obs_norm
         else
            println("something is wrong")
         end
         i += obs_len
      end
   end

   for i in keys(config_errors)
      for j in keys(config_errors[i])
         config_errors[i][j] = sqrt(config_errors[i][j] / config_counts[i][j])
         config_norms[i][j] = sqrt(config_norms[i][j] / config_counts[i][j])
         config_norms[i][j] = config_errors[i][j] / config_norms[i][j]
      end
   end

   return Dict("rmse"=>config_errors, "relrmse"=>config_norms)

end

function error_llsq_new(params, approx, exact)

   errors = approx - exact
   config_types = String[]
   config_counts = Dict("set"=>Dict("E"=>0, "F"=>0, "V"=>0))
   config_errors = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   config_norms = Dict("set"=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0))
   for dict in ExtXYZ.iread_frames(params["data"]["fname"])
       atoms = JuLIP._extxyz_dict_to_atoms(dict)
       dat = _atoms_to_data(atoms, params)
       if !(dat.configtype in config_types)
          push!(config_types, dat.configtype)
          merge!(config_counts, Dict(dat.configtype=>Dict("E"=>0,   "F"=>0,   "V"=>0)))
          merge!(config_errors, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
          merge!(config_norms, Dict(dat.configtype=>Dict("E"=>0.0, "F"=>0.0, "V"=>0.0)))
       end
    end

   i = 1
   for dict in ExtXYZ.iread_frames(params["data"]["fname"])
      atoms = JuLIP._extxyz_dict_to_atoms(dict)
      dat = _atoms_to_data(atoms, params)
      for o in observations(dat)
         obs_len = length(vec_obs(o))
         obs_errors = errors[i:i+obs_len-1]
         obs_values = exact[i:i+obs_len-1]
         # TODO: we store the ref energy because it is used for the relrmse
         #       calculation ... but does it make sense to use the total energy?
         if hasproperty(o, :E)
            obs_values = obs_values .+ o.E_ref
         end
         if hasproperty(o, :E) || hasproperty(o, :V)
            obs_errors = obs_errors ./ length(dat.config)
            obs_values = obs_values ./ length(dat.config)
         end
         obs_error = sum(obs_errors.^2)
         obs_norm = sum(obs_values.^2)
         if hasproperty(o, :E)
            config_counts["set"]["E"] += obs_len
            config_errors["set"]["E"] += obs_error
            config_norms["set"]["E"] += obs_norm
            config_counts[dat.configtype]["E"] += obs_len
            config_errors[dat.configtype]["E"] += obs_error
            config_norms[dat.configtype]["E"] += obs_norm
         elseif hasproperty(o, :F)
            config_counts["set"]["F"] += obs_len
            config_errors["set"]["F"] += obs_error
            config_norms["set"]["F"] += obs_norm
            config_counts[dat.configtype]["F"] += obs_len
            config_errors[dat.configtype]["F"] += obs_error
            config_norms[dat.configtype]["F"] += obs_norm
         elseif hasproperty(o, :V)
            config_counts["set"]["V"] += obs_len
            config_errors["set"]["V"] += obs_error
            config_norms["set"]["V"] += obs_norm
            config_counts[dat.configtype]["V"] += obs_len
            config_errors[dat.configtype]["V"] += obs_error
            config_norms[dat.configtype]["V"] += obs_norm
         else
            println("something is wrong")
         end
         i += obs_len
      end
   end

   for i in keys(config_errors)
      for j in keys(config_errors[i])
         config_errors[i][j] = sqrt(config_errors[i][j] / config_counts[i][j])
         config_norms[i][j] = sqrt(config_norms[i][j] / config_counts[i][j])
         config_norms[i][j] = config_errors[i][j] / config_norms[i][j]
      end
   end

   return Dict("rmse"=>config_errors, "relrmse"=>config_norms)

end
