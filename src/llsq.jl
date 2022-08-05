using LinearAlgebra
using ACEbase: ACEBasis
using JuLIP
using ExtXYZ

# TODO: will likely be useful to return to this more abstract approach eventually.
#       requires defining get_basis and set_params
#function llsq!(model, data::AbstractVector, Vref, par = :serial; solver = QR())
#   basis = get_basis(model) # should return an ACEBasis 
#   IP, errors = llsq(basis, data, Vref, par; solver = solver)
#   set_params!(model, θ)
#   return IP, errors
#end

function llsq(basis, data::AbstractVector, Vref, par = :serial; solver = QR())
   if par == :serial
      _iterate = siterate
      A, y, w = assemble_llsq(basis, data, _iterate)
      c = solve_llsq(solver, A, y)
      config_errors = error_llsq(data, (A*c)./w, y./w)
      IP = JuLIP.MLIPs.combine(basis, c)
      if Vref != nothing
         IP = JuLIP.MLIPs.SumIP(Vref, IP)
      end
      return IP, config_errors
   else 
      error("unknown assembly type")
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
