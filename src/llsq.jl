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
      A, y, w = llsq_assemble(basis, data, _iterate)
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

function llsq_assemble(basis, data, _iterate)

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

function process_xyz_dict(xyz_dict, v_ref, energy_key, force_key, virial_key, weights, basis)
    atoms = JuLIP._extxyz_dict_to_atoms(xyz_dict)
    dat = _atoms_to_data(atoms, v_ref, weights, energy_key, force_key, virial_key)
    nrows = 0
    for o in observations(dat)
        nrows += length(vec_obs(o))
    end
    a = zeros(nrows, length(basis))
    y = zeros(nrows)
    w = zeros(nrows)
    row = 0
    for o in observations(dat)
        bobs = basis_obs(typeof(o), basis, dat.config)
        yobs = vec_obs(o)
        wobs = get_weight(o)
        if hasproperty(o, :E) || hasproperty(o, :V)
            wobs = wobs ./ sqrt(length(dat.config))
        end
        for j=1:length(basis)
            aobs = vec_obs(bobs[j])  # todo: must be a more elegant approach
            for i in 1:length(yobs)
                a[row+i,j] = wobs*aobs[i]
            end
        end
        for i=1:length(yobs)
            y[row+i] = wobs*yobs[i]
            w[row+i] = wobs
        end
        row = row + length(yobs)
    end
    return a, y, w
end
