

module ObsExamples

   using JuLIP, ACEfit 

   import JuLIP: Atoms, energy, forces, 
                 JVec, mat, vecs 

   import ACEfit: Dat, eval_obs, vec_obs, devec_obs

   # export ObsPotentialEnergy, ObsForces 

   struct ObsPotentialEnergy{T} 
      E::T
   end

   # evaluating an observation type on a model - 
   # here we assume implicitly that `at = dat.config::Atoms` and that 
   # `energy(model, at)` has been implemented. Different models could either 
   # overload just `energy(model, at)` or overload ``
   eval_obs(::Type{TOBS}, model, config) where {TOBS <: ObsPotentialEnergy} = 
         TOBS( energy(model, config) )

   # now given an observation we need to convert it 
   vec_obs(obs::ObsPotentialEnergy) = [ obs.E ]
   
   function devec_obs(obs::TOBS, x::AbstractVector) where {TOBS <: ObsPotentialEnergy}
      @assert length(x) == 1
      return TOBS(x[1])
   end


   struct ObsForces{T}
      F::Vector{JVec{T}}
   end

   eval_obs(::Type{TOBS}, model, cfg) where {TOBS <: ObsForces} = 
         TOBS( forces(model, cfg) )

   # converts [ [F11 F12 F13] [F21 F22 F23] ... ] into 
   #          [ F11, F12, F13, F21, F22, F23, ... ]
   vec_obs(obs::ObsForces) = mat(obs.F)[:]
   
   # and the converse; collect just gets rid of the re-interpret business... 
   devec_obs(obs::TOBS, x::AbstractVector) where {TOBS <: ObsForces} = 
         TOBS(collect(vecs(x)))

end