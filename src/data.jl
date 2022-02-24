
import Base: ==, convert

import ACEbase: read_dict, write_dict

export Dat, eval_obs, observations, vec_obs, devec_obs 


# -----------------------------------------------------------------
# registry of observation types 

# TODO: not sure yet that we really need this - leave it for now but revisit.

const _obstypes = Dict{String, Any}()

function register_obstype!(key::String, ObsType; force = false)
   if !forces && haskey(_obstypes, key)
      error("""This observation type key is already registered. To overwrite it, 
               use `register_obstype!(...; force=true)`.""")
   end 
   _obstype[key] = ObsType
   return nothing 
end

function empty_obstypes!()
   empty!(_obstypes)
   return nothing 
end




# -----------------------------------------------------------------
# basic data structure to manage training data 


"""
`Dat`: store one configuration (input, e.g., structure, state, ...)  that can 
have multiple observations attached to it. Fields:
- `config::Any` : the structure 
- `configtype::String` : Each `dat::Dat` belongs to a group identified by a string `dat.configtype` to allow filtering, and grouping. 
- `obs::Vector{Any}`  : list of observations 
- `meta::Dict{String, Any}` : any additional meta information that we may want to attach to this data point; this needs to be raw json.
"""
mutable struct Dat
   config                             # configuration
   configtype::String                 # group identifier
   obs::Vector{Any}                  # list of observations
   meta::Dict{String, Any}           # anything else ... 
end

Dat(config, configtype="", obs=Any[]) = 
      Dat(config, configtype, obs, Dict{String, Any}())

==(d1::Dat, d2::Dat) = (
      (d1.config == d2.config) && 
      (d1.configtype == d2.configtype) && 
      (d1.obs == d2.obs) &&
      (d1.meta == d2.meta) 
   )


write_dict(d::Dat) =
   Dict("__id__" => "ACEfit_Dat",
         "config" => write_dict(d.config),
         "configtype" => d.configtype,
         "obs" => write_dict.(d.obs),
         "meta" => d.meta)


Dat(D::Dict) = Dat( read_dict(D["config"]), 
                    D["configtype"],
                    Vector{Any}(D["obs"]), 
                    Dict{String, Any}(D["meta"]) )

read_dict(::Val{:ACEfit_Dat}, D::Dict) = Dat(D)

observations(d::Dat) = d.obs 

cost(d::Dat) = 1


# -----------------------------------------------------------------
# Abstract Observations interface 
# at this point it is not clear that we need an abstract Observations 
# type hierarchy. Better to try and live without for now. 

# prototypes

"""
convert some real data, in some generic format, into a vector to be stored
in a `Dat` or Lsq system. E.g.,
```julia 
V = virial(...)::Matrix{Float64}
obsV = ObsVirial(V)
vec_obs(obsV::ObsVirial) = obsV.V[ [1,2,3,5,6,9] ]  # NB: V is symmetric
```
"""
function vec_obs end

"""
convert a Vector{T} to some real data, e.g.,
```julia 
x::Vector{Float64}
devec_obs(::Type{ObsVirial}, x) = [ x[1] x[2] x[3]; 
                                    x[2] x[4] x[5];
                                    x[3] x[5] x[6] ]
```
"""
function devec_obs end

"""
Evaluate a specific observation type: Given an observation `obs`, 
a model `model` and  a configuration `cfg = dat.config`, the call 
```julia 
eval_obs(obs, model, cfg)
```
must return the corresponding observation. For example, if 
`obs::ObsPotentialEnergy` and `cfg = at::Atoms`, and `model` is an interatomic 
potential, then 
```julia 
eval_obs(obs::ObsPotentialEnergy, model, cfg) = 
      ObsPotentialEnergy( energy(model, cfg) )
```
"""
function eval_obs end

