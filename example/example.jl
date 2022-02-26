
#
# This script is intended to see whether we have all the interface we need 
# whether we need additional structure, and see how things would look like 
# from a developer and user perspective. 
#

using JuLIP, ACE, ACEfit, ACEatoms

using ACEfit: eval_obs, vec_obs

# load some example observation types 
include("obsexamples.jl")

OE = ObsExamples.ObsPotentialEnergy
OF = ObsExamples.ObsForces

## [1] create a dataset 

function create_dataset(N)
   sw = StillingerWeber()
   data = Dat[] 
   for _=1:N 
      at = bulk(:Si, cubic=true) * 3
      rattle!(at, 0.1)
      dat = Dat(at, "bulk", 
                [ eval_obs(OE, sw, at), 
                  eval_obs(OF, sw, at) ] )
      push!(data, dat)
   end
   return data 
end

data = create_dataset(10)

## [2] create a model, in this case a linear model represented by a basis
Bsel = SimpleSparseBasis(3, 8)
B1p = ACE.Utils.RnYlm_1pbasis(r0 = rnn(:Si), rcut = cutoff(StillingerWeber()))
basis = ACE.SymmetricBasis(ACE.Invariant(), B1p, Bsel)
model = ACEatoms.ACESitePotential(Dict(:Si => ACE.LinearACEModel(basis)))

# we now need to explain how to evaluate the observations on a basis; so we 
# need to implement `energy` and `forces`. But this is already done for us 
# in ACEatoms ... 

## [3] assemble a loss function 

loss(model, dat::Dat) = sum(  
      sum(abs2, vec_obs(o) - vec_obs( eval_obs(typeof(o), model, dat.config) ))
      for o in ACEfit.observations(dat) )

function Loss(model, data) 
   L = 0.0 
   ACEfit.siterate( (i, dat) -> (L += loss(model, dat)), data )
   return L 
end 

Loss(model, data) 

# note that the gradient of loss would always be obtained via AD, we never 
# do this by hand. 


## [4] assemble a lsq system
# this is a little more challenging and still needs a few fixes...
# at the moment, I'm "hacking" it so it will do something sensible but it 
# is not a good solution yet.

import ACEbase

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# temporary hack to get get_basis and set_params! into ACEfit 
ACEfit.get_basis(m::typeof(model)) = ACEatoms.basis(m)
ACEfit.eval(:(function set_params! end))
ACEfit.set_params!(m::typeof(model), args...) = ACE.set_params!(m, args...)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ACEfit.llsq!(model, data, :serial; solver = ACEfit.QR())

# and we can confirm that this actually makes the loss small :)
Loss(model, data)

