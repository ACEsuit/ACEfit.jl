module ACEfit

# data management 
include("data.jl")

include("datautils.jl")

# iteration utilities over training data 
include("iterate.jl")

# managing and visualising training and test errors 

# loss functions and nonlinear solvers 
# should this stay in ACEflux? 

# this stuff should probably go elsewhere
include("obs.jl")

# linear least squares assembly for linear models
include("llsq.jl")

# linear least squares solvers for linear models
include("linearsolvers.jl")

include("helper.jl")

end
