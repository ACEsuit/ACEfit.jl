module ACEfit

# data management 
include("data.jl")

include("datautils.jl")

# iteration utilities over training data 
include("iterate.jl")

# managing and visualising training and test errors 

# loss functions and nonlinear solvers 
# should this stay in ACEflux? 

# linear least squares assembly for linear models
include("llsq.jl")

# linear least squares solvers for linear models
include("linearsolvers.jl")


end