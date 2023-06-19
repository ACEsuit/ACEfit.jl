module ACEfit


# CO: removed this since couldn't remove PyCall from the dependencies 
#     and no time to figure this out right now, sorry. 
#
# using Requires
# function __init__()
#     @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" include("solvers_pycall.jl")
# end


include("solvers_pycall.jl")

include("bayesianlinear.jl")
include("data.jl")
include("assemble.jl")
include("solvers.jl")
include("fit.jl")

end
