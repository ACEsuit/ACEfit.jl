module ACEfit

using Requires

function __init__()
    @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" include("solvers_pycall.jl")
end

include("bayesianlinear.jl")
include("data.jl")
include("assemble.jl")
include("solvers.jl")
include("solvers_pycall.jl")
include("fit.jl")

end
