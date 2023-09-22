using Distributed
using ProgressMeter


"""
    assemble(data::AbstractArray, basis; kwargs...)

Assemble feature matrix and target vector for given data and basis.
`kwargs` are used to control `feature_matrix`, `target_vector` and
`weight_vector` calculations.
"""
function assemble(data::AbstractArray, basis; batch_size=1, kwargs...)
    W = Threads.@spawn ACEfit.assemble_weights(data; kwargs...)
    raw_data = @showprogress desc="Assembly progress:" pmap( data; batch_size=batch_size ) do d
        A = ACEfit.feature_matrix(d, basis; kwargs...)
        Y = ACEfit.target_vector(d; kwargs...)
        (A, Y)
    end
    A = [ a[1] for a in raw_data ]
    Y = [ a[2] for a in raw_data ]

    A_final = reduce(vcat, A)
    Y_final = reduce(vcat, Y)
    return A_final, Y_final, fetch(W)
end

"""
    assemble_weights(data::AbstractArray; kwargs...)

Assemble full weight vector for vector of data elements.
`kwargs` are used to give extra commands for `weight_vector calculation`.
"""
function assemble_weights(data::AbstractArray; kwargs...)
    w = map( data ) do d
        ACEfit.weight_vector(d; kwargs...)
    end
    return reduce(vcat, w)
end