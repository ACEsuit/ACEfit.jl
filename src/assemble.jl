using Distributed
using ParallelDataTransfer
using ProgressMeter
using SharedArrays

"""
    basis_size(model)

Return the length of the basis, assuming that `model` is a linear model, or 
when interpreted as a linear model. The returned integer must match the 
size of the feature matrix that will be assembled for the given model.    

It defaults to `Base.length` but can be overloaded if needed.  
"""
basis_size(model) = Base.length(model) 

struct DataPacket{T <: AbstractData}
    rows::UnitRange
    data::T
end

Base.length(d::DataPacket) = count_observations(d.data)

"""
Assemble feature matrix and target vector for given data and basis.
"""
function assemble(data::AbstractVector{<:AbstractData}, basis)
    @info "Assembling linear problem."
    rows = Array{UnitRange}(undef, length(data))  # row ranges for each element of data
    rows[1] = 1:count_observations(data[1])
    for i in 2:length(data)
        rows[i] = rows[i - 1][end] .+ (1:count_observations(data[i]))
    end
    packets = DataPacket.(rows, data)
    sort!(packets, by = length, rev = true)
    (nprocs() > 1) && sendto(workers(), basis = basis)
    @info "  - Creating feature matrix with size ($(rows[end][end]), $(basis_size(basis)))."
    A = SharedArray(zeros(rows[end][end], basis_size(basis)))
    Y = SharedArray(zeros(size(A, 1)))
    @info "  - Beginning assembly with processor count:  $(nprocs())."
    @showprogress pmap(packets) do p
        A[p.rows, :] .= feature_matrix(p.data, basis)
        Y[p.rows] .= target_vector(p.data)
        GC.gc()
    end
    @info "  - Assembly completed."
    return Array(A), Array(Y), assemble_weights(data)
end

"""
Assemble full weight vector for vector of data elements.
"""
function assemble_weights(data::AbstractVector{<:AbstractData})
    @info "Assembling full weight vector."
    rows = Array{UnitRange}(undef, length(data))  # row ranges for each element of data
    rows[1] = 1:count_observations(data[1])
    for i in 2:length(data)
        rows[i] = rows[i - 1][end] .+ (1:count_observations(data[i]))
    end
    packets = DataPacket.(rows, data)
    sort!(packets, by = length, rev = true)
    W = SharedArray(zeros(rows[end][end]))
    @showprogress pmap(packets) do p
        W[p.rows] .= weight_vector(p.data)
    end
    return Array(W)
end
