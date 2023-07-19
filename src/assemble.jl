using Distributed
using ParallelDataTransfer
using ProgressMeter
using SharedArrays
using ThreadedIterables
using ThreadsX

struct DataPacket{T <: AbstractData}
    rows::UnitRange
    data::T
end

Base.length(d::DataPacket) = count_observations(d.data)

"""
Assemble feature matrix, target vector, and weight vector for given data and basis.
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
    @info "  - Creating feature matrix with size ($(rows[end][end]), $(length(basis)))."
    A = SharedArray(zeros(rows[end][end], length(basis)))
    Y = SharedArray(zeros(size(A, 1)))
    W = SharedArray(zeros(size(A, 1)))
    @info "  - Beginning assembly with processor count:  $(nprocs())."
    @showprogress pmap(packets) do p
        A[p.rows, :] .= feature_matrix(p.data, basis)
        Y[p.rows] .= target_vector(p.data)
        W[p.rows] .= weight_vector(p.data)
        GC.gc()
    end
    @info "  - Assembly completed."
    return Array(A), Array(Y), Array(W)
end

function assemble_threadediterables(data::AbstractVector{<:AbstractData}, basis)
    @info "Assembling linear problem."
    rows = Array{UnitRange}(undef, length(data))  # row ranges for each element of data
    rows[1] = 1:count_observations(data[1])
    for i in 2:length(data)
        rows[i] = rows[i - 1][end] .+ (1:count_observations(data[i]))
    end
    packets = DataPacket.(rows, data)
    sort!(packets, by = length, rev = true)
    @info "  - Creating feature matrix with size ($(rows[end][end]), $(length(basis)))."
    A = Array(zeros(rows[end][end], length(basis)))
    Y = Array(zeros(size(A, 1)))
    W = Array(zeros(size(A, 1)))
    @info "  - Beginning assembly with thread count:  $(Threads.nthreads())."
    @time tmap(packets) do p
        A[p.rows, :] .= feature_matrix(p.data, basis)
        Y[p.rows] .= target_vector(p.data)
        W[p.rows] .= weight_vector(p.data)
    end
    @info "  - Assembly completed."
    return A, Y, W
end

function assemble_threadsx(data::AbstractVector{<:AbstractData}, basis)
    @info "Assembling linear problem."
    rows = Array{UnitRange}(undef, length(data))  # row ranges for each element of data
    rows[1] = 1:count_observations(data[1])
    for i in 2:length(data)
        rows[i] = rows[i - 1][end] .+ (1:count_observations(data[i]))
    end
    packets = DataPacket.(rows, data)
    sort!(packets, by = length, rev = true)
    @info "  - Creating feature matrix with size ($(rows[end][end]), $(length(basis)))."
    A = Array(zeros(rows[end][end], length(basis)))
    Y = Array(zeros(size(A, 1)))
    W = Array(zeros(size(A, 1)))
    @info "  - Beginning assembly with thread count:  $(Threads.nthreads())."
    @time ThreadsX.map(packets) do p
        A[p.rows, :] .= feature_matrix(p.data, basis)
        Y[p.rows] .= target_vector(p.data)
        W[p.rows] .= weight_vector(p.data)
    end
    @info "  - Assembly completed."
    return A, Y, W
end

function assemble_threadsx_v2(data::AbstractVector{<:AbstractData}, basis)
    @info "Assembling linear problem."
    rows = Array{UnitRange}(undef, length(data))  # row ranges for each element of data
    rows[1] = 1:count_observations(data[1])
    for i in 2:length(data)
        rows[i] = rows[i - 1][end] .+ (1:count_observations(data[i]))
    end
    packets = DataPacket.(rows, data)
    sort!(packets, by = length, rev = true)
    @info "  - Creating feature matrix with size ($(rows[end][end]), $(length(basis)))."
    A = Array(zeros(rows[end][end], length(basis)))
    Y = Array(zeros(size(A, 1)))
    W = Array(zeros(size(A, 1)))
    @info "  - Beginning assembly with thread count:  $(Threads.nthreads())."
    @time ThreadsX.map!(similar(packets), packets) do p
        A[p.rows, :] .= feature_matrix(p.data, basis)
        Y[p.rows] .= target_vector(p.data)
        W[p.rows] .= weight_vector(p.data)
    end
    @info "  - Assembly completed."
    return A, Y, W
end
