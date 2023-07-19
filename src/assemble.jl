using Distributed
using Folds
using ParallelDataTransfer
using ProgressMeter
using SharedArrays

struct DataPacket{T <: AbstractData}
    rows::UnitRange
    data::T
end

Base.length(d::DataPacket) = count_observations(d.data)

"""
Assemble feature matrix, target vector, and weight vector for given data and basis.
"""
function assemble(data::AbstractVector{<:AbstractData}, basis; mode=:threaded)
    @info "Assembling linear problem."
    rows = Array{UnitRange}(undef, length(data))  # row ranges for each element of data
    rows[1] = 1:count_observations(data[1])
    for i in 2:length(data)
        rows[i] = rows[i - 1][end] .+ (1:count_observations(data[i]))
    end
    packets = DataPacket.(rows, data)
    sort!(packets, by = length, rev = true)
    @info "  - Creating feature matrix with size ($(rows[end][end]), $(length(basis)))."
    A = SharedArray(zeros(rows[end][end], length(basis)))
    Y = SharedArray(zeros(size(A, 1)))
    W = SharedArray(zeros(size(A, 1)))
    if mode == :serial
        @info "  - Beginning serial assembly."
    elseif mode == :threaded
        @info "  - Beginning threaded assembly with $(Threads.nthreads()) threads."
        map = Folds.map
    elseif mode == :distributed
        @info "  - Beginning distributed assembly with $(nprocs()) processes."
        map = pmap
        (nprocs() > 1) && sendto(workers(), basis = basis)
    end
    progress = Progress(length(data))
    map(packets) do p
        A[p.rows,:] .= feature_matrix(p.data, basis)
        Y[p.rows] .= target_vector(p.data)
        W[p.rows] .= weight_vector(p.data)
        next!(progress)
        GC.gc()
    end
    @info "  - Assembly completed."
    return Array(A), Array(Y), Array(W)
end
