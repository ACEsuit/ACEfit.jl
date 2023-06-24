using Distributed
using ParallelDataTransfer
using ProgressMeter
using SharedArrays
using Base.Threads: nthreads, @threads

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
    end
    @info "  - Assembly completed."
    return Array(A), Array(Y), Array(W)
end


"""
Assemble feature matrix, target vector, and weight vector for given data and basis.
"""
function mt_assemble(data::AbstractVector{<:AbstractData}, basis)
    @info "Multi-threaded assembly of linear problem."
    rows = Array{UnitRange}(undef, length(data))  # row ranges for each element of data
    rows[1] = 1:count_observations(data[1])
    for i in 2:length(data)
        rows[i] = rows[i - 1][end] .+ (1:count_observations(data[i]))
    end
    packets = DataPacket.(rows, data)
    sort!(packets, by = length, rev = true)
    @info "  - Creating feature matrix with size ($(rows[end][end]), $(length(basis)))."
    A = zeros(rows[end][end], length(basis))
    Y = zeros(size(A, 1))
    W = zeros(size(A, 1))
    @info "  - Beginning assembly with $(Threads.nthreads()) threads."
    _lock = ReentrantLock()
    _prog = Progress(sum(length, rows))
    _prog_ctr = 0
    next = 1

    failed = Int[] 

    Threads.@threads for _i = 1:nthreads() 

        while next <= length(packets)
            # retrieve the next packet 
            if next > length(packets)
                break
            end
            lock(_lock)
            cur = next 
            next += 1
            unlock(_lock)
            if cur > length(packets)
                break 
            end 
            p = packets[cur]

            # assemble the corresponding data 
            try 
                Ap = feature_matrix(p.data, basis)
                Yp = target_vector(p.data)
                Wp = weight_vector(p.data)

                # write into global design matrix 
                lock(_lock)
                A[p.rows, :] .= Ap 
                Y[p.rows] .= Yp
                W[p.rows] .= Wp
                _prog_ctr += length(p.rows)
                ProgressMeter.update!(_prog, _prog_ctr)
                unlock(_lock)
            catch 
                @info("failed assembly: cur = $cur")
                push!(failed, cur)
            end
        end
        @info("thread $_i done")
    end
    @info "  - Assembly completed."
    @show failed 
    return Array(A), Array(Y), Array(W)
end
