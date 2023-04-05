using Distributed
using LinearAlgebra
using ParallelDataTransfer
using ProgressMeter
using SharedArrays

function linear_fit(data::AbstractVector, basis, solver=QR(); P=nothing)
    @info "Entering linear_assemble"
    A, Y, W = linear_assemble(data, basis)
    @info "After linear_assemble"
    flush(stdout); flush(stderr)
    lmul!(Diagonal(W),A)
    Y = W.*Y
    !isnothing(P) && (A = A*pinv(P))
    GC.gc()
    @info "Entering linear_solve"
    results = linear_solve(solver, A, Y)
    C = results["C"]
    @info "After linear_solve"
    if !isnothing(P)
        A = A*P
        C = pinv(P)*C
        # TODO: deapply preconditioner to committee
    end
    lmul!(inv(Diagonal(W)),A)
    Y = (1.0./W).*Y
    fit = Dict{String,Any}("C" => C)
    haskey(results, "committee") && (fit["committee"] = results["committee"])
    return fit
end

struct DataPacket{T <: AbstractData}
   rows::UnitRange
   data::T
end

Base.length(d::DataPacket) = count_observations(d.data)

function linear_assemble(data, basis)
   @info "Assembling linear problem."
   rows = Array{UnitRange}(undef,length(data))
   rows[1] = 1:count_observations(data[1])
   for i in 2:length(data)
      rows[i] = rows[i-1][end] .+ (1:count_observations(data[i]))
   end
   packets = DataPacket.(rows, data)
   sort!(packets, by=length, rev=true)
   @info "  - Creating feature matrix with size ($(rows[end][end]), $(length(basis)))."
   A = SharedArray(zeros(rows[end][end],length(basis)))
   Y = SharedArray(zeros(size(A,1)))
   W = SharedArray(zeros(size(A,1)))
   @info "  - Beginning assembly with processor count:  $(nprocs())."
   (nprocs() > 1) && sendto(workers(), basis=basis)
   @showprogress pmap(packets) do packet
      rows, data = packet.rows, packet.data
      A[rows,:] .= feature_matrix(data, basis)
      Y[rows] .= target_vector(data)
      W[rows] .= weight_vector(data)
   end
   @info "  - Assembly completed."
   flush(stdout); flush(stderr)
   return Array(A), Array(Y), Array(W)
end
