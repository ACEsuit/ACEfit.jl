using LinearAlgebra

function linear_fit(data::AbstractVector{<:AbstractData},
                    basis,
                    solver = QR();
                    P = nothing)
    @info "Assembling linear problem."
    A, Y, W = assemble(data, basis)
    @info "Finished assembling."
    flush(stdout)
    flush(stderr)
    lmul!(Diagonal(W), A)
    Y = W .* Y
    !isnothing(P) && (A = A * pinv(P))
    GC.gc()
    @info "Solving linear problem."
    results = solve(solver, A, Y)
    C = results["C"]
    @info "Finished solving."
    if !isnothing(P)
        A = A * P
        C = pinv(P) * C
        # TODO: deapply preconditioner to committee
    end
    lmul!(inv(Diagonal(W)), A)
    Y = (1.0 ./ W) .* Y
    fit = Dict{String, Any}("C" => C)
    haskey(results, "committee") && (fit["committee"] = results["committee"])
    return fit
end
