
@doc raw"""

`ASP` : Active Set Pursuit solver

Solves the lasso optimization problem. 
```math 
\max_{y} \left( b^T y - \frac{1}{2} λ y^T y \right)
```
subject to
```math
   \|A^T y\|_{\infty} \leq 1.
```

### Constructor Keyword arguments 
```julia
ACEfit.ASP(; P = I, select = (:byerror, 1.0), 
            params...)
``` 

* `select` : Selection criterion for the final solution (required) 
    * `:final` : final solution (largest computed basis)
    * `(:byerror, q)` : solution with error within `q` times the minimum error 
       along the path; if training error is used and `q == 1.0`, then this is 
       equivalent to to `:final`.
    * `(:bysize, n)` : best solution with at most `n` non-zero features; if 
       training error is used, then it will be the solution with exactly `n` 
       non-zero features. 
* `P = I` : prior / regularizer (optional)

The remaining kwarguments to `ASP` are parameters for the ASP homotopy solver. 

* `actMax` : Maximum number of active constraints.
* `min_lambda` : Minimum value for `λ`.  (defaults to 0)
* `loglevel` : Logging level.
* `itnMax` : Maximum number of iterations.

### Extended syntax for `solve` 

```julia
solve(solver::ASP, A, y, Aval=A, yval=y)
```
* `A` : `m`-by-`n` design matrix. (required)
* `b` : `m`-vector. (required)
* `Aval = nothing` : `p`-by-`n` validation matrix (only for `validate` mode).
* `bval = nothing` : `p`- validation vector (only for `validate` mode).

If independent `Aval` and `yval` are provided (instead of detaults `A, y`), 
then the solver will use this separate validation set instead of the training
set to select the best solution along the model path. 
# """

struct ASP
    P
    select
    mode::Symbol
    tsvd::Bool
    nstore::Integer
    params
end

function ASP(; P = I, select, mode=:train, tsvd=false, nstore=100, params...)
    return ASP(P, select, mode, tsvd, nstore, params)
end

function solve(solver::ASP, A, y, Aval=A, yval=y)
    # Apply preconditioning
    AP = A / solver.P
    
    tracer = asp_homotopy(AP, y; solver.params...)

    q = length(tracer) 
    every = max(1, q ÷ solver.nstore)
    istore = unique([1:every:q; q])
    new_tracer = [ (solution = solver.P \ tracer[i][1], λ = tracer[i][2], σ = 0.0 ) 
                   for i in istore ]

    if solver.tsvd  # Post-processing if tsvd is true
        post = post_asp_tsvd(new_tracer, A, y, Aval, yval)
        new_post = [ (solution = p.θ, λ = p.λ, σ = p.σ) for p in post ]
    else
        new_post = new_tracer 
    end

    xs, in = select_solution(new_post, solver, Aval, yval)

    return Dict( "C" => xs, 
                "path" => new_post, 
                "nnzs" => length( (new_tracer[in][:solution]).nzind) )
end


function select_solution(tracer, solver, A, y)
    if solver.select == :final
      criterion = :final 
    else
      criterion, p = solver.select
    end
    
    if criterion == :final
        return tracer[end][:solution], length(tracer)
    end 

    if criterion == :byerror
        maxind = length(tracer)
        threshold = p 
    elseif criterion == :bysize
        maxind = findfirst(t -> length((t[:solution]).nzind) > p, 
                         tracer) - 1
        threshold = 1.0                          
    else 
        error("Unknown selection criterion: $criterion")
    end

    errors = [ norm(A * t[:solution] - y) for t in tracer[1:maxind] ]
    min_error = minimum(errors)
    for (i, error) in enumerate(errors)
        if error <= threshold * min_error
            return tracer[i][:solution], i
        end
    end

    error("selection failed for unknown reasons; please file an issue with a MWE to reproduce this error.")
end



using SparseArrays

function solve_tsvd(At, yt, Av, yv) 
   Ut, Σt, Vt = svd(At); zt = Ut' * yt
   Qv, Rv = qr(Av); zv = Matrix(Qv)' * yv
   @assert issorted(Σt, rev=true)

   Rv_Vt = Rv * Vt

   θv = zeros(size(Av, 2))
   θv[1] = zt[1] / Σt[1] 
   rv = Rv_Vt[:, 1] * θv[1] - zv 

   tsvd_errs = Float64[] 
   push!(tsvd_errs, norm(rv))

   for k = 2:length(Σt)
      θv[k] = zt[k] / Σt[k]
      rv += Rv_Vt[:, k] * θv[k]
      push!(tsvd_errs, norm(rv))
   end

   imin = argmin(tsvd_errs)
   θv[imin+1:end] .= 0
   return Vt * θv, Σt[imin]
end

function post_asp_tsvd(path, At, yt, Av, yv) 
   Qt, Rt = qr(At); zt = Matrix(Qt)' * yt
   Qv, Rv = qr(Av); zv = Matrix(Qv)' * yv

   function _post(θλ)
      (θ, λ) = θλ
      if isempty(θ.nzind); return (θ = θ, λ = λ, σ = Inf); end  
      inz = θ.nzind 
      θ1, σ = solve_tsvd(Rt[:, inz], zt, Rv[:, inz], zv)
      θ2 = copy(θ); θ2[inz] .= θ1
      return (θ = θ2, λ = λ, σ = σ)
   end

   return _post.(path)

#    post = [] 
#    for (θ, λ) in path
#       if isempty(θ.nzind); push!(post, (θ = θ, λ = λ, σ = Inf)); continue; end  
#       inz = θ.nzind 
#       θ1, σ = solve_tsvd(Rt[:, inz], zt, Rv[:, inz], zv)
#       θ2 = copy(θ); θ2[inz] .= θ1
#       push!(post, (θ = θ2, λ = λ, σ = σ))
#    end 
#    return identity.(post)
end
