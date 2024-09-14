
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
ACEfit.ASP(; P = I, select = (:byerror, 1.0), tsvd = false, nstore=100, 
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
* `Aval = nothing` : `p`-by-`n` validation matrix
* `bval = nothing` : `p`- validation vector

If independent `Aval` and `yval` are provided (instead of detaults `A, y`), 
then the solver will use this separate validation set instead of the training
set to select the best solution along the model path. 
"""
struct ASP
    P
    select
    tsvd::Bool
    nstore::Integer
    params
end

function ASP(; P = I, select, tsvd=false, nstore=100, params...)
    return ASP(P, select, tsvd, nstore, params)
end

function solve(solver::ASP, A, y, Aval=A, yval=y)
    # Apply preconditioning
    AP = A / solver.P
    AvalP = Aval / solver.P
    tracer = asp_homotopy(AP, y; solver.params..., traceFlag = true)

    q = length(tracer) 
    every = max(1, q / solver.nstore)
    istore = unique(round.(Int, [1:every:q; q]))
    new_tracer = [ (solution = tracer[i][1], λ = tracer[i][2], σ = 0.0 ) 
                   for i in istore ]

    if solver.tsvd  # Post-processing if tsvd is true
        post = post_asp_tsvd(new_tracer, AP, y, AvalP, yval)
        new_post = [ (solution = solver.P \ p.θ, λ = p.λ, σ = p.σ) 
                     for p in post ]
    else
        new_post = [ (solution = solver.P \ p.solution, λ = p.λ, σ = 0.0) 
                     for p in new_tracer ]
    end

    tracer_final = _add_errors(new_post, Aval, yval)
    xs, in = asp_select(tracer_final, solver.select)

    return Dict(   "C" => xs, 
                "path" => tracer_final, )
end


function _add_errors(tracer, A, y) 
    rtN = sqrt(length(y))
    return [ ( solution = t.solution, λ = t.λ, σ = t.σ, 
               rmse = norm(A * t.solution - y) / rtN ) 
             for t in tracer ]
end

asp_select(D::Dict, select) = asp_select(D["path"], select)

function asp_select(tracer, select)
    if select == :final
      criterion = :final 
    else
      criterion, p = select
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

    errors = [ t.rmse for t in tracer[1:maxind] ]
    min_error = minimum(errors)
    for (i, error) in enumerate(errors)
        if error <= threshold * min_error
            return tracer[i][:solution], i
        end
    end

    error("selection failed for unknown reasons; please file an issue with a MWE to reproduce this error.")
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
end

# TODO: revisit this idea. Maybe we do want to keep this, not as `select` 
#       but as `solve`. But if we do, then it might be nice to be able to 
#       extend the path somehow. For now I'm removing it since I don't see 
#       the immediate need yet. Just calling asp_select is how I would normally 
#       use this.  
#
# function select(tracer, solver, A, y) #can be called by the user to warm-start the selection
#     xs, in = select_solution(tracer, solver, A, y)
#     return Dict("C" => xs, 
#                 "path" => tracer, 
#                 "nnzs" => length( (tracer[in][:solution]).nzind) )  
# end
