
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
"""
struct ASP
    P
    select
    params
end

function ASP(; P = I, select, mode=:train, params...)
    return ASP(P, select, params)
end

function solve(solver::ASP, A, y, Aval=A, yval=y)
    # Apply preconditioning
    AP = A / solver.P
    
    tracer = asp_homotopy(AP, y; solver.params...)
    q = length(tracer)
    new_tracer = Vector{NamedTuple{(:solution, :λ), Tuple{Any, Any}}}(undef, q)

    for i in 1:q
        new_tracer[i] = (solution = solver.P \ tracer[i][1], λ = tracer[i][2])
    end

    xs, in = select_solution(new_tracer, solver, Aval, yval)

   #  println("done.")
    return Dict(   "C" => xs, 
                "path" => new_tracer, 
                "nnzs" => length((new_tracer[in][:solution]).nzind) )
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


#=
function select_smart(tracer, solver, Aval, yval)

    best_metric = Inf
    best_iteration = 0
    validation_metric = 0
    q = length(tracer)
    errors = [norm(Aval * t[:solution] - yval) for t in tracer]
    nnzss = [(t[:solution]).nzind for t in tracer]
    best_iteration = argmin(errors)
    validation_metric = errors[best_iteration]
    validation_end = norm(Aval * tracer[end][:solution] - yval)

    if validation_end < validation_metric #make sure to check the last one too in case q<<100
        best_iteration = q
    end

    criterion, threshold = solver.select
    
    if criterion == :val
        return tracer[best_iteration][:solution], best_iteration

    elseif criterion == :byerror
        for (i, error) in enumerate(errors)
            if error <= threshold * validation_metric
                return tracer[i][:solution], i
            end
        end

    elseif criterion == :bysize
        first_index = findfirst(sublist -> threshold in sublist, nnzss)
        relevant_errors = errors[1:first_index - 1] 
        min_error = minimum(relevant_errors)
        min_error_index = findfirst(==(min_error), relevant_errors)
        return tracer[min_error_index][:solution], min_error_index

    else
        @error("Unknown selection criterion: $criterion")
    end
end
=#