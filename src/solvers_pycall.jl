
@doc raw"""
SKLEARN_BRR
"""
struct SKLEARN_BRR
    tol::Number
    n_iter::Integer
end
SKLEARN_BRR(; tol = 1e-3, n_iter = 300) = SKLEARN_BRR(tol, n_iter)



@doc raw"""
SKLEARN_ARD
"""
struct SKLEARN_ARD
    n_iter::Integer
    tol::Number
    threshold_lambda::Number
end
function SKLEARN_ARD(; n_iter = 300, tol = 1e-3, threshold_lambda = 10000)
    SKLEARN_ARD(n_iter, tol, threshold_lambda)
end


