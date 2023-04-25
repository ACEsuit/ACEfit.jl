module BayesianLinear

export bayesian_linear_regression

#=

### Model and prior

Assume a linear model
$$
f(\mathbf{x}) = \sum_{i=1}^M c_i \phi_i(\mathbf{x}) = \mathbf{c}^T \boldsymbol{\phi}(\mathbf{x} )
$$
where the elements of the $M$-dimensional coefficient vector have independent priors $p(c_i)=\mathcal{N}(c_i | 0,\sigma_i^2)$. Suppose $N$ noisy input-target pairs are available, $\{\mathbf{x}_i, y_i\}_{i=1}^N$, such that
$$
y_i = f(\mathbf{x}_i) + \epsilon_i,
$$
with noise prior $p(\epsilon_i) = \mathcal{N}(\epsilon_i | 0,\sigma_\epsilon^2)$.

### Posterior

For fixed values of $\{\sigma_i^2\}_{i=1}^M$ and $\sigma_\epsilon^2$, the posterior mean used for predictions is
$$
\boldsymbol{\mu}_c = \sigma_\epsilon^{-2} \boldsymbol{\Sigma}_c \boldsymbol{\Phi}^T \mathbf{y}
$$
with associated covariance
$$
\begin{aligned}
\boldsymbol{\Sigma}_c
&= \left[ \sigma_\epsilon^{-2} \boldsymbol{\Phi}^T \boldsymbol{\Phi} + \boldsymbol{\Sigma}_0^{-1} \right]^{-1} \\
&= \boldsymbol{\Sigma}_0 - \boldsymbol{\Sigma}_0 \boldsymbol{\Phi}^T \boldsymbol{\Sigma}_y^{-1} \boldsymbol{\Phi} \boldsymbol{\Sigma}_0 \\
&= \mathbf{V} \left[\sigma_\epsilon^{-2} \mathbf{D}^T \mathbf{D} + \mathbf{V}^T \boldsymbol{\Sigma}_0^{-1} \mathbf{V} \right]^{-1} \mathbf{V}^T
\end{aligned} ,
$$
where $\boldsymbol{\Phi}$ is the design matrix
$$
\boldsymbol{\Phi} = \left[
\begin{matrix}
\text{---} \boldsymbol{\phi}(\mathbf{x}_1) \text{---} \\
\text{---} \boldsymbol{\phi}(\mathbf{x}_2) \text{---} \\
\vdots \\
\text{---} \boldsymbol{\phi}(\mathbf{x}_3) \text{---}
\end{matrix}
\right] ,
$$
$\boldsymbol{\Sigma}_0$ is diagonal with $[\boldsymbol{\Sigma}_0]_{ii} = \sigma_i^2$, and $\boldsymbol{\Sigma}_y$ is defined below. Some formulas rely on the singular value decomposition of the design matrix
$$
\boldsymbol{\Phi} = \boldsymbol{U} \boldsymbol{D} \boldsymbol{V}^T
$$

### Likelihood and log likelihood
With flat hyperpriors, the marginal likelihood---the "evidence for the model"---is
$$
p(\mathbf{y} | \boldsymbol{\Sigma}_0, \sigma_\epsilon^2) = \mathcal{N}(\mathbf{y} | \mathbf{0}, \boldsymbol{\Sigma}_y)
$$
with
$$
\begin{aligned}
\boldsymbol{\Sigma}_y
&= \boldsymbol{\Phi} \boldsymbol{\Sigma}_0 \boldsymbol{\Phi}^T + \sigma_\epsilon^2 \mathbf{I} \\
&= \left[ \sigma_\epsilon^{-2} \mathbf{I} - \sigma_\epsilon^{-2} \boldsymbol{\Phi} \boldsymbol{\Sigma}_c \boldsymbol{\Phi}^T \sigma_\epsilon^{-2}  \right]^{-1}
\end{aligned}
$$
When maximizing this quantity, one typically works with its log,
$$
\begin{aligned}
\mathrm{\mathcal{L}}
&= \ln p(\mathbf{y} | \boldsymbol{\Sigma}_0, \sigma_\epsilon^2) \\
&= -\frac{1}{2} \mathbf{y}^T \boldsymbol{\Sigma}_y^{-1} \mathbf{y}
    - \frac{1}{2} \ln \left| \boldsymbol{\Sigma}_y \right|
    - \frac{N}{2} \ln (2\pi) \\
&= -\frac{1}{2} \sigma_\epsilon^{-2} \mathbf{y}^T \left[  \mathbf{y} - \boldsymbol{\Phi} \boldsymbol{\mu}_c \right]
    + \frac{1}{2} \ln \left| \boldsymbol{\Sigma}_c \right|
    - \frac{1}{2} \ln \left| \boldsymbol{\Sigma}_0 \right|
    - \frac{N}{2} \ln (\sigma_\epsilon^2)
    - \frac{N}{2} \ln (2\pi) \\
&= -\frac{1}{2} \sigma_\epsilon^{-2}  ||  \mathbf{y} - \boldsymbol{\Phi} \boldsymbol{\mu}_c ||^2
    - \frac{1}{2} \boldsymbol{\mu}_c^T \boldsymbol{\Sigma}_0^{-1} \boldsymbol{\mu}_c
    + \frac{1}{2} \ln \left| \boldsymbol{\Sigma}_c \right|
    - \frac{1}{2} \ln \left| \boldsymbol{\Sigma}_0 \right|
    - \frac{N}{2} \ln (\sigma_\epsilon^2)
    - \frac{N}{2} \ln (2\pi) \\
\end{aligned} .
$$
The relevant derivatives with respect to the hyperparameters are
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \sigma_i^2}
&= \frac{1}{2} [\boldsymbol{\Phi}^T \boldsymbol{\Sigma}_y^{-1} \mathbf{y}]_i^2
    - \frac{1}{2} [\boldsymbol{\Phi}^T \boldsymbol{\Sigma}_y^{-1} \boldsymbol{\Phi}]_{ii} \\
&= \frac{1}{2} \sigma_i^{-4} \left[ \mu_i^2 + [\boldsymbol{\Sigma}_c]_{ii} - \sigma_i^2 \right]
\end{aligned}
$$
and
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \sigma_\epsilon^2}
& = \frac{1}{2} \left\| \boldsymbol{\Sigma}_y^{-1} \mathbf{y} \right\|^2 - \frac{1}{2} \mathrm{tr}\left(\boldsymbol{\Sigma}_y^{-1}\right) \\
&= \frac{1}{2} \sigma_\epsilon^{-4} \left[
    \left\| \mathbf{y} - \boldsymbol{\Phi} \boldsymbol{\mu}_c \right\|^2
    + \mathrm{tr}\left( \boldsymbol{\Phi} \boldsymbol{\Sigma}_c \boldsymbol{\Phi}^T \right)
    - N \sigma_\epsilon^2
\right]
\end{aligned} .
$$

### Special case of Bayesian Ridge Regression

For the simpler case of Bayesian Ridge Regression, defined by

$$
\boldsymbol{\Sigma}_0 = \sigma_0^2 \mathbf{I},
$$

the formulas reduce to

$$
\boldsymbol{\mu}_c = \sigma_\epsilon^{-2} \mathbf{V} \left[\sigma_\epsilon^{-2} \mathbf{D}^T \mathbf{D} + \sigma_0^{-2} \mathbf{I} \right]^{-1} \mathbf{D}^T \mathbf{U}^T \mathbf{y}
$$

$$
\boldsymbol{\Sigma}_c
= \mathbf{V} \left[\sigma_\epsilon^{-2} \mathbf{D}^T \mathbf{D} + \sigma_0^{-2} \mathbf{I} \right]^{-1} \mathbf{V}^T
$$

$$
\begin{aligned}
\mathrm{\mathcal{L}}
= &- \frac{1}{2} \sum_{i=1}^r \frac{[\mathbf{U}^T \mathbf{y}]_i^2}{\sigma_0^2 d_i^2 + \sigma_\epsilon^2}
    - \frac{1}{2} \sum_{i=r+1}^N \frac{[\mathbf{U}^T \mathbf{y}]_i^2}{\sigma_\epsilon^2} 
\\
    &- \frac{1}{2} \ln \prod_{i=1}^r (\sigma_0^2 d_i^2 + \sigma_\epsilon^2)
    - \frac{N-r}{2} \ln (\sigma_\epsilon^2)
\\
    &- \frac{N}{2} \ln (2\pi) \\
\end{aligned}
$$

$$
\frac{\partial \mathcal{L}}{\partial \sigma_0^2}
= \frac{1}{2} \sum_{i=1}^r \frac{[\mathbf{U}^T \mathbf{y}]_i^2 d_i^2}{(\sigma_0^2 d_i^2 + \sigma_\epsilon^2)^2}
    - \frac{1}{2} \sum_{i=1}^r \frac{d_i^2}{\sigma_0^2 d_i^2 + \sigma_\epsilon^2}
$$

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \sigma_\epsilon^2}
= &\frac{1}{2} \sum_{i=1}^r \frac{[\mathbf{U}^T \mathbf{y}]_i^2}{(\sigma_0^2 d_i^2 + \sigma_\epsilon^2)^2}
    + \frac{1}{2} \sum_{i=r+1}^N \frac{[\mathbf{U}^T \mathbf{y}]_i^2}{\sigma_\epsilon^4}
\\
    &- \frac{1}{2} \sum_{i=1}^r \frac{1}{\sigma_0^2 d_i^2 + \sigma_\epsilon^2}
    - \frac{N-r}{2\sigma_\epsilon^2}
\end{aligned}
$$

### References

* M. E. Tipping, "Sparse Bayesian learning and the relevance vector machine," Journal of Machine Learning Research 1, 211 (2001)
* C. K. Williams and C. E. Rasmussen, _Gaussian Processes for Machine Learning_, MIT Press (2006)

=#

using LinearAlgebra
using Optim

"""
    bayesian_linear_regression(A, Y; <keyword arguments>)

Perform Bayesian linear regression, possibly with automatic relevance determination.

# Arguments

- `A::Matrix{<:AbstractFloat}`: design matrix, with observations as rows.
- `Y::Vector{<:AbstractFloat}`: target vector.

### regularization settings
- `sig_0_floor::AbstractFloat = 1e-8`: lower bound for σ_0, the standard deviation for the coefficient prior.
- `sig_e_floor::AbstractFloat = 1e-8`: lower bound for σ_ε, the standard deviation for the model error.
- `ard_threshold::AbstractFloat = 0.0`: automatic relevance determination.

### output settings
- `verbose::Bool = true`
- `committee_size::Int = 0`: if nonzero, sample from the posterior and include a committee in the results.
- `ret_covar::Bool = false`: whether to supply the covariance matrix in the results.

### solver settings
- `factorization::Symbol = :cholesky`: if "cholesky" performs poorly, try "svd" or "qr".
- `optimizer::Symbol = :LBFGS`: method for evidence maximization.
- `tol::Float64 = 1e-3`: tolerance to use for terminating the evidence maximization.
- `max_iter::Int = 1000`: iteration limit for evidence maximization.
"""

function bayesian_linear_regression(A, Y;
                                    # regularization settings
                                    sig_0_floor = 1e-4,
                                    sig_e_floor = 1e-4,
                                    ard_threshold = 0.0,
                                    # output settings
                                    verbose::Bool = true,
                                    committee_size::Int = 0,
                                    ret_covar::Bool = false,
                                    # solver settings
                                    factorization = :cholesky,
                                    optimizer = :LBFGS,
                                    tol::AbstractFloat = 1e-3,
                                    max_iter::Int = 1000)
    if (ard_threshold == 0.0) && (factorization == :cholesky)
        return bayesian_fit(A, Y;
                            variance_c_floor = sig_0_floor * sig_0_floor,
                            variance_e_floor = sig_e_floor * sig_e_floor,
                            verbose = verbose)

    elseif (ard_threshold == 0.0) && (factorization == :svd)
        return bayesian_linear_regression_svd(A, Y;
                                              variance_0_floor = sig_0_floor * sig_0_floor,
                                              variance_e_floor = sig_e_floor * sig_e_floor,
                                              committee_size,
                                              verbose,
                                              ret_covar)

    elseif (ard_threshold > 0.0) && (factorization == :cholesky)
        @warn "ard_tolerance not passed"
        return ard_fit(A, Y;
                       variance_c_floor = sig_0_floor * sig_0_floor,
                       variance_e_floor = sig_e_floor * sig_e_floor,
                       verbose = verbose)
    end
end

# -----

function solve(y::Vector{<:AbstractFloat},
               X::Matrix{<:AbstractFloat},
               var_c::AbstractFloat,
               var_e::AbstractFloat)
    return solve(y, X, var_c * ones(size(X, 2)), var_e)
end

function solve(y::Vector{<:AbstractFloat},
               X::Matrix{<:AbstractFloat},
               var_c::Vector{<:AbstractFloat},
               var_e::AbstractFloat)
    M = size(X, 2)
    XTX = X' * X
    Σ_c = Array{Float64}(undef, M, M)
    BLAS.blascopy!(length(Σ_c), XTX, stride(XTX, 1), Σ_c, stride(Σ_c, 1))
    BLAS.scal!(1.0 / var_e, Σ_c)
    for i in 1:M
        Σ_c[i, i] += 1.0 / var_c[i]
    end
    C = cholesky!(Symmetric(Σ_c))
    return 1.0 / var_e * (C \ (X' * y))
end

function log_marginal_likelihood_overdetermined!(lml::AbstractFloat,
                                                 grad::Vector{<:AbstractFloat},
                                                 X::Matrix{<:AbstractFloat},
                                                 y::Vector{<:AbstractFloat},
                                                 var_c::AbstractFloat,
                                                 var_e::AbstractFloat,
                                                 XTX::Matrix{<:AbstractFloat})
    var_c_vec = var_c * ones(size(X, 2))
    grad_vec = zeros(size(X, 2) + 1)
    lml = log_marginal_likelihood_overdetermined!(lml, grad_vec, X, y, var_c_vec, var_e,
                                                  XTX)
    grad[1] = sum(grad_vec[1:(end - 1)])
    grad[2] = grad_vec[end]
    return lml
end

function log_marginal_likelihood_overdetermined!(lml::AbstractFloat,
                                                 grad::Vector{<:AbstractFloat},
                                                 X::Matrix{<:AbstractFloat},
                                                 y::Vector{<:AbstractFloat},
                                                 var_c::Vector{<:AbstractFloat},
                                                 var_e::AbstractFloat,
                                                 XTX::Matrix{<:AbstractFloat})
    N = size(X, 1)
    M = size(X, 2)
    Σ_c = Array{Float64}(undef, M, M)
    BLAS.blascopy!(length(Σ_c), XTX, stride(XTX, 1), Σ_c, stride(Σ_c, 1))
    BLAS.scal!(1.0 / var_e, Σ_c)
    for i in 1:M
        Σ_c[i, i] += 1.0 / var_c[i]
    end
    C = cholesky!(Symmetric(Σ_c))
    Σ_c = C \ I(M)
    μ_c = 1.0 / var_e * (C \ (X' * y))
    lml = -0.5 * logdet(C) - 0.5 * sum(log.(var_c)) - 0.5 * N * log(var_e) -
          0.5 * N * log(2 * π)
    lml -= 0.5 / var_e * y' * (y - X * μ_c)
    grad[1:M] .= 0.5 * (μ_c .^ 2 .+ diag(Σ_c) .- var_c) ./ var_c .^ 2
    grad[M + 1] = 0.5 / var_e^2 * (sum((y - X * μ_c) .^ 2) + dot(XTX, Σ_c) - N * var_e)
    return lml
end

function log_marginal_likelihood_underdetermined!(lml::AbstractFloat,
                                                  grad::Vector{<:AbstractFloat},
                                                  X::Matrix{<:AbstractFloat},
                                                  y::Vector{<:AbstractFloat},
                                                  var_c::AbstractFloat,
                                                  var_e::AbstractFloat)
    var_c_vec = var_c * ones(size(X, 2))
    grad_vec = zeros(size(X, 2) + 1)
    lml = log_marginal_likelihood_underdetermined!(lml, grad_vec, X, y, var_c_vec, var_e)
    grad[1] = sum(grad_vec[1:(end - 1)])
    grad[2] = grad_vec[end]
    return lml
end

function log_marginal_likelihood_underdetermined!(lml::AbstractFloat,
                                                  grad::Vector{<:AbstractFloat},
                                                  X::Matrix{<:AbstractFloat},
                                                  y::Vector{<:AbstractFloat},
                                                  var_c::Vector{<:AbstractFloat},
                                                  var_e::AbstractFloat)
    N = size(X, 1)
    M = size(X, 2)
    Σ_y = X * Diagonal(var_c) * X'
    for i in 1:N
        Σ_y[i, i] += var_e
    end
    C = cholesky!(Symmetric(Σ_y))
    invΣy_y = C \ y
    lml = -0.5 * y' * invΣy_y - 0.5 * logdet(C) - 0.5 * N * log(2 * π)
    grad[1:M] .= 0.5 * (X' * invΣy_y) .^ 2
    W = C \ X
    @views for i in 1:M
        grad[i] -= 0.5 * dot(X[:, i], W[:, i])
    end
    grad[M + 1] = 0.5 * dot(invΣy_y, invΣy_y) - 0.5 * tr(C \ I(N))
    return lml
end

function bayesian_fit(X::Matrix{<:AbstractFloat},
                      y::Vector{<:AbstractFloat};
                      variance_c_floor::AbstractFloat = 1e-8,
                      variance_e_floor::AbstractFloat = 1e-8,
                      verbose::Bool = false)
    if size(X, 1) >= size(X, 2)
        XTX = X' * X  # advantageous to precompute for overdetermined case
    end

    function fg!(f, g, x)
        var_c = variance_c_floor + x[1] * x[1]
        var_e = variance_e_floor + x[2] * x[2]
        if size(X, 1) >= size(X, 2)
            f = log_marginal_likelihood_overdetermined!(f, g, X, y, var_c, var_e, XTX)
        else
            f = log_marginal_likelihood_underdetermined!(f, g, X, y, var_c, var_e)
        end
        if f != nothing
            f = -f
        end
        if g != nothing
            g .*= -2 * x
        end
        return f
    end

    res = optimize(Optim.only_fg!(fg!),
                   ones(2),
                   Optim.LBFGS(),
                   Optim.Options(x_tol = 1e-8, g_tol = 0.0, show_trace = verbose))
    verbose && println(res)

    lml = -Optim.minimum(res)
    var_c, var_e = Optim.minimizer(res)
    var_c = variance_c_floor + var_c * var_c
    var_e = variance_e_floor + var_e * var_e

    C = solve(y, X, var_c, var_e)

    return Dict("C" => C)
end

function ard_fit(X::Matrix{<:AbstractFloat},
                 y::Vector{<:AbstractFloat};
                 variance_c_floor::AbstractFloat = 1e-8,
                 variance_e_floor::AbstractFloat = 1e-8,
                 verbose::Bool = false)
    if size(X, 1) >= size(X, 2)
        XTX = X' * X  # advantageous to precompute for overdetermined case
    end

    function fg!(f, g, x)
        var_c = variance_c_floor .+ x[1:(end - 1)] .* x[1:(end - 1)]
        var_e = variance_e_floor + x[end] * x[end]
        if size(X, 1) >= size(X, 2)
            f = log_marginal_likelihood_overdetermined!(f, g, X, y, var_c, var_e, XTX)
        else
            f = log_marginal_likelihood_underdetermined!(f, g, X, y, var_c, var_e)
        end
        if f != nothing
            f = -f
        end
        if g != nothing
            g .*= -2 * x
        end
        return f
    end

    res = optimize(Optim.only_fg!(fg!),
                   ones(size(X, 2) + 1),
                   Optim.LBFGS(),
                   Optim.Options(x_tol = 1e-8, g_tol = 0.0, show_trace = verbose))
    verbose && println(res)

    lml = -Optim.minimum(res)
    x = Optim.minimizer(res)
    var_c = variance_c_floor .+ x[1:(end - 1)] .* x[1:(end - 1)]
    var_e = variance_e_floor + x[end] * x[end]

    mask = var_c .> 10 * variance_c_floor
    var_c[.~mask] .= 0
    c_mask = solve(y, X[:, mask], var_c[mask], var_e)
    c = zeros(length(var_c))
    c[mask] .= c_mask

    return Dict("C" => c)
end

function bayesian_linear_regression_svd(X::Matrix{<:AbstractFloat},
                                        Y::Vector{<:AbstractFloat};
                                        variance_0_floor::AbstractFloat = 1e-8,
                                        variance_e_floor::AbstractFloat = 1e-8,
                                        committee_size::Int = 0,
                                        verbose::Bool = false,
                                        ret_covar = false)
    @info "Entering bayesian_linear_regression_svd"
    @info "Computing SVD of $(size(X)) matrix" BLAS.get_num_threads() BLAS.get_config()
    flush(stdout)
    flush(stderr)
    #elapsed = @elapsed U, S, V = svd!(X; full=true, alg=LinearAlgebra.QRIteration())
    elapsed = @elapsed U, S, V = svd!(X; full = false, alg = LinearAlgebra.QRIteration())
    @info "SVD completed after $(elapsed/60) minutes"

    UT_Y = transpose(U) * Y

    function log_marginal_likelihood!(lml, grad_lml, var_0, var_e)
        lml = 0.0
        dlml_d0 = 0.0
        dlml_de = 0.0
        for i in 1:length(S)
            t = var_0 * S[i] * S[i] + var_e
            lml += -0.5 * UT_Y[i] * UT_Y[i] / t - 0.5 * log(t)
            dlml_d0 += 0.5 * UT_Y[i] * UT_Y[i] * S[i] * S[i] / (t * t) -
                       0.5 * S[i] * S[i] / t
            dlml_de += 0.5 * UT_Y[i] * UT_Y[i] / (t * t) - 0.5 / t
        end
        #lml += -0.5*dot(UT_Y[length(S)+1:end],UT_Y[length(S)+1:end])/var_e  # for full svd
        #dlml_de += 0.5*dot(UT_Y[length(S)+1:end],UT_Y[length(S)+1:end])/(var_e*var_e)
        lml += -0.5 / var_e * (dot(Y, Y) - dot(UT_Y, UT_Y))                         # for thin svd
        dlml_de += 0.5 / (var_e * var_e) * (dot(Y, Y) - dot(UT_Y, UT_Y))
        lml += -0.5 * (size(X, 1) - length(S)) * log(var_e)
        dlml_de += -0.5 * (size(X, 1) - length(S)) / var_e
        lml += -0.5 * size(X, 1) * log(2 * pi)
        grad_lml .= [dlml_d0, dlml_de]
        return lml
    end

    function fg!(f, g, x)
        flush(stdout)
        flush(stderr)
        var_0, var_e = (variance_0_floor, variance_e_floor) .+ x .^ 2
        f = log_marginal_likelihood!(f, g, var_0, var_e)
        if f != nothing
            f = -f
        end
        if g != nothing
            g .*= -2 * x
        end
        return f
    end

    @info "Beginning to maximize marginal likelihood"
    flush(stdout)
    flush(stderr)
    res = optimize(Optim.only_fg!(fg!),
                   ones(2),
                   Optim.LBFGS(),
                   Optim.Options(x_tol = 1e-8, g_tol = 0.0, show_trace = verbose))
    @info "Optimization complete" "Results"=res
    lml = -Optim.minimum(res)
    var_0, var_e = (variance_0_floor, variance_e_floor) .+ Optim.minimizer(res) .^ 2

    UT_Y[1:length(S)] .*= var_0 .* S ./ (var_0 .* S .* S .+ var_e)
    c = V * UT_Y[1:length(S)]

    res = Dict{String, Any}("c" => c,
                            "lml" => lml,
                            "var_0" => var_0,
                            "var_e" => var_e)

    if committee_size > 0
        committee = zeros(size(X, 2), committee_size)
        covar = 1.0 ./ (S .* S / var_e .+ 1.0 / var_0)
        for i in 1:committee_size
            committee[:, i] .= c +
                               V * Diagonal(sqrt.(covar)) * transpose(V) * randn(size(X, 2))
        end
        res["committee"] = committee
        ret_covar && (res["covar"] = V * Diagonal(covar) * transpose(V))
    end

    return res
end

end
