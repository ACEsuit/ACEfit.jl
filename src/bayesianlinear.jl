module BayesianRegression

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
&= \boldsymbol{\Sigma}_0 - \boldsymbol{\Sigma}_0 \boldsymbol{\Phi}^T \boldsymbol{\Sigma}_y^{-1} \boldsymbol{\Phi} \boldsymbol{\Sigma}_0
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
$\boldsymbol{\Sigma}_0$ is diagonal with $[\boldsymbol{\Sigma}_0]_{ii} = \sigma_i^2$, and $\boldsymbol{\Sigma}_y$ is defined below.

### Likelihood and log likelihood

With flat hyperpriors, the marginal likelihood---the "evidence" for the model---is
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

### References

* M. E. Tipping, "Sparse Bayesian learning and the relevance vector machine," Journal of Machine Learning Research 1, 211 (2001)
* C. K. Williams and C. E. Rasmussen, _Gaussian Processes for Machine Learning_, MIT Press (2006)

=#

using LinearAlgebra
using Optim
using Statistics

function solve(
    y::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat},
    var_c::AbstractFloat,
    var_e::AbstractFloat,
)
    return solve(y, X, var_c*ones(size(X,2)), var_e)
end

function solve(
    y::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat},
    var_c::Vector{<:AbstractFloat},
    var_e::AbstractFloat,
)
    Σ_0 = Diagonal(var_c)
    Σ_c = inv(cholesky(Hermitian(1/var_e*X'*X + inv(Σ_0))))
    return 1/var_e*Σ_c*X'*y
end

function log_marginal_likelihood!(
    lml::AbstractFloat,
    grad::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat},
    y::Vector{<:AbstractFloat},
    var_c::AbstractFloat,
    var_e::AbstractFloat
)
    var_c_vec = var_c*ones(size(X,2))
    grad_vec = zeros(size(X,2)+1)
    lml = log_marginal_likelihood!(lml, grad_vec, X, y, var_c_vec, var_e)
    grad[1] = sum(grad_vec[1:end-1])
    grad[2] = grad_vec[end]
    return lml
end

function log_marginal_likelihood!(
    lml::AbstractFloat,
    grad::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat},
    y::Vector{<:AbstractFloat},
    var_c::Vector{<:AbstractFloat},
    var_e::AbstractFloat,
)
    N = size(X,1)
    M = size(X,2)
    Σ_0 = Diagonal(var_c)
    if N <= M
        inv_Σ_y = inv(cholesky(Hermitian(X*Σ_0*X' + var_e*I)))
        lml = -0.5*y'*inv_Σ_y*y + 0.5*logdet(inv_Σ_y) - 0.5*N*log(2*π)
        grad[1:M] = 0.5*(X'*inv_Σ_y*y).^2 - 0.5*diag(X'*inv_Σ_y*X)
        grad[M+1] = 0.5*y'*inv_Σ_y*inv_Σ_y*y - 0.5*tr(inv_Σ_y)
    else
        Σ_c = var_e*inv(cholesky(Hermitian(X'*X+var_e*inv(Σ_0))))
        μ_c = solve(y, X, var_c, var_e)
        lml = (-0.5/var_e*y'*(y-X*μ_c) + 0.5*logdet(Σ_c) - 0.5*logdet(Σ_0)
                    - 0.5*N*log(var_e) - 0.5*N*log(2*π))
        grad[1:M] = 0.5*(μ_c.^2 + diag(Σ_c) - var_c)./var_c.^2
        grad[M+1] = 0.5*((y-X*μ_c)'*(y-X*μ_c) + tr(X*Σ_c*X') - N*var_e)/var_e^2
    end
    return lml
end

function bayesian_fit(
    y::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat};
    variance_floor::AbstractFloat=1e-8,
    verbose::Bool=false,
)
    function fg!(f, g, x)
        var_c = variance_floor + x[1]*x[1]
        var_e = variance_floor + x[2]*x[2]
        f = log_marginal_likelihood!(f, g, X, y, var_c, var_e)
        if f != nothing
            f = -f
        end
        if g != nothing
            g .*= -2*x
        end
        return f
    end

    res = optimize(Optim.only_fg!(fg!),
                   ones(2),
                   Optim.LBFGS(),
                   Optim.Options(g_tol=1e-5, show_trace=verbose))
    verbose && println(res)

    lml = -Optim.minimum(res)
    var_c, var_e = Optim.minimizer(res)
    var_c = variance_floor + var_c*var_c
    var_e = variance_floor + var_e*var_e

    return solve(y, X, var_c, var_e), var_c, var_e, lml
end

function ard_fit(
    y::Vector{<:AbstractFloat},
    X::Matrix{<:AbstractFloat},
    variance_floor::AbstractFloat=1e-8;
    verbose::Bool=false,
)
    function fg!(f, g, x)
        var_c = variance_floor .+ x[1:end-1].*x[1:end-1]
        var_e = variance_floor + x[end]*x[end]
        f = log_marginal_likelihood!(f, g, X, y, var_c, var_e)
        if f != nothing
            f = -f
        end
        if g != nothing
            g .*= -2*x
        end
        return f
    end

    res = optimize(Optim.only_fg!(fg!),
                   ones(size(X,2)+1),
                   Optim.LBFGS(),
                   Optim.Options(g_tol=1e-5, show_trace=verbose))
    verbose && println(res)

    lml = -Optim.minimum(res)
    x = Optim.minimizer(res)
    var_c = variance_floor .+ x[1:end-1].*x[1:end-1]
    var_e = variance_floor + x[end]*x[end]

    mask = var_c .> 10*variance_floor
    var_c[.~mask] .= 0
    c_mask = solve(y, X[:,mask], var_c[mask], var_e)
    c = zeros(length(var_c))
    c[mask] .= c_mask

    return c, var_c, var_e, lml, mask
end

end
