using ACEfit
using LinearAlgebra
using Test

@info("Test ensemble creation with bayesian_ridge_regression_svd.")
# create ensemble
ensemble_size = 20000
X = float.([1 0 0; 1 1 0; 0 2 1])
Y = float.([1; 2; 1])
mean, var_0, var_e, ensemble = 
    ACEfit.BayesianRegression.bayesian_ridge_regression_svd(
        X, Y; ensemble_size=ensemble_size)
# test mean
mean_approx = sum(ensemble; dims=2) / ensemble_size
@test maximum(abs.(mean_approx-mean)) < 0.01
# test covariance
U, S, V = svd(X; full=true, alg=LinearAlgebra.QRIteration())
covar = V * Diagonal(1.0./(S.*S/var_e.+1.0/var_0)) * transpose(V)
covar_approx = zeros(size(X,2),size(X,2))
for n in 1:ensemble_size
    covar_approx .+= (ensemble[:,n]-mean) * transpose(ensemble[:,n]-mean)
end
covar_approx .= covar_approx / ensemble_size
@test maximum(abs.(covar_approx-covar)) < 0.01
