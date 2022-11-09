using ACEfit
using LinearAlgebra
using Test

@info("Test committee creation with bayesian_ridge_regression_svd.")
# create committee
@warn "Why is such a large committee required?"
committee_size = 20000
X = float.([1 0 0; 1 1 0; 0 2 1])
Y = float.([1; 2; 1])
res = ACEfit.BayesianLinear.bayesian_linear_regression_svd(
        X, Y; committee_size=committee_size, ret_covar=true)
mean, committee, covar = (res["c"], res["committee"], res["covar"])
# test mean
mean_approx = sum(committee; dims=2) / committee_size
@test maximum(abs.(mean_approx-mean)) < 0.01
# test covariance
U, S, V = svd(X; full=true, alg=LinearAlgebra.QRIteration())
covar_approx = zeros(size(X,2),size(X,2))
for n in 1:committee_size
    covar_approx .+= (committee[:,n]-mean) * transpose(committee[:,n]-mean)
end
covar_approx .= covar_approx / committee_size
@test maximum(abs.(covar_approx-covar)) < 0.01
