Sys.setenv(CUDA_HOME = "/usr/local/cuda-11.7/lib64")
Sys.setenv(CUDA = "11.7")
pacman::p_load(torch, GPUmatrix)

Xt_times_diag_w_times_X_gpu = function(X, w, num_cores){
  Xgpu = gpu.matrix(X)
  wgpu = gpu.matrix(diag(w))
  GPUmatrix::t(Xgpu) %*% wgpu %*% Xgpu
}

sqrt_diag_matrix_inverse_gpu = function(X, num_cores){
  sqrt(GPUmatrix::diag(GPUmatrix::solve(X)))
}

# n = 100
# p = 10
# X = matrix(rnorm(n*p), nrow = n, ncol = p)
# w = rnorm(n)
# 
# M = Xt_times_diag_w_times_X_gpu(X, w)
# dim(M)
# sqrtdiagMinv = sqrt(diag(solve(M)))
# sqrtdiagMinv = sqrt_diag_matrix_inverse_gpu(M)
# dim(Minv)

pacman::p_load(fastLogisticRegressionWrap, microbenchmark)

gc()
set.seed(123)
n = 5000
p = 1000 #must be even
X = cbind(1, matrix(rnorm(n * p), n))
# colnames(X) = c("(intercept)", paste0("x_useful_", 1 : (p / 2)), paste0("x_useless_", 1 : (p / 2)))
beta = c(runif(p / 2 + 1), rep(0, p / 2))
y = rbinom(n, 1, 1 / (1 + exp(-c(X %*% beta))))

microbenchmark(
  flr = fast_logistic_regression(X, y, do_inference_on_var = "all"),
  flr_gpu = fast_logistic_regression(X, y, do_inference_on_var = "all", Xt_times_diag_w_times_X_fun = Xt_times_diag_w_times_X_gpu, sqrt_diag_matrix_inverse_fun = sqrt_diag_matrix_inverse_gpu),
  times = 10
)



