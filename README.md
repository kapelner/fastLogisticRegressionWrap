# fastLogisticRegressionWrap

The public repository for the R package [fastLogisticRegressionWrap](https://cran.rstudio.com/web/packages/fastLogisticRegressionWrap/) which extends [RcppNumerical::fastLR](https://rdrr.io/cran/RcppNumerical/man/fastLR.html). We now allow for using GPU
speedups but not natively in the package (read further on). Here are some compelling benchmarks.

For vanilla logistic regression with coefficient standard error and p-values computed:

```
library(fastLogisticRegressionWrap)
# Welcome to fastLogisticRegressionWrap v1.0.1.
library(microbenchmark)

set.seed(123)
n = 5000
p = 500 #must be even
X = cbind(1, matrix(rnorm(n * p), n))
colnames(X) = c("(intercept)", paste0("x_useful_", 1 : (p / 2)), paste0("x_useless_", 1 : (p / 2)))
beta = c(runif(p / 2 + 1), rep(0, p / 2))
y = rbinom(n, 1, 1 / (1 + exp(-c(X %*% beta))))

microbenchmark(
  glm = glm(y ~ 0 + ., data.frame(X), family = "binomial"),
  flr = fast_logistic_regression(X, y, do_inference_on_var = "all"),
  times = 10
)
```

yields a 15x speedup on one core:

```
Unit: milliseconds
 expr       min        lq      mean    median        uq       max neval
  glm 3082.2709 3089.9263 3159.1201 3180.7372 3217.9334 3226.9130    10
  flr  201.8558  202.6285  217.8595  204.7916  208.3821  333.0663    10
```

For forward stepwise logistic regression using lowest AIC:

```
set.seed(123)
n = 1000
p = 100 #must be even
X = cbind(1, matrix(rnorm(n * p), n))
beta = c(runif(p / 2 + 1), rep(0, p / 2))
y = rbinom(n, 1, 1 / (1 + exp(-c(X %*% beta))))

microbenchmark( #stepwise benchmark
  glm_stepwise = {
    fullmod = glm(y ~ 0 + ., data = data.frame(X), family = binomial)
    nothing = glm(y ~ 0,     data = data.frame(X), family = binomial)
    forwards = step(nothing, scope = list(lower = formula(nothing), 
                      upper = formula(fullmod)), direction = "forward", trace = 0)
  },
  flr_stepwise = fast_logistic_regression_stepwise_forward(X, y, verbose = FALSE),
  times = 3
)
```

yields an 8x speedup on one core:

```
Unit: seconds
         expr       min        lq      mean    median        uq       max neval
 glm_stepwise 39.909766 40.056819 40.279971 40.203873 40.465073 40.726273     3
 flr_stepwise  4.786947  4.809651  4.880977  4.832356  4.927991  5.023627     3
```


For high `n` situations, parallelization may help further since matrix multiplication is embarassingly
parallelizable. We welcome anyone that can show this helps performance as I've tried many `n x p` combinations
without seeing a statistically significant boost using the `num_cores` argument. 
If `p` is large, whatever gains are seemingly swamped by the `p x p` matrix inversion step which is not very parallelizable.

However, the real gains in parallelization are to be had with GPUs. To duplicate the following benchmark, first setup the 
package `GPUmatrix` (whose source code is [here](https://github.com/ceslobfer/GPUmatrix)) by following the instructions [here](https://cran.r-project.org/web/packages/GPUmatrix/vignettes/vignette.html). The 
demo below uses the `torch` setup which requires initialization code a la:

```
Sys.setenv(CUDA_HOME = "/usr/local/cuda-11.7/lib64")
Sys.setenv(CUDA = "11.7")
library(torch)
library(GPUmatrix)
```

We create the matrix multiplication and inverse custom function that uses CUDA and the GPU's via the `GPUmatrix` interface:

```
Xt_times_diag_w_times_X_gpu = function(X, w, num_cores){
  Xgpu = gpu.matrix(X)
  wgpu = gpu.matrix(diag(w))
  GPUmatrix::t(Xgpu) %*% wgpu %*% Xgpu
}

sqrt_diag_matrix_inverse_gpu = function(X, num_cores){
  sqrt(GPUmatrix::diag(GPUmatrix::solve(X)))
}
```

We make no claim these are the fastest implementations using CUDA. We then benchmark it with large n and p:

```
set.seed(123)
n = 5000
p = 1000 #must be even
X = cbind(1, matrix(rnorm(n * p), n))
colnames(X) = c("(intercept)", paste0("x_useful_", 1 : (p / 2)), paste0("x_useless_", 1 : (p / 2)))
beta = c(runif(p / 2 + 1), rep(0, p / 2))
y = rbinom(n, 1, 1 / (1 + exp(-c(X %*% beta))))

microbenchmark(
  flr = fast_logistic_regression(X, y, do_inference_on_var = "all"),
  flr_gpu = fast_logistic_regression(X, y, do_inference_on_var = "all", Xt_times_diag_w_times_X_fun = Xt_times_diag_w_times_X_gpu, sqrt_diag_matrix_inverse_fun = sqrt_diag_matrix_inverse_gpu),
  times = 10
)
```

to arrive at an additional 4x performance boost:

```
Unit: milliseconds
    expr      min       lq     mean   median       uq      max neval
     flr 797.1429 799.4470 818.9443 803.5378 811.2130 954.4911    10
 flr_gpu 179.1967 192.4239 209.6288 194.4077 201.7585 339.7079    10
```

We also implemented a fast method for inference for one of the coefficients. This gives a modest
gain only in the case of medium p relative to n e.g.

```
set.seed(123)
n = 5000
p = 500 #must be even
X = cbind(1, matrix(rnorm(n * p), n))
colnames(X) = c("(intercept)", paste0("x_useful_", 1 : (p / 2)), paste0("x_useless_", 1 : (p / 2)))
beta = c(runif(p / 2 + 1), rep(0, p / 2))
y = rbinom(n, 1, 1 / (1 + exp(-c(X %*% beta))))
j = 137

microbenchmark(
  flr_all = fast_logistic_regression(X, y, do_inference_on_var = "all")$approx_pval[j],
  flr_j = fast_logistic_regression(X, y, do_inference_on_var = j)$approx_pval[j],
  times = 10
)
```

yields a 10% gain over computing inference for all coefficients:

```
Unit: milliseconds
    expr      min       lq     mean  median       uq      max neval
 flr_all 204.6230 205.4252 205.8986 205.870 206.1752 207.4855    10
   flr_j 187.3354 188.1089 189.4443 189.082 189.9023 194.1588    10
```


Remember: all benchmark multiples shown here change with the `n, p, num_cores` used and your specific settings and hardware.
