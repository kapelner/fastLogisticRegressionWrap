# fastLogisticRegressionWrap

The public repository for the R package fastLogisticRegressionWrap on CRAN which extends [fastLR](https://rdrr.io/cran/RcppNumerical/man/fastLR.html). We now allow for using GPU
speedups but not natively in the package (read further on). Here are some compelling benchmarks 
found in `testing.R`.

For vanilla logistic regression with coefficient standard error and p-values computed:

```
library(fastLogisticRegressionWrap)
# Welcome to fastLogisticRegressionWrap v1.0.1.
library(microbenchmark)

set.seed(123)
n = 5000
p = 500 #must be even
X = cbind(1, matrix(rnorm(n * p), n))
# colnames(X) = c("(intercept)", paste0("x_useful_", 1 : (p / 2)), paste0("x_useless_", 1 : (p / 2)))
beta = c(runif(p / 2 + 1), rep(0, p / 2))
y = rbinom(n, 1, 1 / (1 + exp(-c(X %*% beta))))

microbenchmark(
  glm = glm(y ~ 0 + ., data.frame(X), family = "binomial"),
  flr = fast_logistic_regression(X, y, do_inference_on_var = "all"),
  times = 10
)
```

yields a 35x speedup on one core:

```
Unit: milliseconds
 expr        min         lq       mean     median         uq       max neval
  glm 10205.7267 10345.3674 10554.3971 10485.0080 10728.7324 10972.457     3
  flr   282.7045   285.9829   303.0269   289.2613   313.1882   337.115     3
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

yields a 8x speedup on one core:

```
Unit: seconds
         expr       min        lq      mean    median        uq       max neval
 glm_stepwise 39.909766 40.056819 40.279971 40.203873 40.465073 40.726273     3
 flr_stepwise  4.786947  4.809651  4.880977  4.832356  4.927991  5.023627     3
```


For high `n` situations, parallelization may help further since matrix multiplication is embarassingly
parallelizable. We welcome anyone that can show this helps performance as I've tried many `n x p` combinations
without seeing a statistically significant boost. If `p` is large, whatever gains are swamped by the `p x p` matrix inversion step which is not very parallelizable.

However, the real gains are to be had with GPUs. To duplicate the following, first set up package `GPUmatrix` whose
source code is [here](https://github.com/ceslobfer/GPUmatrix) by following the instructions [here](https://cran.r-project.org/web/packages/GPUmatrix/vignettes/vignette.html). The 
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
  sqrt(diag(solve(X)))
}
```

We make no claim these are the fastest implementations using CUDA. We then benchmark it with large n and p:

```
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
```

to arrive an additional gain of 4x performance boost:

```
Unit: milliseconds
    expr      min       lq     mean   median       uq      max neval
     flr 797.1429 799.4470 818.9443 803.5378 811.2130 954.4911    10
 flr_gpu 179.1967 192.4239 209.6288 194.4077 201.7585 339.7079    10
```

WARNING: all benchmark multiples shown here change with the `n, p, num_cores` used and your specific settings.
