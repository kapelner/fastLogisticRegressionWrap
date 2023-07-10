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
  glm = glm(y ~ 0 + ., data.frame(X), family = binomial()),
  flr = fast_logistic_regression(X, y, do_inference_on_var = TRUE),
  times = 3
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

However, the real gains are to be had with GPUs. To duplicate the following, first set up package `GPUmatrix` by
following the instructions [here](https://cran.r-project.org/web/packages/GPUmatrix/vignettes/vignette.html). The 
demo below uses the `torch` setup.

```

```

WARNING: all benchmark multiples shown here change with the `n, p, num_cores` used and your specific settings.
