# FastLR Wrapper

Returns most of what you get from glm

## Usage

``` r
fast_logistic_regression(
  Xmm,
  ybin,
  drop_collinear_variables = FALSE,
  lm_fit_tol = 1e-07,
  do_inference_on_var = "none",
  Xt_times_diag_w_times_X_fun = NULL,
  sqrt_diag_matrix_inverse_fun = NULL,
  num_cores = 1,
  ...
)
```

## Arguments

- Xmm:

  The model.matrix for X (you need to create this yourself before)

- ybin:

  The binary response vector

- drop_collinear_variables:

  Should we drop perfectly collinear variables? Default is `FALSE` to
  inform the user of the problem.

- lm_fit_tol:

  When `drop_collinear_variables = TRUE`, this is the tolerance to
  detect collinearity among predictors. We use the default value from
  `base::lm.fit`'s which is 1e-7. If you fit the logistic regression and
  still get p-values near 1 indicating high collinearity, we recommend
  making this value smaller.

- do_inference_on_var:

  Which variables should we compute approximate standard errors of the
  coefficients and approximate p-values for the test of no linear
  log-odds probability effect? Default is `"none"` for inference on none
  (for speed). If not default, then `"all"` to indicate inference should
  be computed for all variables. The final option is to pass one index
  to indicate the column number of `Xmm` where inference is desired. We
  have a special routine to compute inference for one variable only. It
  consists of a conjugate gradient descent which is another
  approximation atop the coefficient-fitting approximation in
  RcppNumerical. Note: if you are just comparing nested models using
  anova, there is no need to compute inference for coefficients (keep
  the default of `FALSE` for speed).

- Xt_times_diag_w_times_X_fun:

  A custom function whose arguments are `X` (an n x m matrix), `w` (a
  vector of length m) and this function's `num_cores` argument in that
  order. The function must return an m x m R matrix class object which
  is the result of the computing X^T function is not parallelized, the
  `num_cores` argument is ignored. Default is `NULL` which uses the
  function
  [`eigen_Xt_times_diag_w_times_X`](https://kapelner.github.io/fastLogisticRegressionWrap/reference/eigen_Xt_times_diag_w_times_X.md)
  which is implemented with the Eigen C++ package and hence very fast.
  The only way we know of to beat the default is to use a method that
  employs GPUs. See README on github for more information.

- sqrt_diag_matrix_inverse_fun:

  A custom function that returns a numeric vector which is square root
  of the diagonal of the inverse of the inputted matrix. Its arguments
  are `X` (an n x n matrix) and this function's `num_cores` argument in
  that order. If your custom function is not parallelized, the
  `num_cores` argument is ignored. The object returned must further have
  a defined function `diag` which returns the diagonal of the matrix as
  a vector. Default is `NULL` which uses the function
  [`eigen_inv`](https://kapelner.github.io/fastLogisticRegressionWrap/reference/eigen_inv.md)
  which is implemented with the Eigen C++ package and hence very fast.
  The only way we know of to beat the default is to use a method that
  employs GPUs. See README on github for more information.

- num_cores:

  Number of cores to use to speed up matrix multiplication and matrix
  inversion (used only during inference computation). Default is 1.
  Unless the number of variables, i.e. `ncol(Xmm)`, is large, there does
  not seem to be a performance gain in using multiple cores.

- ...:

  Other arguments to be passed to `fastLR`. See documentation there.

## Value

A list of raw results

## Examples

``` r
library(MASS); data(Pima.te)
flr = fast_logistic_regression(
   Xmm = model.matrix(~ . - type, Pima.te), 
  ybin = as.numeric(Pima.te$type == "Yes")
)
```
