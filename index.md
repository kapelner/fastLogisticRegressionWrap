# fastLogisticRegressionWrap

`fastLogisticRegressionWrap` provides fast logistic regression with
coefficient inference, forward stepwise model selection, and hooks for
GPU-backed matrix operations. It wraps
[`RcppNumerical::fastLR()`](https://rdrr.io/pkg/RcppNumerical/man/fastLR.html)
in a higher-level R interface.

See the [repository
README](https://github.com/kapelner/fastLogisticRegressionWrap#readme)
for benchmarks and GPU examples, or browse the [function
reference](https://kapelner.github.io/fastLogisticRegressionWrap/reference/).

## Installation

Install the latest development version from R-universe:

``` r

install.packages(
  "fastLogisticRegressionWrap",
  repos = c("https://kapelner.r-universe.dev", "https://cloud.r-project.org")
)
```

Install the CRAN release with:

``` r

install.packages("fastLogisticRegressionWrap")
```
