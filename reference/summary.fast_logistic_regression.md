# FastLR Wrapper Summary

Returns the summary table a la glm

## Usage

``` r
# S3 method for class 'fast_logistic_regression'
summary(object, alpha_order = TRUE, ...)
```

## Arguments

- object:

  The object built using the `fast_logistic_regression` or
  `fast_logistic_regression_stepwise` wrapper functions

- alpha_order:

  Should the coefficients be ordered in alphabetical order? Default is
  `TRUE`.

- ...:

  Other arguments to be passed to `summary`.

## Value

The summary as a data.frame

## Examples

``` r
library(MASS); data(Pima.te)
flr = fast_logistic_regression(
  Xmm = model.matrix(~ . - type, Pima.te), 
 ybin = as.numeric(Pima.te$type == "Yes"))
summary(flr)
#>              approx_coef
#> (Intercept) -9.514133789
#> age          0.018056338
#> bmi          0.078956292
#> bp          -0.008681487
#> glu          0.037480100
#> npreg        0.140978699
#> ped          1.110403052
#> skin         0.013173387
```
