# FastLR Wrapper Print

Returns the summary table a la glm

## Usage

``` r
# S3 method for class 'fast_logistic_regression'
print(x, ...)
```

## Arguments

- x:

  The object built using the `fast_logistic_regression` or
  `fast_logistic_regression_stepwise` wrapper functions

- ...:

  Other arguments to be passed to print

## Value

The summary as a data.frame

## Examples

``` r
library(MASS); data(Pima.te)
flr = fast_logistic_regression(
  Xmm = model.matrix(~ . - type, Pima.te), 
 ybin = as.numeric(Pima.te$type == "Yes"))
print(flr)
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
