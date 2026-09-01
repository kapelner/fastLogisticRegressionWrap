# A fast det(X) function

Via the eigen package

## Usage

``` r
eigen_det(X, num_cores = 1)
```

## Arguments

- X:

  A numeric matrix of size p x p

- num_cores:

  The number of cores to use. Unless p is large, keep to the default of
  1.

## Value

The determinant as a scalar numeric value

## Examples

``` r
  p = 30
  eigen_det(matrix(rnorm(p^2), nrow = p))
#> [1] 5.963285e+15
```
