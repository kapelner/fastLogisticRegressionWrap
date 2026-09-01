# Compute Single Value of the Diagonal of a Symmetric Matrix's Inverse

Via the eigen package's conjugate gradient descent algorithm.

## Usage

``` r
eigen_compute_single_entry_of_diagonal_matrix(M, j, num_cores = 1)
```

## Arguments

- M:

  The symmetric matrix which to invert (and then extract one element of
  its diagonal)

- j:

  The diagonal entry of `M`'s inverse

- num_cores:

  The number of cores to use. Default is 1.

## Value

The \\(j, j)\\ entry of \\M^{-1}\\.

## Author

Adam Kapelner

## Examples

``` r
  n = 500
  X = matrix(rnorm(n^2), nrow = n, ncol = n)
  M = t(X) %*% X
  j = 137
  eigen_compute_single_entry_of_diagonal_matrix(M, j)
#> [1] 3.185037
  solve(M)[j, j] #to ensure it's the same value
#> [1] 3.185037
```
