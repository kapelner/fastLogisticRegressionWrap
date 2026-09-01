# Asymmetric Cost Explorer

Given a set of desired proportions of predicted outcomes, what is the
error rate for each of those models?

## Usage

``` r
asymmetric_cost_explorer_cross_validated(phat, ybin, K_CV = 5, ...)
```

## Arguments

- phat:

  The vector of probability estimates to be thresholded to make a binary
  decision

- ybin:

  The true binary responses

- K_CV:

  We wish to fit the `phat` thresholds out of sample using this number
  of folds. Default is `5`.

- ...:

  Other parameters to be passed into the `asymmetric_cost_explorer`
  function

## Value

A table with column 1: `proportions_desired`, column 2: actual
proportions (as close as possible), column 3: error rate, column 4:
probability threshold.

## Author

Adam Kapelner
