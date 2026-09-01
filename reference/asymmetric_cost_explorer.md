# Asymmetric Cost Explorer

Given a set of desired proportions of predicted outcomes, what is the
error rate for each of those models?

## Usage

``` r
asymmetric_cost_explorer(
  phat,
  ybin,
  steps = seq(from = 0.001, to = 0.999, by = 0.001),
  outcome_of_analysis = 0,
  proportions_desired = seq(from = 0.1, to = 0.9, by = 0.1),
  proportion_tolerance = 0.01
)
```

## Arguments

- phat:

  The vector of probability estimates to be thresholded to make a binary
  decision

- ybin:

  The true binary responses

- steps:

  All possibile thresholds which must be a vector of numbers in (0, 1).
  Default is `seq(from = 0.001, to = 0.999, by = 0.001)`.

- outcome_of_analysis:

  Which class do you care about performance? Either 0 or 1 for the
  negative class or positive class. Default is `0`.

- proportions_desired:

  Which proportions of `outcome_of_analysis` class do you wish to
  understand performance for?

- proportion_tolerance:

  If the model cannot match the proportion_desired within this amount,
  it does not return that model's performance. Default is `0.01`.

## Value

A table with column 1: `proportions_desired`, column 2: actual
proportions (as close as possible), column 3: error rate, column 4:
probability threshold.

## Author

Adam Kapelner
