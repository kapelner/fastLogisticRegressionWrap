# Binary Confusion Table and Errors

Provides a binary confusion table and error metrics

## Usage

``` r
confusion_results(yhat, ybin, skip_argument_checks = FALSE)
```

## Arguments

- yhat:

  The binary predictions

- ybin:

  The true binary responses

- skip_argument_checks:

  If `TRUE` it does not check this function's arguments for
  appropriateness. It is not recommended unless you truly need speed and
  thus the default is `FALSE`.

## Value

A list of raw results

## Examples

``` r
library(MASS); data(Pima.te)
ybin = as.numeric(Pima.te$type == "Yes")
flr = fast_logistic_regression(
  Xmm = model.matrix(~ . - type, Pima.te), 
  ybin = ybin
)
phat = predict(flr, model.matrix(~ . - type, Pima.te))
confusion_results(phat > 0.5, ybin)
#> $confusion_sums
#>       0  1 sum
#> 0   201 22 223
#> 1    46 63 109
#> sum 247 85 332
#> 
#> $confusion_proportion_and_errors
#>                    0          1 proportion error_rate
#> 0          0.6054217 0.06626506  0.6716867 0.09865471
#> 1          0.1385542 0.18975904  0.3283133 0.42201835
#> proportion 0.7439759 0.25602410  1.0000000         NA
#> error_rate 0.1862348 0.25882353         NA 0.20481928
#> 
```
