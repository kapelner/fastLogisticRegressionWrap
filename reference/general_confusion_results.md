# General Confusion Table and Errors

Provides a confusion table and error metrics for general factor vectors.
There is no need for the same levels in the two vectors.

## Usage

``` r
general_confusion_results(yhat, yfac, proportions_scaled_by_column = FALSE)
```

## Arguments

- yhat:

  The factor predictions

- yfac:

  The true factor responses

- proportions_scaled_by_column:

  When returning the proportion table, scale by column? Default is
  `FALSE` to keep the probabilities unconditional to provide the same
  values as the function `confusion_results`. Set to `TRUE` to
  understand error probabilities by prediction bucket.

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
yhat = array(NA, length(ybin))
yhat[phat <= 1/3] = "no"
yhat[phat >= 2/3] = "yes"
yhat[is.na(yhat)] = "maybe"
general_confusion_results(factor(yhat, levels = c("no", "yes", "maybe")), factor(ybin)) 
#> $confusion_sums
#>      no yes maybe sum
#> 0   182  11    30 223
#> 1    26  50    33 109
#> sum 208  61    63 332
#> 
#> $confusion_proportion_and_errors
#>                    no        yes      maybe proportion error_rate
#> 0          0.54819277 0.03313253 0.09036145  0.6716867  0.1838565
#> 1          0.07831325 0.15060241 0.09939759  0.3283133  0.5412844
#> proportion 0.62650602 0.18373494 0.18975904  1.0000000         NA
#> error_rate 0.12500000 0.18032787 1.00000000         NA  0.3012048
#> 
#you want the "no" to align with 0, the "yes" to align with 1 and the "maybe" to be 
#last to align with nothing
```
