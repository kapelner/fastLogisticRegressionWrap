# FastLR Wrapper Predictions

Predicts returning p-hats

## Usage

``` r
# S3 method for class 'fast_logistic_regression'
predict(object, newdata, type = "response", ...)
```

## Arguments

- object:

  The object built using the `fast_logistic_regression` or
  `fast_logistic_regression_stepwise` wrapper functions

- newdata:

  A matrix of observations where you wish to predict the binary
  response.

- type:

  The type of prediction required. The default is `"response"` which is
  on the response scale (i.e. probability estimates) and the alternative
  is `"link"` which is the linear scale (i.e. log-odds).

- ...:

  Further arguments passed to or from other methods

## Value

A numeric vector of length `nrow(newdata)` of estimates of P(Y = 1) for
each unit in `newdata`.

## Examples

``` r
library(MASS); data(Pima.te)
flr = fast_logistic_regression(
  Xmm = model.matrix(~ . - type, Pima.te), 
  ybin = as.numeric(Pima.te$type == "Yes")
)
phat = predict(flr, model.matrix(~ . - type, Pima.te))
```
