# FastLR Wrapper Predictions

Predicts returning p-hats

## Usage

``` r
# S3 method for class 'fast_logistic_regression_stepwise'
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
flr = fast_logistic_regression_stepwise_forward(
  Xmm = model.matrix(~ . - type, Pima.te), 
  ybin = as.numeric(Pima.te$type == "Yes")
)
#> iteration # 1 of possibly 8 added intercept 
#>    sub iteration # 1 of 7 with feature npreg resulted in aic 405.5482 
#>    sub iteration # 2 of 7 with feature glu resulted in aic 329.9911 
#>    sub iteration # 3 of 7 with feature bp resulted in aic 414.44 
#>    sub iteration # 4 of 7 with feature skin resulted in aic 399.9796 
#>    sub iteration # 5 of 7 with feature bmi resulted in aic 390.839 
#>    sub iteration # 6 of 7 with feature ped resulted in aic 403.7897 
#>    sub iteration # 7 of 7 with feature age resulted in aic 398.4388 
#> iteration # 2 of possibly 8 added feature # 3 named glu with aic 329.9911 
#>    sub iteration # 1 of 6 with feature npreg resulted in aic 317.0171 
#>    sub iteration # 2 of 6 with feature bp resulted in aic 329.8766 
#>    sub iteration # 3 of 6 with feature skin resulted in aic 321.1592 
#>    sub iteration # 4 of 6 with feature bmi resulted in aic 318.4944 
#>    sub iteration # 5 of 6 with feature ped resulted in aic 323.5249 
#>    sub iteration # 6 of 6 with feature age resulted in aic 320.868 
#> iteration # 3 of possibly 8 added feature # 2 named npreg with aic 317.0171 
#>    sub iteration # 1 of 5 with feature bp resulted in aic 318.1562 
#>    sub iteration # 2 of 5 with feature skin resulted in aic 309.0735 
#>    sub iteration # 3 of 5 with feature bmi resulted in aic 302.5406 
#>    sub iteration # 4 of 5 with feature ped resulted in aic 311.7246 
#>    sub iteration # 5 of 5 with feature age resulted in aic 318.0768 
#> iteration # 4 of possibly 8 added feature # 6 named bmi with aic 302.5406 
#>    sub iteration # 1 of 4 with feature bp resulted in aic 304.2813 
#>    sub iteration # 2 of 4 with feature skin resulted in aic 303.9285 
#>    sub iteration # 3 of 4 with feature ped resulted in aic 297.441 
#>    sub iteration # 4 of 4 with feature age resulted in aic 303.353 
#> iteration # 5 of possibly 8 added feature # 7 named ped with aic 297.441 
#>    sub iteration # 1 of 3 with feature bp resulted in aic 299.2284 
#>    sub iteration # 2 of 3 with feature skin resulted in aic 298.9583 
#>    sub iteration # 3 of 3 with feature age resulted in aic 298.7261 
phat = predict(flr, model.matrix(~ . - type, Pima.te))
```
