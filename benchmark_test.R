library(fastLogisticRegressionWrap)
# Welcome to fastLogisticRegressionWrap v1.0.1.
library(microbenchmark)

set.seed(123)
n = 5000
p = 500 #must be even
X = cbind(1, matrix(rnorm(n * p), n))
# colnames(X) = c("(intercept)", paste0("x_useful_", 1 : (p / 2)), paste0("x_useless_", 1 : (p / 2)))
beta = c(runif(p / 2 + 1), rep(0, p / 2))
y = rbinom(n, 1, 1 / (1 + exp(-c(X %*% beta))))

microbenchmark(
  glm = glm(y ~ 0 + ., data.frame(X), family = "binomial"),
  flr = fast_logistic_regression(X, y, do_inference_on_var = "all"),
  times = 10
)


set.seed(123)
n = 5000
p = 500 #must be even
X = cbind(1, matrix(rnorm(n * p), n))
# colnames(X) = c("(intercept)", paste0("x_useful_", 1 : (p / 2)), paste0("x_useless_", 1 : (p / 2)))
beta = c(runif(p / 2 + 1), rep(0, p / 2))
y = rbinom(n, 1, 1 / (1 + exp(-c(X %*% beta))))
j = 137

microbenchmark(
  glm = coef(summary(glm(y ~ 0 + ., data.frame(X), family = "binomial")))[j, 4],
  flr_all = fast_logistic_regression(X, y, do_inference_on_var = "all")$approx_pval[j],
  flr_j = fast_logistic_regression(X, y, do_inference_on_var = j)$approx_pval[j],
  times = 10
)

