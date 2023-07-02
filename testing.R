pacman::p_load(fastLogisticRegressionWrap, microbenchmark)

p = 100
M = matrix(rnorm(p^2), nrow = p)
fastLogisticRegressionWrap::eigen_det(M)

#ensure equality with glm
ybin_test = as.numeric(MASS::Pima.te$type == "Yes")
summary(glm(ybin_test ~ . - type, MASS::Pima.te, family = "binomial"))
flr = fast_logistic_regression(Xmm = model.matrix(~ . - type, MASS::Pima.te), ybin = ybin_test, do_inference_on_var = TRUE)
summary(flr)


Rcpp::cppFunction(depends = "RcppEigen", '


Eigen::MatrixXd eigen_inv(const Eigen::Map<Eigen::MatrixXd> X, int n_cores) {
  Eigen::setNbThreads(n_cores);
  return X.inverse();
}

double eigen_det(const Eigen::Map<Eigen::MatrixXd> X, int n_cores) {
  Eigen::setNbThreads(n_cores);
  return X.determinant();
}

')

p = 100
M = matrix(rnorm(p^2), nrow = p)
fastLogisticRegressionWrap::eigen_det(M)

microbenchmark(
  R_inv = diag(solve(M)),
  R_inv_single_entry = c(rep(0, p-1), det(M[-1,-1]) / det(M)),
  R_inv_two_entry = {
    det_M = det(M)
    diag_vec = array(NA, p)
    diag_vec[1] = det(M[-1,-1]) / det_M
    diag_vec[2] = det(M[-2,-2]) / det_M
  },  
  R_inv_three_entry = {
    det_M = det(M)
    diag_vec = array(NA, p)
    diag_vec[1] = det(M[-1,-1]) / det_M
    diag_vec[2] = det(M[-2,-2]) / det_M
    diag_vec[3] = det(M[-3,-3]) / det_M
  }, 
  R_inv_four_entry = {
    det_M = det(M)
    diag_vec = array(NA, p)
    diag_vec[1] = det(M[-1,-1]) / det_M
    diag_vec[2] = det(M[-2,-2]) / det_M
    diag_vec[3] = det(M[-3,-3]) / det_M
    diag_vec[4] = det(M[-4,-4]) / det_M
  },
  source_eigen_inv = eigen_inv(M, 1),
  source_eigen_inv_single_entry = c(rep(0, p-1), eigen_det(M[-1,-1], 1) / eigen_det(M, 1)),
  source_eigen_inv_two_entry = {
    det_M = eigen_det(M, 1)
    diag_vec = array(NA, p)
    diag_vec[1] = eigen_det(M[-1,-1], 1) / det_M
    diag_vec[2] = eigen_det(M[-2,-2], 1) / det_M
  },  
  source_eigen_inv_three_entry = {
    det_M = eigen_det(M, 1)
    diag_vec = array(NA, p)
    diag_vec[1] = eigen_det(M[-1,-1], 1) / det_M
    diag_vec[2] = eigen_det(M[-2,-2], 1) / det_M
    diag_vec[3] = eigen_det(M[-3,-3], 1) / det_M
  }, 
  source_eigen_inv_four_entry = {
    det_M = eigen_det(M, 1)
    diag_vec = array(NA, p)
    diag_vec[1] = eigen_det(M[-1,-1], 1) / det_M
    diag_vec[2] = eigen_det(M[-2,-2], 1) / det_M
    diag_vec[3] = eigen_det(M[-3,-3], 1) / det_M
    diag_vec[4] = eigen_det(M[-4,-4], 1) / det_M
  },
  
  package_eigen_inv = fastLogisticRegressionWrap::eigen_inv(M, 1),
  package_eigen_inv_single_entry = c(rep(0, p-1), fastLogisticRegressionWrap::eigen_det(M[-1,-1], 1) / fastLogisticRegressionWrap::eigen_det(M, 1)),
  package_eigen_inv_two_entry = {
    det_M = fastLogisticRegressionWrap::eigen_det(M, 1)
    diag_vec = array(NA, p)
    diag_vec[1] = fastLogisticRegressionWrap::eigen_det(M[-1,-1], 1) / det_M
    diag_vec[2] = fastLogisticRegressionWrap::eigen_det(M[-2,-2], 1) / det_M
  },  
  package_eigen_inv_three_entry = {
    det_M = fastLogisticRegressionWrap::eigen_det(M, 1)
    diag_vec = array(NA, p)
    diag_vec[1] = fastLogisticRegressionWrap::eigen_det(M[-1,-1], 1) / det_M
    diag_vec[2] = fastLogisticRegressionWrap::eigen_det(M[-2,-2], 1) / det_M
    diag_vec[3] = fastLogisticRegressionWrap::eigen_det(M[-3,-3], 1) / det_M
  }, 
  package_eigen_inv_four_entry = {
    det_M = fastLogisticRegressionWrap::eigen_det(M, 1)
    diag_vec = array(NA, p)
    diag_vec[1] = fastLogisticRegressionWrap::eigen_det(M[-1,-1], 1) / det_M
    diag_vec[2] = fastLogisticRegressionWrap::eigen_det(M[-2,-2], 1) / det_M
    diag_vec[3] = fastLogisticRegressionWrap::eigen_det(M[-3,-3], 1) / det_M
    diag_vec[4] = fastLogisticRegressionWrap::eigen_det(M[-4,-4], 1) / det_M
  },
  times = 10 
)


pacman::p_load(fastLogisticRegressionWrap, microbenchmark)

set.seed(123)
n = 5000
p = 500 #must be even
X = cbind(1, matrix(rnorm(n * p), n))
# colnames(X) = c("(intercept)", paste0("x_useful_", 1 : (p / 2)), paste0("x_useless_", 1 : (p / 2)))
beta = c(runif(p / 2 + 1), rep(0, p / 2))
y = rbinom(n, 1, 1 / (1 + exp(-c(X %*% beta))))

microbenchmark(
  glm = glm(y ~ 0 + ., data.frame(X), family = binomial()),
  flr = fast_logistic_regression(X, y, do_inference_on_var = TRUE),
  times = 3
)


# flr = fast_logistic_regression(X, y, do_inference_on_var = TRUE)
# flr_stepwise = fast_logistic_regression_stepwise_forward(X, y, drop_collinear_variables = FALSE, verbose = TRUE, mode = "aic")

set.seed(123)
n = 5000
p = 500 #must be even
X = cbind(1, matrix(rnorm(n * p), n))
# colnames(X) = c("(intercept)", paste0("x_useful_", 1 : (p / 2)), paste0("x_useless_", 1 : (p / 2)))
beta = c(runif(p / 2 + 1), rep(0, p / 2))
y = rbinom(n, 1, 1 / (1 + exp(-c(X %*% beta))))

microbenchmark(
  flr1 = fast_logistic_regression(X, y, do_inference_on_var = TRUE, num_cores = 1),
  flr2 = fast_logistic_regression(X, y, do_inference_on_var = TRUE, num_cores = 2),
  flr4 = fast_logistic_regression(X, y, do_inference_on_var = TRUE, num_cores = 4),
  times = 10
)

n = 1000
p = 100 #must be even
X = cbind(1, matrix(rnorm(n * p), n))
beta = c(runif(p / 2 + 1), rep(0, p / 2))
y = rbinom(n, 1, 1 / (1 + exp(-c(X %*% beta))))

microbenchmark( #stepwise benchmark
  glm_stepwise = {
    fullmod = glm(y ~ 0 + ., data = data.frame(X), family = binomial)
    nothing = glm(y ~ 0,     data = data.frame(X), family = binomial)
    forwards = step(nothing, scope = list(lower = formula(nothing), 
                      upper = formula(fullmod)), direction = "forward", trace = 0)
  },
  flr_stepwise = fast_logistic_regression_stepwise_forward(X, y, verbose = FALSE),
  times = 3
)



system.time(res1 <- glm(y ~ 0 + X, family = binomial()))
system.time(res2 <- fast_logistic_regression(X, y, do_inference_on_var = TRUE))
system.time(res3 <- fast_logistic_regression(X, y, do_inference_on_var = c(TRUE, rep(FALSE, p))))
summary(res1)
summary(res2)
summary(res3)

solve_tol = .Machine$double.eps
system.time({XmmtWmatXmminv = solve(XmmtWmatXmm, tol = solve_tol)}) #NOTE: Matrix::chol2inv is slightly faster, but it requires another package
system.time({flr$se = sqrt(diag(XmmtWmatXmminv))})
system.time({flr$z = b / flr$se})
flr$approx_pval = 2 * pnorm(-abs(flr$z))





############# dead
# Rcpp::cppFunction(depends = "RcppArmadillo", ' 
# arma::mat sandwich_multiply( arma::mat X, arma::mat W) {
#   return X.t() * W * X;
# }
# ')
# Rcpp::cppFunction(depends = "RcppEigen", ' 
# SEXP sandwich_multiply(const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::MatrixXd> W, int n_cores) {
#   Eigen::setNbThreads(n_cores);
#   Eigen::MatrixXd C = X.transpose() * W * X;
#   return Rcpp::wrap(C);
# }
# ')