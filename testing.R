pacman::p_load(fastLogisticRegressionWrap, microbenchmark)

p = 100
M = matrix(rnorm(p^2), nrow = p)
fastLogisticRegressionWrap::eigen_det(M)

#ensure equality with glm
ybin_test = as.numeric(MASS::Pima.te$type == "Yes")
summary(glm(ybin_test ~ . - type, MASS::Pima.te, family = "binomial"))
flr = fast_logistic_regression(Xmm = model.matrix(~ . - type, MASS::Pima.te), ybin = ybin_test, do_inference_on_var = TRUE)
summary(flr)


# glm_obj = glm.fit(x, y, family = binomial())
# fastlr_obj <- RcppNumerical::fastLR(x, y)
# 
# microbenchmark(
#   glm_fit = glm.fit(x, y, family = binomial()),
#   fast_lr_fit = RcppNumerical::fastLR(x, y),
#   times = 10
# )

# system.time(res1 <- glm.fit(x, y, family = binomial()))
# system.time(res2 <- RcppNumerical::fastLR(x, y))
# system.time(res2 <- fast_logistic_regression(x, y))
# system.time({qr_obj = qr(x); qr.Q(qr_obj); qr.R(qr_obj)})
# 
# system.time(res1 <- glm(y ~ x, family = binomial()))
# system.time(res2 <- fast_logistic_regression(x, y, do_inference_on_var = TRUE))
flr <- fast_logistic_regression(x, y)
b = flr$coefficients
system.time({exp_Xmm_dot_b = exp(flr$Xmm %*% b)})
system.time({Wmat = diag(as.numeric(exp_Xmm_dot_b / (1  + exp_Xmm_dot_b)^2))})
# system.time({XmmtWmatXmm = t(flr$Xmm) %*% Wmat %*% flr$Xmm})



Rcpp::cppFunction(depends = "RcppEigen", '

Eigen::MatrixXd sandwich_multiply_diag(const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::VectorXd> w, int n_cores) {
  Eigen::setNbThreads(n_cores);
  return X.transpose() * w.asDiagonal() * X;
}

Eigen::MatrixXd eigen_inv(const Eigen::Map<Eigen::MatrixXd> X, int n_cores) {
  Eigen::setNbThreads(n_cores);
  return X.inverse();
}

double eigen_det(const Eigen::Map<Eigen::MatrixXd> X, int n_cores) {
  Eigen::setNbThreads(n_cores);
  return X.determinant();
}

')

sandwich_multiply(flr$Xmm, Wmat, n_cores = 1)
XmmtWmatXmm = sandwich_multiply_diag(flr$Xmm, diag(Wmat), n_cores = 1)

microbenchmark(
  # R =    t(flr$Xmm) %*% Wmat %*% flr$Xmm,
  eigen_within_R = sandwich_multiply(flr$Xmm, Wmat, n_cores = 1),
  eigen_diag_within_R = sandwich_multiply_diag(flr$Xmm, diag(Wmat), n_cores = 1),
  eigen_diag_within_R_nc_4 = sandwich_multiply_diag(flr$Xmm, diag(Wmat), n_cores = 4),
  # eigen_within_package = fastLogisticRegressionWrap::sandwich_multiply_via_eigen_package_in_cpp(flr$Xmm, Wmat, n_cores = 1),
  times = 1
)

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
n = 10000
p = 400
x = cbind(1, matrix(rnorm(n * p), n))
beta = runif(p + 1)
xb = c(x %*% beta)
prob_y_eq_1 = 1 / (1 + exp(-xb))
y = rbinom(n, 1, prob_y_eq_1)
rm(beta, xb, prob_y_eq_1)

microbenchmark(
  glm = glm(y ~ 0 + x, family = binomial()),
  flr = fast_logistic_regression(x, y, do_inference_on_var = TRUE),
  flr_just_one_coef = fast_logistic_regression(x, y, do_inference_on_var = c(TRUE, rep(FALSE, p))),
  times = 10
)
system.time(res1 <- glm(y ~ 0 + x, family = binomial()))
system.time(res2 <- fast_logistic_regression(x, y, do_inference_on_var = TRUE))
summary(res1)
summary(res2)

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