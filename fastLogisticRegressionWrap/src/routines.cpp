#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;

// [[Rcpp::export]]
Eigen::MatrixXd eigen_Xt_times_diag_w_times_X_cpp(const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::VectorXd> w, int n_cores) {
  Eigen::setNbThreads(n_cores);
  return X.transpose() * w.asDiagonal() * X;
}

// [[Rcpp::export]]
Eigen::MatrixXd eigen_inv_cpp(const Eigen::Map<Eigen::MatrixXd> X, int n_cores) {
  Eigen::setNbThreads(n_cores);
  return X.inverse();
}

// [[Rcpp::export]]
double eigen_det_cpp(const Eigen::Map<Eigen::MatrixXd> X, int n_cores) {
  Eigen::setNbThreads(n_cores);
  return X.determinant();
}

//// [[Rcpp::export]]
//Eigen::MatrixXd least_squares_coefficient_estimate_cpp(const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::VectorXd> y, int n_cores) {
//  Eigen::setNbThreads(n_cores);
//  return (X.transpose() * X).inverse() * y;
//}
