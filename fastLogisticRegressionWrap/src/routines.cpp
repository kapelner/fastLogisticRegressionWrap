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

// [[Rcpp::export]]
double eigen_compute_single_entry_of_diagonal_matrix_cpp(const Eigen::Map<Eigen::MatrixXd> M, int j, int n_cores) {
  Eigen::setNbThreads(n_cores);

  Eigen::VectorXd b;
  b.resize(M.rows());
  b.setZero();
  b(j - 1) = 1;

  Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> cg;
  cg.compute(M);
  Eigen::VectorXd x = cg.solve(b);

  return x(j - 1);
}

//// [[Rcpp::export]]
//Eigen::MatrixXd least_squares_coefficient_estimate_cpp(const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::VectorXd> y, int n_cores) {
//  Eigen::setNbThreads(n_cores);
//  return (X.transpose() * X).inverse() * y;
//}

// [[Rcpp::export]]
IntegerMatrix fast_two_by_two_binary_table_cpp(const NumericVector ybin, const NumericVector yhat) {
	IntegerMatrix conf(2, 2);
	for (int i = 0; i < ybin.length(); i++){
		double ybin_i = ybin[i];
		double yhat_i = yhat[i];
		if        (ybin_i == 0 && yhat_i == 0){
			conf(0, 0)++;
		} else if (ybin_i == 1 && yhat_i == 0){
			conf(1, 0)++;
		} else if (ybin_i == 0 && yhat_i == 1){
			conf(0, 1)++;
		} else if (ybin_i == 1 && yhat_i == 1){
			conf(1, 1)++;
		}
	}
	return conf;
}
