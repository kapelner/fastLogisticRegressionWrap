.onAttach = function(libname, pkgname){
	packageStartupMessage(paste(
		"Welcome to fastLogisticRegressionWrap v", utils::packageVersion("fastLogisticRegressionWrap"), ".\n", 
		sep = ""
	))
}
#.onLoad = function(libname, pkgname) {	
#	assign("fastLogisticRegressionWrap_globals", new.env(), envir = parent.env(environment()))
#	
#	eigen_Xt_times_diag_w_times_X_cpp = Rcpp::cppFunction(depends = "RcppEigen", '					
#	Eigen::MatrixXd eigen_Xt_times_diag_w_times_X_cpp(const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::VectorXd> w, int n_cores) {
#		Eigen::setNbThreads(n_cores);
#		return X.transpose() * w.asDiagonal() * X;
#	}
#	')
#	assign("eigen_Xt_times_diag_w_times_X_cpp", eigen_Xt_times_diag_w_times_X_cpp, fastLogisticRegressionWrap_globals)
#	
#	eigen_inv_cpp = Rcpp::cppFunction(depends = "RcppEigen", '
#	Eigen::MatrixXd eigen_inv_cpp(const Eigen::Map<Eigen::MatrixXd> X, int n_cores) {
#		Eigen::setNbThreads(n_cores);
#		return X.inverse();
#	}
#	')
#	assign("eigen_inv_cpp", eigen_inv_cpp, fastLogisticRegressionWrap_globals)
#	
#	eigen_det_cpp = Rcpp::cppFunction(depends = "RcppEigen", '					
#	double eigen_det_cpp(const Eigen::Map<Eigen::MatrixXd> X, int n_cores) {
#		Eigen::setNbThreads(n_cores);
#		return X.determinant();
#	}
#	')
#	assign("eigen_det_cpp", eigen_det_cpp, fastLogisticRegressionWrap_globals)	
	
#	least_squares_coefficient_estimate_cpp = Rcpp::cppFunction(depends = "RcppEigen", '	
#	Eigen::MatrixXd least_squares_coefficient_estimate_cpp(const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::VectorXd> y, int n_cores) {
#		Eigen::setNbThreads(n_cores);
#		return (X.transpose() * X).inverse() * y;
#	}
#	')
#	assign("least_squares_coefficient_estimate_cpp", least_squares_coefficient_estimate_cpp, fastLogisticRegressionWrap_globals)	
#}