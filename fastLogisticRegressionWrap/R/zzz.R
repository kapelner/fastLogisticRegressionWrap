.onAttach = function(libname, pkgname){
	packageStartupMessage(paste(
		"Welcome to fastLogisticRegressionWrap v", utils::packageVersion("fastLogisticRegressionWrap"), ".\n", 
		sep = ""
	))
}