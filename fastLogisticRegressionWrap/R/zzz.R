.onAttach = function(libname, pkgname){
	packageStartupMessage(paste(
		"\nWelcome to fastLogisticRegressionWrap v", utils::packageVersion("fastLogisticRegressionWrap"), ".\n", 
		sep = ""
	))
}