assert_binary_vector_then_cast_to_numeric = function(vec){
  checkmate::assert_choice(class(vec), c("numeric", "integer", "logical"))
  vec = as.numeric(vec)
  if(!(checkmate::testSetEqual(unique(vec), c(0, 1)) | checkmate::testSetEqual(unique(vec), c(0)) | checkmate::testSetEqual(unique(vec), c(1)))){ #binary only
	  stop("Set must consist of zeroes and/or ones.")
  }
  vec
}
assert_model_matrix = function(Xmm){
  checkmate::assert_class(Xmm, "matrix")
  checkmate::assert_numeric(Xmm)
}

#' FastLR Wrapper
#' 
#' Returns most of what you get from glm
#'
#' @param Xmm   						The model.matrix for X (you need to create this yourself before)
#' @param ybin  						The binary response vector
#' @param drop_collinear_variables   	Should we drop perfectly collinear variables? Default is \code{FALSE} to inform the user of the problem.
#' @param lm_fit_tol					When \code{drop_collinear_variables = TRUE}, this is the tolerance to detect collinearity among predictors.
#' 										We use the default value from \code{base::lm.fit}'s which is 1e-7. If you fit the logistic regression and
#' 										still get p-values near 1 indicating high collinearity, we recommend making this value smaller.
#' @param solve_tol                     Tolerance when inverting X^T W X, a quantity needed when computing standard errors of the coefficients. 
#' 										The default is \code{.Machine$double.eps} which is used in \code{base::solve}. Make this value smaller if you 
#' 										still get errors even when setting \code{drop_collinear_variables = TRUE}. You may have to play around with 
#' 										this parameter and \code{lm_fit_tol} a bit to get your desired result.
#' @param ...   						Other arguments to be passed to \code{fastLR} (see documentation there)
#'
#' @return      A list of raw results
#' @export
#' @examples
#'  \dontrun{
#' flr = fast_logistic_regression(
#' 	Xmm = model.matrix(~ . - type, MASS::Pima.te), 
#'  ybin = as.numeric(MASS::Pima.te$type == "Yes")
#' )
#' 	}
fast_logistic_regression = function(Xmm, ybin, drop_collinear_variables = FALSE, lm_fit_tol = 1e-7, solve_tol = .Machine$double.eps, ...){
  assert_model_matrix(Xmm)
  ybin = assert_binary_vector_then_cast_to_numeric(ybin)
  assert_logical(drop_collinear_variables)
  assert_numeric(lm_fit_tol, lower = 0)
  assert_numeric(solve_tol, lower = 0)
  if (length(ybin) != nrow(Xmm)){
    stop("The number of rows in Xmm must be equal to the length of ybin")
  }
  #cat("ncol Xmm:", ncol(Xmm), "\n")
  #cat("rank Xmm:", Matrix::rankMatrix(Xmm), "\n")
  if (drop_collinear_variables){
	  collinear_variables = c()
	  repeat {
		  b = coef(lm(ybin ~ 0 + Xmm))
		  b_NA = b[is.na(b)]
		  if (length(b_NA) == 0){
			  break
		  }
		  bad_var = gsub("Xmm", "", names(b_NA)[1])
		  #cat("bad_var", bad_var, "\n")
		  Xmm = Xmm[, colnames(Xmm) != bad_var] #kill this bad variable!!
		  collinear_variables = c(collinear_variables, bad_var)
	  }
	  warning(paste("Dropped the following variables due to collinearity:\n", paste0(collinear_variables, collapse = ", ")))
	  
	  #cat("ncol Xmm after:", ncol(Xmm), "\n")
	  #cat("rank Xmm after:", Matrix::rankMatrix(Xmm), "\n")
	  b = coef(lm.fit(Xmm, ybin, tol = lm_fit_tol))
	  #print(b)
	  #solve(t(Xmm) %*% Xmm, tol = inversion_tol)
	  rm(b, b_NA, bad_var, collinear_variables)
  }
  
  flr = RcppNumerical::fastLR(Xmm, ybin, ...)
  flr$Xmm = Xmm
  flr$ybin = ybin
  flr$regressor_names = colnames(Xmm)
  b = flr$coefficients
  
  #print(b)
  #we just need the std errors of the coefficient estimators 
  #we compute them via notes found in https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture26.pdf
  exp_Xmm_dot_b = exp(Xmm %*% b)
  Wmat = diag(as.numeric(exp_Xmm_dot_b / (1  + exp_Xmm_dot_b)^2))
  XmmtWmatXmm = t(Xmm) %*% Wmat %*% Xmm
  
  tryCatch({
	XmmtWmatXmminv = solve(XmmtWmatXmm, tol = solve_tol) #NOTE: Matrix::chol2inv is slightly faster, but it requires another package
  }, 
  error = function(e){
	print(e)
	stop("Error in inverting X^T X.\nTry setting drop_collinear_variables = TRUE\nto automatically drop perfectly collinear variables\nand if that didn't work then additionally try\nsetting the \"solve_tol\" argument to a value less than the \"reciprocal condition number\" above.\n")
  })
  
  flr$se = sqrt(diag(XmmtWmatXmminv))
  flr$z = b / flr$se
  flr$approx_pval = 2 * pnorm(-abs(flr$z))
  #return
  class(flr) = "fast_logistic_regression"
  flr
}

#' FastLR Wrapper Summary
#' 
#' Returns the summary table a la glm
#'
#' @param object     The object built using the \code{fast_logistic_regression} or \code{fast_logistic_regression_stepwise} wrapper functions
#' @param ...        Other arguments to be passed to \code{summary}.
#'
#' @return           The summary as a data.frame
#' @export
#' @examples
#'  \dontrun{
#' flr = fast_logistic_regression(
#' 	Xmm = model.matrix(~ . - type, MASS::Pima.te), 
#'  ybin = as.numeric(MASS::Pima.te$type == "Yes"))
#' summary(flr)
#' 	}
summary.fast_logistic_regression = function(object, ...){
  checkmate::assert_choice(class(object), c("fast_logistic_regression", "fast_logistic_regression_stepwise"))
  if (!object$converged){
    warning("fast LR did not converge")
  }
  df = data.frame(
    approx_coef = object$coefficients,
    approx_se = object$se,
    approx_z = object$z,
    approx_pval = object$approx_pval,
    signif = ifelse(object$approx_pval < 0.001, "***", ifelse(object$approx_pval < 0.01, "**", ifelse(object$approx_pval < 0.05, "*", "")))
  )
  rownames(df) = object$regressor_names
  df
}

#' FastLR Wrapper Summary
#' 
#' Returns the summary table a la glm
#'
#' @param object     The object built using the \code{fast_logistic_regression_stepwise} wrapper functions
#' @param ...        Other arguments to be passed to \code{summary}.
#'
#' @return           The summary as a data.frame
#' @export
#' @examples
#'  \dontrun{
#' flr = fast_logistic_regression_stepwise_forward(
#' 	Xmm = model.matrix(~ . - type, MASS::Pima.te), 
#'  ybin = as.numeric(MASS::Pima.te$type == "Yes"))
#' summary(flr)
#' 	}
summary.fast_logistic_regression_stepwise = function(object, ...){
	checkmate::assert_class(object, "fast_logistic_regression_stepwise")
	summary(object$flr, ...)
}

#' FastLR Wrapper Print
#' 
#' Returns the summary table a la glm
#'
#' @param x     The object built using the \code{fast_logistic_regression} or \code{fast_logistic_regression_stepwise} wrapper functions
#' @param ...   Other arguments to be passed to print
#' 
#' @return      The summary as a data.frame
#' @export
#' @examples
#'  \dontrun{
#' flr = fast_logistic_regression(
#' 	Xmm = model.matrix(~ . - type, MASS::Pima.te), 
#'  ybin = as.numeric(MASS::Pima.te$type == "Yes"))
#' print(flr)
#' 	}
print.fast_logistic_regression = function(x, ...){
	summary(x, ...)
}

#' FastLR Wrapper Print
#' 
#' Returns the summary table a la glm
#'
#' @param x     The object built using the \code{fast_logistic_regression} or \code{fast_logistic_regression_stepwise} wrapper functions
#' @param ...   Other arguments to be passed to print
#' 
#' @return      The summary as a data.frame
#' @export
#' @examples
#'  \dontrun{
#' flr = fast_logistic_regression_stepwise_forward(
#' 	Xmm = model.matrix(~ . - type, MASS::Pima.te), 
#'  ybin = as.numeric(MASS::Pima.te$type == "Yes"))
#' print(flr)
#' 	}
print.fast_logistic_regression_stepwise = function(x, ...){
	summary(x$flr, ...)
}


#' FastLR Wrapper Predictions
#' 
#' Predicts returning p-hats
#'
#' @param object     The object built using the \code{fast_logistic_regression} or \code{fast_logistic_regression_stepwise} wrapper functions
#' @param newdata    A matrix of observations where you wish to predict the binary response.
#' @param type       The type of prediction required. The default is \code{"response"} which is on the response scale (i.e. probability estimates) and the alternative is \code{"link"} which is the linear scale (i.e. log-odds).
#' @param ...        Further arguments passed to or from other methods
#' 
#' @return           A numeric vector of length \code{nrow(newdata)} of estimates of P(Y = 1) for each unit in \code{newdata}.
#' @export
#' @examples
#'  \dontrun{
#' flr = fast_logistic_regression(
#'   Xmm = model.matrix(~ . - type, MASS::Pima.te), 
#'   ybin = as.numeric(MASS::Pima.te$type == "Yes")
#' )
#' phat = predict(flr, model.matrix(~ . - type, MASS::Pima.te))
#' 	}
predict.fast_logistic_regression = function(object, newdata, type = "response", ...){
  checkmate::assert_class(object, "fast_logistic_regression")
  assert_model_matrix(newdata)
  checkmate::assert_choice(type, c("link", "response"))
  
  #if new_data has more features than training data, we can subset it
  old_data_features = colnames(object$Xmm)
  newdata = newdata[, old_data_features]
  
  #now we need to make sure newdata is legal
  new_data_features = colnames(newdata)
  if (length(new_data_features) != length(old_data_features)){
    stop("newdata has ", length(new_data_features), " features and training data has ", length(old_data_features))
  }
  # new_features_minus_old_features = setdiff(new_data_features, old_data_features)
  # if (length(setdiff(new_features_minus_old_features)) > 0){
  #   stop("newdata must have same columns as the original training data matrix in the same order.\nHere, newdata has features\n", paste(new_features_minus_old_features, collapse = ", "), "\nwhich training data did not have")
  # }
  new_features_minus_old_features = setdiff(old_data_features, new_data_features)
  if (!all(colnames(newdata) == colnames(object$Xmm))){
    stop("newdata must have same columns as the original training data matrix in the same order.\nHere, training data has features\n", paste(new_features_minus_old_features, collapse = ", "), "\nwhich newdata did not have")
  }
  if (!all(new_data_features == old_data_features)){
    stop("newdata must have same columns as the original training data matrix in the same order. They have the same features but the order is different.")
  }
  if (!object$converged){
    warning("fast LR did not converge")
  }
  log_odds_predictions = c(newdata %*% object$coefficients)
  if (type == "response"){
    exp_Xmm_dot_b = exp(log_odds_predictions)
    exp_Xmm_dot_b / (1  + exp_Xmm_dot_b)
  } else if (type == "link"){
    log_odds_predictions
  }
}

#' FastLR Wrapper Predictions
#' 
#' Predicts returning p-hats
#'
#' @param object     The object built using the \code{fast_logistic_regression} or \code{fast_logistic_regression_stepwise} wrapper functions
#' @param newdata    A matrix of observations where you wish to predict the binary response.
#' @param type       The type of prediction required. The default is \code{"response"} which is on the response scale (i.e. probability estimates) and the alternative is \code{"link"} which is the linear scale (i.e. log-odds).
#' @param ...        Further arguments passed to or from other methods
#' 
#' @return           A numeric vector of length \code{nrow(newdata)} of estimates of P(Y = 1) for each unit in \code{newdata}.
#' @export
#' @examples
#'  \dontrun{
#' flr = fast_logistic_regression_stepwise_forward(
#'   Xmm = model.matrix(~ . - type, MASS::Pima.te), 
#'   ybin = as.numeric(MASS::Pima.te$type == "Yes")
#' )
#' phat = predict(flr, model.matrix(~ . - type, MASS::Pima.te))
#' 	}
predict.fast_logistic_regression_stepwise = function(object, newdata, type = "response", ...){	
	checkmate::assert_class(object, "fast_logistic_regression_stepwise")
	predict.fast_logistic_regression(object$flr, newdata, type = "response", ...)
}

#' Rapid Forward Stepwise Logistic Regression
#' 
#' Returns most of what you get from glm
#'
#' @param Xmm             			The model.matrix for X (you need to create this yourself before)
#' @param ybin            			The binary response vector
#' @param pval_threshold  			The significance threshold to include a new variable. Default is \code{0.05}.
#' @param use_intercept   			Should we automatically begin with an intercept? Default is \code{TRUE}.
#' @param drop_collinear_variables 	Parameter used in \code{fast_logistic_regression}. See documentation there.
#' @param lm_fit_tol	  			Parameter used in \code{fast_logistic_regression}. See documentation there.
#' @param solve_tol       			Parameter used in \code{fast_logistic_regression}. See documentation there.
#' @param verbose         			Print out messages during the loop? Default is \code{TRUE}.
#' @param ...             			Other arguments to be passed to \code{fastLR} (see documentation there)
#'
#' @return                A list of raw results
#' @export
#' @examples
#'  \dontrun{
#' flr = fast_logistic_regression_stepwise_forward(
#'   Xmm = model.matrix(~ . - type, MASS::Pima.te), 
#'   ybin = as.numeric(MASS::Pima.te$type == "Yes")
#' )
#' 	}
fast_logistic_regression_stepwise_forward = function(
		Xmm, 
		ybin, 
		pval_threshold = 0.05, 
		use_intercept = TRUE, 
		verbose = TRUE, 
		drop_collinear_variables = FALSE, 
		lm_fit_tol = 1e-7, 
		solve_tol = .Machine$double.eps, 
		...){
  assert_model_matrix(Xmm)
  ybin = assert_binary_vector_then_cast_to_numeric(ybin)
  if (length(ybin) != nrow(Xmm)){
    stop("The number of rows in Xmm must be equal to the length of ybin")
  }
  assert_numeric(pval_threshold, lower = .Machine$double.eps, upper = 1 - .Machine$double.eps)
  assert_logical(use_intercept)
  assert_logical(verbose)
  
  #create starting point
  n = nrow(Xmm)
  p = ncol(Xmm)
  if (use_intercept){
    if (unique(Xmm[, 1]) == 1){
      Xmmt = Xmm[, 1, drop = FALSE]
      js = 1
      iter = 1
      if (verbose){
        cat("iteration #", iter, "of possibly", p, "added intercept", "\n")
      }
    } else {
      Xmmt = matrix(1, nrow = n, ncol = 1)
      colnames(Xmmt) = "(Intercept)"
      js = 0
      iter = 0
    }
  } else {
    Xmmt = matrix(NA, nrow = n, ncol = 0)
    js = 0
    iter = 0
  }
  pvals_star = c()
  repeat {
    js_to_try = setdiff(1 : p, js)
    if (length(js_to_try) == 0){
      break
    }
    pvals = array(NA, p)
    for (i_j in 1 : length(js_to_try)){
      j = js_to_try[i_j]
      Xmmtemp = Xmmt
      Xmmtemp = cbind(Xmmtemp, Xmm[, j, drop = FALSE])
      # tryCatch({
        flrtemp = fast_logistic_regression(Xmmtemp, ybin, drop_collinear_variables, lm_fit_tol, solve_tol)
        pvals[j] = flrtemp$approx_pval[ncol(Xmmtemp)] #the last one
        cat("   sub iteration #", i_j, "of", length(js_to_try), "with feature", colnames(Xmm)[j], "resulted in pval", pvals[j], "\n")
      # }, error = function(e){
      #   cat("   iter #", i_j, "of", length(js_to_try), "with feature", colnames(Xmm)[j], "resulted in ERROR\n")
      # })
    }
    if (!any(pvals < pval_threshold, na.rm = TRUE)){
      break
    }
    j_star = which.min(pvals)
    js = c(js, j_star)
    pvals_star = c(pvals_star, pvals[j_star])
    Xmmt = cbind(Xmmt, Xmm[, j_star, drop = FALSE])
    
    iter = iter + 1
    if (verbose){
      cat("iteration #", iter, "of possibly", p, "added feature #", j_star, "named", colnames(Xmm)[j_star], "with pval", pvals[j_star], "\n")
    }
  }
  #return some information you would like to see
  flr_stepwise = list(js = js, pvals_star = pvals_star, flr = fast_logistic_regression(Xmmt, ybin, drop_collinear_variables, lm_fit_tol, solve_tol))
  class(flr_stepwise) = "fast_logistic_regression_stepwise"
  flr_stepwise
}

#' Binary Confusion Table and Errors
#' 
#' Provides a binary confusion table and error metrics
#'
#' @param yhat            The binary predictions
#' @param ybin            The true binary responses
#'
#' @return                A list of raw results
#' @export
#' @examples
#'  \dontrun{
#' ybin = as.numeric(MASS::Pima.te$type == "Yes")
#' flr = fast_logistic_regression(
#'   Xmm = model.matrix(~ . - type, MASS::Pima.te), 
#'   ybin = ybin
#' )
#' phat = predict(flr, model.matrix(~ . - type, MASS::Pima.te))
#' confusion_results(phat > 0.5, ybin)
#' 	}
confusion_results = function(yhat, ybin){
  yhat = assert_binary_vector_then_cast_to_numeric(yhat)
  ybin = assert_binary_vector_then_cast_to_numeric(ybin)
  n = length(yhat)
  if (n != length(ybin)){
    stop("yhat and ybin must be same length")
  }
  conf = table(ybin, yhat)
  tp = conf[2, 2]
  tn = conf[1, 1]
  fp = conf[1, 2]
  fn = conf[2, 1]
  
  fdr =  fp / sum(conf[, 2])
  fomr = fn / sum(conf[, 1])
  fpr = fp / sum(conf[1, ])
  fnr = fn / sum(conf[2, ])
  
  confusion_sums = matrix(NA, 3, 3)
  confusion_sums[1 : 2, 1 : 2] = conf
  confusion_sums[1, 3] = tn + fp
  confusion_sums[2, 3] = fn + tp
  confusion_sums[3, 3] = n
  confusion_sums[3, 1] = tn + fn
  confusion_sums[3, 2] = fp + tp
  colnames(confusion_sums) = c("0", "1", "sum")
  rownames(confusion_sums) = c("0", "1", "sum")
  
  confusion_proportion_and_errors = matrix(NA, 4, 4)
  confusion_proportion_and_errors[1 : 3, 1 : 3] = confusion_sums / n
  confusion_proportion_and_errors[1, 4] = fpr
  confusion_proportion_and_errors[2, 4] = fnr
  confusion_proportion_and_errors[4, 1] = fomr
  confusion_proportion_and_errors[4, 2] = fdr
  confusion_proportion_and_errors[4, 4] = (fp + fn) / n  
  colnames(confusion_proportion_and_errors) = c("0", "1", "proportion", "error_rate")
  rownames(confusion_proportion_and_errors) = c("0", "1", "proportion", "error_rate")
  
  list(
    confusion_sums = confusion_sums,
    confusion_proportion_and_errors = confusion_proportion_and_errors
  )
}

#' General Confusion Table and Errors
#' 
#' Provides a confusion table and error metrics for general factor vectors.
#' There is no need for the same levels in the two vectors.
#'
#' @param yhat            				The factor predictions
#' @param yfac            				The true factor responses
#' @param proportions_scaled_by_column	When returning the proportion table, scale by column? Default is \code{FALSE} to keep the probabilities 
#' 										unconditional to provide the same values as the function \code{confusion_results}. Set to \code{TRUE}
#' 										to understand error probabilities by prediction bucket.
#'
#' @return                				A list of raw results
#' @export
#' @examples
#'  \dontrun{
#' ybin = as.numeric(MASS::Pima.te$type == "Yes")
#' flr = fast_logistic_regression(
#'   Xmm = model.matrix(~ . - type, MASS::Pima.te), 
#'   ybin = ybin
#' )
#' phat = predict(flr, model.matrix(~ . - type, MASS::Pima.te))
#' yhat = array(NA, length(ybin))
#' yhat[phat <= 1/3] = "no"
#' yhat[phat >= 2/3] = "yes"
#' yhat[is.na(yhat)] = "maybe"
#' general_confusion_results(factor(yhat, levels = c("no", "yes", "maybe")), factor(ybin)) 
#' #you want the "no" to align with 0, the "yes" to align with 1 and the "maybe" to be 
#' #last to align with nothing
#' 	}
general_confusion_results = function(yhat, yfac, proportions_scaled_by_column = FALSE){
	assert_factor(yhat)
	assert_factor(yfac)
	n = length(yhat)
	if (n != length(yfac)){
		stop("yhat and yfac must be same length")
	}
	levels_yfac = levels(yfac)
	levels_yhat = levels(yhat)
	n_r_conf = length(levels_yfac)
	n_c_conf = length(levels_yhat)
	conf = matrix(table(yfac, yhat), ncol = n_c_conf, nrow = n_r_conf)
	rownames(conf) = levels_yfac
	colnames(conf) = levels_yhat
	
	confusion_sums = matrix(NA, n_r_conf + 1, n_c_conf + 1)
	confusion_sums[1 : n_r_conf, 1 : n_c_conf] = conf
	confusion_sums[n_r_conf + 1, 1 : n_c_conf] = colSums(conf)
	confusion_sums[1 : n_r_conf, n_c_conf + 1] = rowSums(conf)
	confusion_sums[n_r_conf + 1, n_c_conf + 1] = n
	rownames(confusion_sums) = c(levels_yfac, "sum")
	colnames(confusion_sums) = c(levels_yhat, "sum")
	
	confusion_proportion_and_errors = matrix(NA, n_r_conf + 2, n_c_conf + 2)
	if (proportions_scaled_by_column){
		for (j in 1 : p){
			confusion_proportion_and_errors[1 : n_r_conf, j] = conf[, j] / sum(conf[, j])		
		}
		confusion_proportion_and_errors[1 : (n_r_conf + 1), p + 1] = confusion_sums[, p + 1] / sum(confusion_sums[, p + 1])
		confusion_proportion_and_errors[(n_r_conf + 1), 1 : (n_c_conf + 1)] = confusion_sums[(n_r_conf + 1), ] / n
	} else {
		confusion_proportion_and_errors[1 : (n_r_conf + 1), 1 : (n_c_conf + 1)] = confusion_sums / n
	}
	
	
	#now calculate all types of errors
	p = min(dim(conf))
	n_correct_classifications = 0
	for (j in 1 : p){
		n_correct_classifications = n_correct_classifications + conf[j, j]
	}	
	for (j in 1 : n_r_conf){
		if (j <= p){
			j_row_sum = sum(conf[j, ])
			confusion_proportion_and_errors[j, n_c_conf + 2] = (j_row_sum - conf[j, j]) / j_row_sum
		} else {
			confusion_proportion_and_errors[j, n_c_conf + 2] = 1
		}
		
	}	
	for (j in 1 : n_c_conf){
		if (j <= p){
			j_col_sum = sum(conf[, j])
			confusion_proportion_and_errors[n_r_conf + 2, j] = (j_col_sum - conf[j, j]) / j_col_sum
		} else {
			confusion_proportion_and_errors[n_r_conf + 2, j] = 1
		}		
	}
	confusion_proportion_and_errors[n_r_conf + 2, n_c_conf + 2] = (n - n_correct_classifications) / n 
	rownames(confusion_proportion_and_errors) = c(levels_yfac, "proportion", "error_rate")
	colnames(confusion_proportion_and_errors) = c(levels_yhat, "proportion", "error_rate")
	
	list(
		confusion_sums = confusion_sums,
		confusion_proportion_and_errors = confusion_proportion_and_errors
	)
}


#how much faster is fastLR over glm?
# set.seed(123)
# n = 100
# p = 1
# x = matrix(rnorm(n * p), n)
# beta = runif(p)
# xb = c(x %*% beta)
# p = 1 / (1 + exp(-xb))
# y = rbinom(n, 1, p)
# 
# system.time(res1 <- glm.fit(x, y, family = binomial()))
# system.time(res2 <- fastLR(x, y))
# max(abs(res1$coefficients - res2$coefficients))

#unit test
# ybin_test = as.numeric(MASS::Pima.te$type == "Yes")
# summary(glm(ybin_test ~ . - type, MASS::Pima.te, family = "binomial"))
# flr = fast_logistic_regression(Xmm = model.matrix(~ . - type, MASS::Pima.te), ybin = ybin_test)
# summary(flr)
