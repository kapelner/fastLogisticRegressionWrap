assert_binary_vector_then_cast_to_numeric = function(vec){
  checkmate::assert_choice(class(vec), c("numeric", "integer", "logical"))
  vec = as.numeric(vec)
  if (!(checkmate::testSetEqual(unique(vec), c(0, 1)) | checkmate::testSetEqual(unique(vec), c(0)) | checkmate::testSetEqual(unique(vec), c(1)))){ #binary only
	  stop("Set must consist of zeroes and/or ones.")
  }
  vec
}

assert_numeric_matrix = function(Xmm){
  checkmate::assert_matrix(Xmm)
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
#' @param do_inference_on_var			Which variables should we compute approximate standard errors of the coefficients and approximate p-values for the test of
#' 										no linear log-odds probability effect? Default is \code{FALSE} for inference on none (for speed). If not default, then \code{TRUE}
#' 										to indicate inference should be computed for all variables. If a logical vector of size \code{ncol(Xmm)} is passed in then
#' 										the indicies of the \code{TRUE} denotes which variables to compute inference for. If variables are dropped when
#' 										\code{drop_collinear_variables = TRUE}, then indicies will likewise be dropped from this vector. We do not recommend using this type
#' 										of piecewise specification until we understand how it behaves in simulation. Note: if you are just comparing
#' 										nested models using anova, there is no need to compute inference for coefficients (keep the default of \code{FALSE} for speed).
#' @param num_cores						Number of cores to use to speed up matrix multiplication and matrix inversion (used only during inference computation). Default is 1.
#' 										Unless the number of variables, i.e. \code{ncol(Xmm)}, is large, there does not seem to be a performance gain in using multiple cores.
#' @param ...   						Other arguments to be passed to \code{fastLR}. See documentation there.
#'
#' @return      A list of raw results
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression(
#' 	 Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = as.numeric(Pima.te$type == "Yes")
#' )
fast_logistic_regression = function(Xmm, ybin, drop_collinear_variables = FALSE, lm_fit_tol = 1e-7, do_inference_on_var = FALSE, num_cores = 1, ...){
  assert_numeric_matrix(Xmm)
  ybin = assert_binary_vector_then_cast_to_numeric(ybin)
  assert_logical(drop_collinear_variables)
  assert_numeric(lm_fit_tol, lower = 0)
  assert_logical(do_inference_on_var)
  assert_count(num_cores, positive = TRUE)
  original_col_names = colnames(Xmm)
  
  p = ncol(Xmm) #the original p before variables are dropped
  if (length(do_inference_on_var) > 1){
	  assert_true(length(do_inference_on_var) == p) 
  } else {
	  do_inference_on_var = rep(do_inference_on_var, p)
  }
  names(do_inference_on_var) = original_col_names
  any_inference_originally = any(do_inference_on_var)
  
  if (length(ybin) != nrow(Xmm)){
    stop("The number of rows in Xmm must be equal to the length of ybin")
  }
  #cat("ncol Xmm:", ncol(Xmm), "\n")
  #cat("rank Xmm:", Matrix::rankMatrix(Xmm), "\n")
	
  variables_retained = rep(TRUE, p)
  names(variables_retained) = original_col_names
  if (drop_collinear_variables){
	  collinear_variables = c()
	  repeat {
		  b = coef(lm.fit(Xmm, ybin, tol = lm_fit_tol))
		  b_NA = b[is.na(b)]
		  if (length(b_NA) == 0){
			  break
		  }
		  bad_var = gsub("Xmm", "", names(b_NA)[1])
		  #cat("bad_var", bad_var, "\n")
		  Xmm = Xmm[, colnames(Xmm) != bad_var] #remove these bad variable(s) from the data!!
		  collinear_variables = c(collinear_variables, bad_var)
	  }
	  #if (length(collinear_variables) > 1){
	  #	  warning(paste("Dropped the following variables due to collinearity:\n", paste0(collinear_variables, collapse = ", ")))
	  #}	  
	  #cat("ncol Xmm after:", ncol(Xmm), "\n")
	  #cat("rank Xmm after:", Matrix::rankMatrix(Xmm), "\n")
	  #b = coef(lm.fit(Xmm, ybin, tol = lm_fit_tol))
	  #print(b)
	  #solve(t(Xmm) %*% Xmm, tol = inversion_tol)
	  do_inference_on_var = do_inference_on_var[!(names(do_inference_on_var) %in% collinear_variables)]
	  if (!any(do_inference_on_var) & any_inference_originally){
		  warning("There is no longer any inference to compute as all variables specified were collinear and thus dropped from the model fit.")
	  }
	  variables_retained[collinear_variables] = FALSE
  }
  
  flr = RcppNumerical::fastLR(Xmm, ybin, ...)
  flr$Xmm = Xmm
  flr$ybin = ybin
  flr$variables_retained = variables_retained
  if (drop_collinear_variables){
	flr$collinear_variables = collinear_variables
	coefs = flr$coefficients #save originals
	flr$coefficients = array(NA, p)
	flr$coefficients[variables_retained] = coefs #all dropped variables will be NA's
  }
  names(flr$coefficients) = original_col_names
  flr$original_regressor_names = original_col_names
  flr$rank = ncol(Xmm)
  flr$deviance = -2 * flr$loglikelihood 
  flr$aic = flr$deviance + 2 * flr$rank
  flr$do_inference_on_var = do_inference_on_var #pass back to the user which variables, if could even be none at this point after collinear variables were dropped

  if (any(do_inference_on_var)){
	  b = flr$coefficients[variables_retained]  
	  
	  flr$se = 						array(NA, p)
	  flr$z = 						array(NA, p)
	  flr$approx_pval = 			array(NA, p)
	  names(flr$se) =   			original_col_names
	  names(flr$z) =   				original_col_names
	  names(flr$approx_pval) =   	original_col_names
	  
	  #compute the std errors of the coefficient estimators 
	  #we compute them via notes found in https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture26.pdf
	  exp_Xmm_dot_b = exp(Xmm %*% b)
	  w = as.numeric(exp_Xmm_dot_b / (1 + exp_Xmm_dot_b)^2)
	  XmmtWmatXmm = eigen_Xt_times_diag_w_times_X(Xmm, w, num_cores) #t(Xmm) %*% diag(w) %*% Xmm
	  
	  if (sum(do_inference_on_var) > 2){ #this seems to be the cutoff in simulations...
		  tryCatch({ #compute the entire inverse (this could probably be sped up by only computing the diagonal a la https://web.stanford.edu/~lexing/diagonal.pdf but I have not found that implemented anywhere)
			  XmmtWmatXmminv = eigen_inv(XmmtWmatXmm, num_cores)
		  }, 
		  error = function(e){
			  print(e)
			  stop("Error in inverting X^T X.\nTry setting drop_collinear_variables = TRUE\nto automatically drop perfectly collinear variables.\n")
		  })
		  
		  flr$se[variables_retained] = sqrt(diag(XmmtWmatXmminv))
	  } else { #only compute the few entries of the inverse that are necessary. This could be sped up using https://math.stackexchange.com/questions/64420/is-there-a-faster-way-to-calculate-a-few-diagonal-elements-of-the-inverse-of-a-h
		  sqrt_det_XmmtWmatXmm = sqrt(eigen_det(XmmtWmatXmm, num_cores))
		  for (j in which(variables_retained)){
			  flr$se[j] = sqrt(eigen_det(XmmtWmatXmm[-j, -j, drop = FALSE], num_cores)) / sqrt_det_XmmtWmatXmm
		  }
	  }

	  flr$z[variables_retained] = 				b / flr$se[variables_retained]
	  flr$approx_pval[variables_retained] = 	2 * pnorm(-abs(flr$z[variables_retained]))
  }

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
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression(
#' 	Xmm = model.matrix(~ . - type, Pima.te), 
#'  ybin = as.numeric(Pima.te$type == "Yes"))
#' summary(flr)
summary.fast_logistic_regression = function(object, ...){
  checkmate::assert_choice(class(object), c("fast_logistic_regression", "fast_logistic_regression_stepwise"))
  if (!object$converged){
      warning("fast LR did not converge")
  }
  if (!any(object$do_inference_on_var)){
	  cat("please refit the model with the \"do_inference_on_var\" argument set to true.\n")
  } else {
	  df = data.frame(
	    approx_coef = object$coefficients,
	    approx_se = object$se,
	    approx_z = object$z,
	    approx_pval = object$approx_pval,
	    signif = ifelse(is.na(object$approx_pval), "", ifelse(object$approx_pval < 0.001, "***", ifelse(object$approx_pval < 0.01, "**", ifelse(object$approx_pval < 0.05, "*", ""))))
	  )
	  rownames(df) = object$original_regressor_names
	  df
  }
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
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression_stepwise_forward(
#' 	Xmm = model.matrix(~ . - type, Pima.te), 
#'  ybin = as.numeric(Pima.te$type == "Yes"))
#' summary(flr)
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
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression(
#' 	Xmm = model.matrix(~ . - type, Pima.te), 
#'  ybin = as.numeric(Pima.te$type == "Yes"))
#' print(flr)
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
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression_stepwise_forward(
#' 	Xmm = model.matrix(~ . - type, Pima.te), 
#'  ybin = as.numeric(Pima.te$type == "Yes"))
#' print(flr)
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
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression(
#'   Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = as.numeric(Pima.te$type == "Yes")
#' )
#' phat = predict(flr, model.matrix(~ . - type, Pima.te))
predict.fast_logistic_regression = function(object, newdata, type = "response", ...){
  checkmate::assert_class(object, "fast_logistic_regression")
  assert_numeric_matrix(newdata)
  checkmate::assert_choice(type, c("link", "response"))
  
  #if new_data has more features than training data, we can subset it
  old_data_features = object$original_regressor_names
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
  if (!all(colnames(newdata) == old_data_features)){
    stop("newdata must have same columns as the original training data matrix in the same order.\nHere, training data has features\n", paste(new_features_minus_old_features, collapse = ", "), "\nwhich newdata did not have")
  }
  if (!object$converged){
    warning("fast LR did not converge")
  }
  b = object$coefficients
  b[is.na(b)] = 0 #this is the way to ignore NA's
  log_odds_predictions = c(newdata %*% b)
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
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression_stepwise_forward(
#'   Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = as.numeric(Pima.te$type == "Yes")
#' )
#' phat = predict(flr, model.matrix(~ . - type, Pima.te))
predict.fast_logistic_regression_stepwise = function(object, newdata, type = "response", ...){	
	checkmate::assert_class(object, "fast_logistic_regression_stepwise")
	predict.fast_logistic_regression(object$flr, newdata, type = "response", ...)
}

#' Rapid Forward Stepwise Logistic Regression
#' 
#' Roughly duplicates the following \code{glm}-style code:
#' 
#'  \code{nullmod = glm(ybin ~ 0,     data.frame(Xmm), family = binomial)}
#'  \code{fullmod = glm(ybin ~ 0 + ., data.frame(Xmm), family = binomial)}
#'  \code{forwards = step(nullmod, scope = list(lower = formula(nullmod), upper = formula(fullmod)), direction = "forward", trace = 0)}
#'
#' @param Xmm             			The model.matrix for X (you need to create this yourself before).
#' @param ybin            			The binary response vector.
#' @param mode						"aic" (default, fast) or "pval" (slow, but possibly yields a better model).
#' @param pval_threshold  			The significance threshold to include a new variable. Default is \code{0.05}.
#' 									If \code{mode == "aic"}, this argument is ignored.
#' @param use_intercept   			Should we automatically begin with an intercept? Default is \code{TRUE}.
#' @param drop_collinear_variables 	Parameter used in \code{fast_logistic_regression}. Default is \code{FALSE}. See documentation there.
#' @param lm_fit_tol	  			Parameter used in \code{fast_logistic_regression}. Default is \code{1e-7}. See documentation there.
#' @param verbose         			Print out messages during the loop? Default is \code{TRUE}.
#' @param ...             			Other arguments to be passed to \code{fastLR}. See documentation there.
#'
#' @return                			A list of raw results
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression_stepwise_forward(
#'   Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = as.numeric(Pima.te$type == "Yes")
#' )
fast_logistic_regression_stepwise_forward = function(
		Xmm, 
		ybin, 
		mode = "aic",
		pval_threshold = 0.05, 
		use_intercept = TRUE, 
		verbose = TRUE, 
		drop_collinear_variables = FALSE, 
		lm_fit_tol = 1e-7, 
		...){
  assert_numeric_matrix(Xmm)
  ybin = assert_binary_vector_then_cast_to_numeric(ybin)
  if (length(ybin) != nrow(Xmm)){
    stop("The number of rows in Xmm must be equal to the length of ybin")
  }
  assert_choice(mode, c("aic", "pval"))
  mode_is_aic = (mode == "aic")
  if (!mode_is_aic){
	  assert_numeric(pval_threshold, lower = .Machine$double.eps, upper = 1 - .Machine$double.eps)
  }
  
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
  if (mode_is_aic){
	  aics_star = c()
	  last_aic_star = .Machine$double.xmax #anything will beat this
  } else {
	  pvals_star = c()
  }
  
  repeat {
    js_to_try = setdiff(1 : p, js)
    if (length(js_to_try) == 0){
      break
    }
	if (mode_is_aic){
		aics = array(NA, p)
	} else {
		pvals = array(NA, p)	
	}
    
    for (i_j in 1 : length(js_to_try)){
      j = js_to_try[i_j]
      Xmmtemp = Xmmt
      Xmmtemp = cbind(Xmmtemp, Xmm[, j, drop = FALSE])
      # tryCatch({
		ptemp = ncol(Xmmtemp)
		do_inference_on_var = 	if (mode_is_aic){
									FALSE
								} else {
									c(rep(FALSE, ptemp - 1), TRUE)
								}
        flrtemp = fast_logistic_regression(Xmmtemp, ybin, drop_collinear_variables, lm_fit_tol, do_inference_on_var = do_inference_on_var)
		if (mode_is_aic){
			aics[j] = flrtemp$aic
		} else {
			if (!is.null(flrtemp$approx_pval)){ #if the last variable got dropped due to collinearity, we skip this
				pvals[j] = flrtemp$approx_pval[ptemp] #the last one
			}			
		}
	
		if (verbose){
			cat("   sub iteration #", i_j, "of", length(js_to_try), "with feature", colnames(Xmm)[j], "resulted in ")
			if (mode_is_aic){
				cat("aic", aics[j], "\n")
			} else {
				cat("pval", pvals[j], "\n")	
			}
			
		}
      # }, error = function(e){
      #   cat("   iter #", i_j, "of", length(js_to_try), "with feature", colnames(Xmm)[j], "resulted in ERROR\n")
      # })
    }
	if (mode_is_aic){
		if (min(aics, na.rm = TRUE) > last_aic_star){
			break
		}
	} else {
		if (!any(pvals < pval_threshold, na.rm = TRUE)){
			break
		}
	}

	j_star = 	if (mode_is_aic){
					which.min(aics)
				} else {
					which.min(pvals)
				}
	
	#if (is.na(j_star) | is.null(j_star) | is.na(aics[j_star])){
	#	stop("j_star problem")
	#}
    js = c(js, j_star)
	if (mode_is_aic){
		aics_star = c(aics_star, aics[j_star])
		last_aic_star = aics[j_star]
	} else {
		pvals_star = c(pvals_star, pvals[j_star])
	}	
    
    Xmmt = cbind(Xmmt, Xmm[, j_star, drop = FALSE])
    
    iter = iter + 1
    if (verbose){
      cat("iteration #", iter, "of possibly", p, "added feature #", j_star, "named", colnames(Xmm)[j_star], "with ")
	  if (mode_is_aic){
		  cat("aic", aics[j_star], "\n")
	  } else {
		  cat("pval", pvals[j_star], "\n")
	  }					  
    }
  }
  #return some information you would like to see
  flr_stepwise = list(js = js, flr = fast_logistic_regression(Xmmt, ybin, drop_collinear_variables, lm_fit_tol, do_inference_on_var = TRUE))
  if (mode_is_aic){
	  flr_stepwise$aics = aics
  } else {
	  flr_stepwise$pvals_star = pvals_star
  }  
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
#' library(MASS); data(Pima.te)
#' ybin = as.numeric(Pima.te$type == "Yes")
#' flr = fast_logistic_regression(
#'   Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = ybin
#' )
#' phat = predict(flr, model.matrix(~ . - type, Pima.te))
#' confusion_results(phat > 0.5, ybin)
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
#' library(MASS); data(Pima.te)
#' ybin = as.numeric(Pima.te$type == "Yes")
#' flr = fast_logistic_regression(
#'   Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = ybin
#' )
#' phat = predict(flr, model.matrix(~ . - type, Pima.te))
#' yhat = array(NA, length(ybin))
#' yhat[phat <= 1/3] = "no"
#' yhat[phat >= 2/3] = "yes"
#' yhat[is.na(yhat)] = "maybe"
#' general_confusion_results(factor(yhat, levels = c("no", "yes", "maybe")), factor(ybin)) 
#' #you want the "no" to align with 0, the "yes" to align with 1 and the "maybe" to be 
#' #last to align with nothing
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

#' A fast Xt [times] diag(w) [times] X function
#' 
#' Via the eigen package
#' 
#' @param X					A numeric matrix of size n x p 
#' @param w 				A numeric vector of length p
#' @param num_cores 		The number of cores to use. Unless p is large, keep to the default of 1.
#' 
#' @return					The resulting matrix 
#' 
#' @useDynLib 				fastLogisticRegressionWrap, .registration=TRUE
#' @export
#' @examples
#'   n = 100
#'   p = 10
#'   X = matrix(rnorm(n * p), nrow = n, ncol = p)
#'   w = rnorm(p)
#'   eigen_Xt_times_diag_w_times_X(t(X), w)
eigen_Xt_times_diag_w_times_X = function(X, w, num_cores = 1){
	assert_numeric_matrix(X)
	assert_numeric(w)
	assert_true(nrow(X) == length(w))
	assert_count(num_cores, positive = TRUE)
#	if (!exists("eigen_Xt_times_diag_w_times_X_cpp", envir = fastLogisticRegressionWrap_globals)){
#		eigen_Xt_times_diag_w_times_X_cpp = Rcpp::cppFunction(depends = "RcppEigen", '					
#		Eigen::MatrixXd eigen_Xt_times_diag_w_times_X_cpp(const Eigen::Map<Eigen::MatrixXd> X, const Eigen::Map<Eigen::VectorXd> w, int n_cores) {
#			Eigen::setNbThreads(n_cores);
#			return X.transpose() * w.asDiagonal() * X;
#		}
#		')
#		assign("eigen_Xt_times_diag_w_times_X_cpp", eigen_Xt_times_diag_w_times_X_cpp, fastLogisticRegressionWrap_globals)
#	}
#	eigen_Xt_times_diag_w_times_X_cpp = get("eigen_Xt_times_diag_w_times_X_cpp", fastLogisticRegressionWrap_globals)
	eigen_Xt_times_diag_w_times_X_cpp(X, w, num_cores)
}

#' A fast solve(X) function
#' 
#' Via the eigen package
#' 
#' @param X					A numeric matrix of size p x p
#' @param num_cores 		The number of cores to use. Unless p is large, keep to the default of 1.
#' 
#' @return					The resulting matrix 
#' 
#' @useDynLib 				fastLogisticRegressionWrap, .registration=TRUE
#' @export
#' @examples
#'   p = 10
#'   eigen_inv(matrix(rnorm(p^2), nrow = p))
eigen_inv = function(X, num_cores = 1){
	assert_numeric_matrix(X)
	assert_true(ncol(X) == nrow(X))
	assert_count(num_cores, positive = TRUE)
#	if (!exists("eigen_inv_cpp", envir = fastLogisticRegressionWrap_globals)){
#		eigen_inv_cpp = Rcpp::cppFunction(depends = "RcppEigen", '
#			Eigen::MatrixXd eigen_inv_cpp(const Eigen::Map<Eigen::MatrixXd> X, int n_cores) {
#			Eigen::setNbThreads(n_cores);
#			return X.inverse();
#		}
#		')
#		assign("eigen_inv_cpp", eigen_inv_cpp, fastLogisticRegressionWrap_globals)
#	}
#	eigen_inv_cpp = get("eigen_inv_cpp", fastLogisticRegressionWrap_globals)
	eigen_inv_cpp(X, num_cores)
}

#' A fast det(X) function
#' 
#' Via the eigen package
#' 
#' @param X					A numeric matrix of size p x p
#' @param num_cores 		The number of cores to use. Unless p is large, keep to the default of 1.
#' 
#' @return					The determinant as a scalar numeric value
#' 
#' @useDynLib 				fastLogisticRegressionWrap, .registration=TRUE
#' @export
#' @examples
#'   p = 30
#'   eigen_det(matrix(rnorm(p^2), nrow = p))
eigen_det = function(X, num_cores = 1){
	assert_numeric_matrix(X)
	assert_true(ncol(X) == nrow(X))
	assert_count(num_cores, positive = TRUE)
#	if (!exists("eigen_det_cpp", envir = fastLogisticRegressionWrap_globals)){
#		eigen_det_cpp = Rcpp::cppFunction(depends = "RcppEigen", '					
#		double eigen_det_cpp(const Eigen::Map<Eigen::MatrixXd> X, int n_cores) {
#			Eigen::setNbThreads(n_cores);
#			return X.determinant();
#		}
#		')
#		assign("eigen_det_cpp", eigen_det_cpp, fastLogisticRegressionWrap_globals)
#	}
#	eigen_det_cpp = get("eigen_det_cpp", fastLogisticRegressionWrap_globals)
	eigen_det_cpp(X, num_cores)
}