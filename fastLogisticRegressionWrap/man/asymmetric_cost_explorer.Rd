% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/code.R
\name{asymmetric_cost_explorer}
\alias{asymmetric_cost_explorer}
\title{Asymmetric Cost Explorer}
\usage{
asymmetric_cost_explorer(
  phat,
  ybin,
  steps = seq(from = 0.001, to = 0.999, by = 0.001),
  outcome_of_analysis = 0,
  proportions_desired = seq(from = 0.1, to = 0.9, by = 0.1),
  proportion_tolerance = 0.01
)
}
\arguments{
\item{phat}{The vector of probability estimates to be thresholded to make a binary decision}

\item{ybin}{The true binary responses}

\item{steps}{All possibile thresholds which must be a vector of numbers in (0, 1). Default is \code{seq(from = 0.001, to = 0.999, by = 0.001)}.}

\item{outcome_of_analysis}{Which class do you care about performance? Either 0 or 1 for the negative class or positive class. Default is \code{0}.}

\item{proportions_desired}{Which proportions of \code{outcome_of_analysis} class do you wish to understand performance for?}

\item{proportion_tolerance}{If the model cannot match the proportion_desired within this amount, it does not return that model's performance. Default is \code{0.01}.}

\item{K_folds}{If not \code{NULL}, this indicates that we wish to fit the \code{phat} thresholds out of sample using this number of folds. Default is \code{NULL} for in-sample fitting.}
}
\value{
A table with column 1: \code{proportions_desired}, column 2: actual proportions (as close as possible), column 3: error rate, column 4: probability threshold.
}
\description{
Given a set of desired proportions of predicted outcomes, what is the error rate for each of those models?
}
\author{
Adam Kapelner
}
