# Changelog

## fastLogisticRegressionWrap 1.2.2

- Made `assert_binary_vector_then_cast_to_numeric()` faster, improving
  regression performance.

## fastLogisticRegressionWrap 1.2.1

- Added a function that returns desired asymmetric-cost models fitted
  out of sample.

## fastLogisticRegressionWrap 1.2.0

CRAN release: 2023-08-08

- Added a function that returns desired asymmetric-cost models.
- Added alphabetical ordering of coefficients to the inference summary
  function.
- Sped up
  [`confusion_results()`](https://kapelner.github.io/fastLogisticRegressionWrap/reference/confusion_results.md)
  with a custom C++ table routine and added an option to skip argument
  checks.
- Fixed inference-disabled operation.
- Fixed `proportions_scaled_by_column = TRUE` in
  [`general_confusion_results()`](https://kapelner.github.io/fastLogisticRegressionWrap/reference/general_confusion_results.md).

## fastLogisticRegressionWrap 1.1.0

- Added support for custom matrix-multiplication and
  square-root-diagonal-of-inverse functions, enabling GPU-backed
  computation (see the README for an example).
- Added conjugate-gradient inference for a single coefficient, which can
  be faster than inference for all coefficients in some settings.

## fastLogisticRegressionWrap 1.0.1

CRAN release: 2023-07-08

- Removed the OpenMP dependency for CRAN compatibility.

## fastLogisticRegressionWrap 1.0.0

CRAN release: 2023-07-07

- Initial release.
