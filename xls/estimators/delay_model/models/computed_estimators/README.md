# Precomputed Estimator Models

This directory contains precomputed `CompiledEstimatorModel` textprotos with the
curve-fitted estimator parameters for the estimator models in the parent
`models/` directory, allowing C++ targets to embed and evaluate the models at
runtime without depending on Python curve-fitting tools (`scipy`) at build time.

**All `.textproto` files in this directory are automatically generated and
should not be edited manually.** To regenerate them after modifying a source
estimator model, run `rebuild_golden_files.sh`.
