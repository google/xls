#!/bin/sh -ex

# Rebuilds the golden files of all tests using
# `xls/common/golden_files.h` infrastructure.
#
# Run with no arguments will rebuild golden files for all such targets. Can also
# be called with a list of test targets to rebuild.

if [[ "$@" ]]
then
  TARGETS=($@)
else
  TARGETS=($(bazel query "kind(cc_test, rdeps(//xls/..., //xls/common:golden_files))"))
fi

if [[ ! -f "$(pwd)/WORKSPACE" ]]
then
  echo "Must be run from root repo directory"
  exit 1
fi

# Some dependencies do not build properly with --spawn_strategyy=standalone so
# build the targets normally first.
bazel build -c opt ${TARGETS[@]}

bazel test -c opt \
  --test_strategy=standalone \
  --spawn_strategy=standalone \
  ${TARGETS[@]} \
  --test_arg=--test_update_golden_files \
  --test_arg=--xls_source_dir="$(pwd)"/xls/ \
  --test_arg=--alsologtostderr \
  --nocache_test_results \
  --test_output=errors
