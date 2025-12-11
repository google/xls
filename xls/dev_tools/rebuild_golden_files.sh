#!/bin/bash -ex

# Rebuilds the golden files of all tests using
# `xls/common/golden_files.h` infrastructure.
#
# Run with no arguments will rebuild golden files for all such targets. Can also
# be called with a list of test targets to rebuild.

# TODO(allight): Include more stuff here.
XLS_TARGET_PATTERN="//xls/..."
XLS_TARGET_EXCLUDE="//xls/common:golden_files_test"

if [[ "$@" ]]
then
  TEST_TARGETS=($(bazel query "kind(rule, $@)" --output=label_kind --keep_going | /bin/grep -E '^cc_test' | cut -f3 -d ' ' || /bin/true))
  RUN_TARGETS=($(bazel query "kind(rule, $@)" --output=label_kind --keep_going | /bin/grep -E '^_xls_update_golden' | cut -f3 -d ' ' || /bin/true))
else
  # Keep-going and ignore failures as the query hits irrelevant errors in OSS.
  TEST_TARGETS=($(bazel query "kind(cc_test, rdeps($XLS_TARGET_PATTERN,//xls/common:golden_files) except ($XLS_TARGET_EXCLUDE))" --keep_going || /bin/true))
  RUN_TARGETS=($(bazel query "kind(_xls_update_golden, $XLS_TARGET_PATTERN) except ($XLS_TARGET_EXCLUDE)" --keep_going || /bin/true))
fi

if [[ ! -f "$(pwd)/WORKSPACE" ]]
then
  echo "Must be run from root repo directory"
  exit 1
fi

# Some dependencies do not build properly with --spawn_strategyy=standalone so
# build the targets normally first.
bazel build -c opt ${TEST_TARGETS[@]} ${RUN_TARGETS[@]}

bazel test -c opt \
  --test_strategy=standalone \
  --spawn_strategy=standalone \
  ${TEST_TARGETS[@]} \
  --test_arg=--test_update_golden_files \
  --test_arg=--xls_source_dir="$(pwd)"/xls/ \
  --test_arg=--alsologtostderr \
  --nocache_test_results \
  --test_output=errors

bazel run -c opt ${RUN_TARGETS[@]}