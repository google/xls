#!/usr/bin/env -S bash -ex

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
  TEST_TARGET_PATTERN="$@"
  UPDATE_TARGET_PATTERN="$@"
else
  # Keep-going and ignore failures as the query hits irrelevant errors in OSS.
  TEST_TARGET_PATTERN="(rdeps($XLS_TARGET_PATTERN,//xls/common:golden_files) + rdeps($XLS_TARGET_PATTERN,//xls/common:golden_files_py)) except ($XLS_TARGET_EXCLUDE)"
  UPDATE_TARGET_PATTERN="$XLS_TARGET_PATTERN except (attr(tags, \"no_update_golden\", $XLS_TARGET_PATTERN) + $XLS_TARGET_EXCLUDE)"
fi

TEST_TARGETS=($(bazel query "kind('(py_test|cc_test)', $TEST_TARGET_PATTERN)" --keep_going || true))

# Update pre-computed estimator protos before building/testing TEST_TARGETS and
# OTHER_RUN_TARGETS: C++ delay/area model libraries embed the checked-in
# precomputed protos, so updating them first ensures downstream golden tests
# (e.g. scheduling/codegen golden files) regenerate against the new models in a
# single pass rather than against stale precomputed protos.
PRECOMPUTED_UPDATE_QUERY="filter('.*_precomputed_up_to_date_test_update_golden$', kind(_xls_update_golden, $UPDATE_TARGET_PATTERN))"
# Run frozen file sha256 updates first so downstream consumers of the frozen files can build.
FROZEN_UPDATE_QUERY="attr(target_name, '.*_frozen$', kind(_xls_update_sha256, $UPDATE_TARGET_PATTERN))"
PRECOMPUTED_UPDATE_TARGETS=($(bazel query "$PRECOMPUTED_UPDATE_QUERY" --keep_going || true))
FROZEN_UPDATE_TARGETS=($(bazel query "$FROZEN_UPDATE_QUERY" --keep_going || true))
OTHER_RUN_TARGETS=($(bazel query "(kind(_xls_update_golden, $UPDATE_TARGET_PATTERN) + kind(_xls_update_sha256, $UPDATE_TARGET_PATTERN)) except ($FROZEN_UPDATE_QUERY + $PRECOMPUTED_UPDATE_QUERY)" --keep_going || true))
RUN_TARGETS=("${FROZEN_UPDATE_TARGETS[@]}" "${OTHER_RUN_TARGETS[@]}")

if [[ ! -f "$(pwd)/WORKSPACE" ]]
then
  echo "Must be run from root repo directory"
  exit 1
fi

for target in "${PRECOMPUTED_UPDATE_TARGETS[@]}"; do
  bazel run -c opt "$target"
done

if [[ ${#TEST_TARGETS[@]} -gt 0 ]]; then
  # Some dependencies do not build properly with --spawn_strategy=standalone so
  # build the targets normally first.
  bazel build -c opt --keep_going ${TEST_TARGETS[@]}

  bazel test -c opt \
    --test_strategy=standalone \
    --spawn_strategy=standalone \
    ${TEST_TARGETS[@]} \
    --test_arg=--test_update_golden_files \
    --test_arg=--xls_source_dir="$(pwd)"/xls/ \
    --test_arg=--alsologtostderr \
    --nocache_test_results \
    --test_output=errors || true
fi

# bazel run can't run multiple targets
# TODO(allight): It would be nice to run these all in parallel.
for target in "${RUN_TARGETS[@]}"; do
  bazel run -c opt "$target"
done
