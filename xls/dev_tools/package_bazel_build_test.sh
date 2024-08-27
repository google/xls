#!/usr/bin/env bash
# Copyright 2020 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TOOLPATH=./xls/dev_tools

echo "Test Tmp Dir: ${TEST_TMPDIR}"
echo "Test Files:"
find ./

# Copy testdata into tmp directory
TESTDATA_DIR="${TEST_TMPDIR}/testdata"
cp -Lr $TOOLPATH/package_bazel_build_testdata ${TESTDATA_DIR}

# Process fake manifest and turn into absolute paths
chmod +w $TESTDATA_DIR/bazel-bin
sed -i "s,{x},$TESTDATA_DIR,g" \
  $TESTDATA_DIR/bazel-bin/package_test.runfiles_manifest

echo "Test Data Dir: ${TESTDATA_DIR}"
find ${TESTDATA_DIR}

echo "Final Manifest:"
cat $TESTDATA_DIR/bazel-bin/package_test.runfiles_manifest

# Run package utility
BINPATH=./xls/dev_tools/package_bazel_build
$BINPATH \
--bazel_bin $TESTDATA_DIR/bazel-bin \
--bazel_execroot $TESTDATA_DIR/hash/execroot/com_google_xls \
--inc_target package_test \
--output_dir $TEST_TMPDIR/out \
--v 1

# Validate output
ERR=0

test_output () {
  sh_script=$1
  expect=$2

  output=`$sh_script`
  if [ "$output" != "$expect" ]; then
    echo "ERROR: Expected $sh_script output to be \"$expect\", got \"$output\""
    ERR=1
  fi
}

test_output "$TEST_TMPDIR/out/package_test" \
  "Hello World"
test_output  "$TEST_TMPDIR/out/package_test.runfiles/copied_dep.sh" \
  "Copied Dependency #0"
test_output "$TEST_TMPDIR/out/package_test.runfiles/c/copied_dep.sh" \
  "Copied Dependency #1"
test_output "$TEST_TMPDIR/out/package_test.runfiles/e/execroot_dep_1.sh" \
  "Execroot Dependency"
test_output "$TEST_TMPDIR/out/package_test.runfiles/e/execroot_dep_0.sh" \
  "Execroot Dependency"
test_output "$TEST_TMPDIR/out/package_test.runfiles/s/src_dep.sh" \
  "Source Dependency"
test_output "$TEST_TMPDIR/out/package_test.runfiles/o/output_root.sh" \
  "Output Root Dependency"

if [ "$ERR" -eq 0 ]; then
  echo "PASS"
else
  echo "FAIL"
fi

exit $ERR
