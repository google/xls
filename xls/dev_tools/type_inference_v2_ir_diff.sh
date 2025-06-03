#!/usr/bin/env bash
# Copyright 2025 The XLS Authors
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

# This script checks whether the input XLS file (or all found XLS files when the
# input is a directory) compiles to the same IR using type system v1 and v2.

bazel build -c opt //xls/dslx/ir_convert:ir_converter_main
bazel build -c opt //xls/dev_tools:remove_identifiers_main
WORKSPACE_ROOT=$(p4 g4d)
for i in $(find $1 -name "*.x" | sort);
do
  fail=0
  echo "Checking" $i
  a=$($WORKSPACE_ROOT/bazel-bin/xls/dslx/ir_convert/ir_converter_main $i 2> >(head -n 5))
  if (( $? )); then
    echo "$a"
    echo "File failed to compile under type system v1"
    fail=1
  fi
  b=$($WORKSPACE_ROOT/bazel-bin/xls/dslx/ir_convert/ir_converter_main $i --type_inference_v2 2> >(head -n 5))
  if (( $? )); then
    echo "$b"
    echo "File failed to compile under type system v2"
    fail=1
  fi
  if (( !fail )); then
    diff <($WORKSPACE_ROOT/bazel-bin/xls/dev_tools/remove_identifiers_main <(echo $a)) <($WORKSPACE_ROOT/bazel-bin/xls/dev_tools/remove_identifiers_main <(echo $b))
    if (( !$? )); then
      printf "\033[0;32mOK\033[0m\n"
    else
      fail=1
    fi
  fi
  if (( fail )); then
    printf "\033[0;31mFail\033[0m\n"
  fi
  echo
done