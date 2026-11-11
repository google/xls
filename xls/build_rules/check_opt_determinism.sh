#!/bin/bash
# Copyright 2026 The XLS Authors
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

set -euo pipefail

# The first argument is the path to the opt_main binary provided by $(rootpath).
# The second argument is the path to the input IR file provided by $(rootpath).
# Any remaining arguments are passed as flags to opt_main.
readonly BINARY_PATH="$1"
readonly IR_PATH="$2"
shift 2

OUTPUT_1=$(mktemp "${TEST_TMPDIR:-/tmp}/opt_out_1.XXXXXX")
OUTPUT_2=$(mktemp "${TEST_TMPDIR:-/tmp}/opt_out_2.XXXXXX")
readonly OUTPUT_1
readonly OUTPUT_2

# Run the binary twice.
"${BINARY_PATH}" "${IR_PATH}" "$@" > "${OUTPUT_1}"
"${BINARY_PATH}" "${IR_PATH}" "$@" > "${OUTPUT_2}"

# Compare outputs
if cmp -s "${OUTPUT_1}" "${OUTPUT_2}"; then
  echo "SUCCESS: Outputs are deterministic."
  exit 0
else
  echo "FAIL: Outputs differ!"
  diff -u "${OUTPUT_1}" "${OUTPUT_2}"
  exit 1
fi
