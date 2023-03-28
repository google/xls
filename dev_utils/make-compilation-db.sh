#!/usr/bin/env bash
# Copyright 2023 The XLS Authors.
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

set -u
set -e

readonly OUTPUT_BASE="$(bazel info output_base)"

readonly COMPDB_SCRIPT="${OUTPUT_BASE}/external/com_grail_bazel_compdb/generate.py"
[ -r "${COMPDB_SCRIPT}" ] || bazel fetch ...

python3 "${COMPDB_SCRIPT}"

# Remove a gcc compiler flag that clang-tidy doesn't understand.
sed -i -e 's/-fno-canonical-system-headers//g' compile_commands.json
