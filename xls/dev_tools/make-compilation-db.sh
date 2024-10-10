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
[ -r "${COMPDB_SCRIPT}" ] || bazel fetch --noshow_progress @com_grail_bazel_compdb//...

python3 "${COMPDB_SCRIPT}"

# Massage the output so that clang-tidy fully undestands the compile commands:
# remove flags where it gets confused.
# Also, make sure that the command also contains -xc++ (bazel sometimes does
# not add that in libraries that don't have a *.cc file but only *.h or *.inc)
sed -i compile_commands.json -f - <<EOF
s/"command": "\([^ ]*\) /"command": "\1 -x c++ /  # -xc++ as first argument.
s/ -f[^ ]*/ /g             # remove all -fxyz options
s/ --target[^ ]*/ /g       # target platform not needed, might confuse
s/ -stdlib=libc++/ /       # otherwise, clang-tidy does not find c++ headers
EOF
