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

# Invocation without parameters simply uses the .clang-tidy config to run on
# all *.{cc,h} files. Additional parameters passed to this script are passed
# to clang-tidy as-is. Typical use could be for instance
#   run-clang-tidy-cached.sh --checks="-*,modernize-use-override" --fix

# We have to first build the binary, then build the compilation-db and
# must not call "bazel run" below ... otherwise bazel will remove some
# of the symbolic links to external dependencies that we carefully
# establish with the compilation-db build.
bazel build dev_tools:run_clang_tidy_cached

$(dirname $0)/make-compilation-db.sh

# Use either CLANG_TIDY provided by the user as environment variable or use
# our own from the toolchain we configured in the WORKSPACE.
export CLANG_TIDY="${CLANG_TIDY:-$(bazel info output_base)/external/llvm_toolchain_llvm/bin/clang-tidy}"

exec bazel-bin/dev_tools/run_clang_tidy_cached "$@"
