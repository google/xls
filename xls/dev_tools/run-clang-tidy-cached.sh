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

set -u

# Build clang-tidy runner; making sure to use the same -c opt flag
# as make-compilation-db.sh will use so that the bazel symbolic links
# are not switched around.
bazel build -c opt //xls/dev_tools:run_clang_tidy_cached

# Use either CLANG_TIDY provided by the user as environment variable or use
# our own from the toolchain we configured in the MODULE.bazel
export CLANG_TIDY=${CLANG_TIDY:-$(bazel run -c opt --run_under="echo" @llvm_toolchain//:clang-tidy 2>/dev/null)}

# Refresh compilation DB
$(dirname $0)/make-compilation-db.sh

exec bazel-bin/xls/dev_tools/run_clang_tidy_cached "$@"
