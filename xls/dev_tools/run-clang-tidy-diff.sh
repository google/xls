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

# Run clang-tidy-diff that we already have bundled in our toolchain.
# Reads a unified diff from stdin. If a header is removed, we want to get
# enough context to see that it is not used anywhere; so invoke like
#
# git diff -U1000 | xls/dev_tools/run-clang-tidy-diff.sh -p1 2>/dev/null

CLANG_TIDY_LLVM_BASE="$(bazel info output_base)/external/llvm_toolchain_llvm/"

"${CLANG_TIDY_LLVM_BASE}/share/clang/clang-tidy-diff.py" \
  -clang-tidy-binary="${CLANG_TIDY_LLVM_BASE}/bin/clang-tidy" "$@"
